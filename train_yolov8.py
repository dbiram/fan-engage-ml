#!/usr/bin/env python3
import argparse, os, sys, time, json
import time
from pathlib import Path
from ultralytics import YOLO
from minio import Minio
from torch.serialization import add_safe_globals
try:
    # Ultralytics detection model class used in checkpoints
    from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel
    add_safe_globals([DetectionModel, SegmentationModel, PoseModel])
except Exception:
    # Fallback: if only DetectionModel exists in this version
    try:
        from ultralytics.nn.tasks import DetectionModel
        add_safe_globals([DetectionModel])
    except Exception:
        pass

def parse_args():
    ap = argparse.ArgumentParser("Fine-tune YOLOv8 on football dataset")
    ap.add_argument("--data", required=True, help="path to data.yaml inside container")
    ap.add_argument("--model", default="yolov8n.pt", help="base model")
    ap.add_argument("--imgsz", type=int, default=576)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolov8n_football")
    ap.add_argument("--project", default="/artifacts/runs")
    ap.add_argument("--device", default="cpu")  # "cuda:0" if GPU available
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--upload", action="store_true", help="upload artifacts to MinIO")
    ap.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT","minio:9000"))
    ap.add_argument("--minio-access", default=os.getenv("MINIO_ROOT_USER","minio_admin"))
    ap.add_argument("--minio-secret", default=os.getenv("MINIO_ROOT_PASSWORD","minio123"))
    ap.add_argument("--minio-secure", action="store_true")
    ap.add_argument("--minio-bucket", default=os.getenv("BUCKET_MODELS","models"))
    ap.add_argument("--artifact-prefix", default="yolov8n_football")
    ap.add_argument("--amp", type=int, default=1, help="1=enable AMP, 0=disable")
    return ap.parse_args()

def upload(minio_client: Minio, bucket: str, local_path: Path, object_key: str):
    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)
    minio_client.fput_object(bucket, object_key, str(local_path))

def main():
    args = parse_args()
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{run_ts}"

    # Train
    model = YOLO(args.model)
    
    _epoch_times = {}
    def on_train_epoch_start(trainer):
        _epoch_times[trainer.epoch] = time.time()

    # Create callback to print metrics
    def on_train_epoch_end(trainer):
        m = getattr(trainer, "metrics", {}) or {}

        # ---- compute epoch duration ----
        start = _epoch_times.get(trainer.epoch, None)
        elapsed = None
        if start is not None:
            elapsed = time.time() - start

        # Last-batch train losses are in trainer.loss_items for detection models (box, cls, dfl)
        train_loss_items = getattr(trainer, "loss_items", None)

        print(f"\nEpoch {trainer.epoch + 1}/{trainer.epochs}")
        if elapsed is not None:
            print(f"Epoch time: {elapsed:.2f} sec")

        if train_loss_items is not None and len(train_loss_items) >= 3:
            box, cls, dfl = map(float, train_loss_items[:3])
            print(f"Train Loss (last batch) -> box: {box:.3f} | cls: {cls:.3f} | dfl: {dfl:.3f}")
        else:
            # Fallback if structure changes
            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                try:
                    print(f"Train Loss (last batch): {float(tloss):.3f}")
                except Exception:
                    print("Train Loss: n/a")
            else:
                print("Train Loss: n/a")

        # Validation metrics (present after val runs)
        def fmt(key):
            v = m.get(key, None)
            try:
                return f"{float(v):.3f}" if v is not None else "n/a"
            except Exception:
                return "n/a"

        # Common YOLOv8 val keys
        print(f"Val box loss: {fmt('val/box_loss')}")
        print(f"mAP50: {fmt('metrics/mAP50(B)')}")
        print(f"mAP50-95: {fmt('metrics/mAP50-95(B)')}")
        print("-" * 50)
    
    # Add callback to model
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=run_name,
        project=args.project,
        patience=100,
        cos_lr=False,
        optimizer="auto",
        deterministic=True,
        amp=bool(args.amp),
        pretrained=True,
        workers=1,
        verbose=True,
        val=True,
    )
    save_dir = Path(results.save_dir)  # /artifacts/runs/<name>
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"

    if not best_pt.exists():
        print("ERROR: best.pt not found after training.", file=sys.stderr)
        sys.exit(2)

    # Export to ONNX 
    onnx_path = None
    if args.export_onnx:
        exp = YOLO(str(best_pt))
        onnx_path = exp.export(format="onnx", dynamic=True, opset=13)
        # ultralytics returns path or list; normalize
        onnx_path = Path(onnx_path if isinstance(onnx_path, str) else exp.exporter.save_dir / "model.onnx")

    # Upload to MinIO
    if args.upload:
        client = Minio(
            args.minio_endpoint, access_key=args.minio_access, secret_key=args.minio_secret,
            secure=args.minio_secure
        )
        # artifacts layout in MinIO: models/<prefix>/<run>/best.pt and model.onnx
        base_prefix = f"{args.artifact_prefix}/{run_name}".replace("//","/")
        upload(client, args.minio_bucket, best_pt, f"{base_prefix}/best.pt")
        # also save a pointer "latest.json"
        meta = {"best_pt": f"{base_prefix}/best.pt"}
        if onnx_path and Path(onnx_path).exists():
            upload(client, args.minio_bucket, Path(onnx_path), f"{base_prefix}/model.onnx")
            meta["onnx"] = f"{base_prefix}/model.onnx"
        # write meta locally and upload
        meta_path = save_dir / "latest.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        upload(client, args.minio_bucket, meta_path, f"{args.artifact_prefix}/latest.json")

    # Print final model performance summary
    print("\n" + "="*70)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    # Load the best model and validate
    final_model = YOLO(str(best_pt))
    val_metrics = final_model.val(
        data=args.data,         
        split="val",            
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=1,
        verbose=False,
    )
    
    # Overall summary
    print("\nValidation summary (overall):")
    try:
        print(f"Precision (mp): {val_metrics.box.mp:.3f}")
        print(f"Recall (mr):    {val_metrics.box.mr:.3f}")
    except Exception:
        pass

    try:
        print(f"mAP50:           {val_metrics.box.map50:.3f}")
    except Exception:
        pass

    try:
        print(f"mAP50-95:        {val_metrics.box.map:.3f}")
    except Exception:
        pass

    print("-" * 50)

    # Per-class metrics
    names = getattr(final_model, "names", None) or getattr(val_metrics, "names", None) or {}
    if isinstance(names, dict):  # {id: name}
        id_to_name = names
    else:
        # Fallback if names is a list or missing
        id_to_name = {i: (names[i] if isinstance(names, (list, tuple)) and i < len(names) else str(i))
                    for i in range(max(len(getattr(val_metrics.box, "maps", []) or []), 0))}

    per_class_map = getattr(val_metrics.box, "maps", None)  # mAP@0.50:0.95 per class
    per_class_map50 = getattr(val_metrics.box, "maps50", None)  # mAP@0.50 per class (may not exist)

    # Pretty print header
    if per_class_map is not None:
        if per_class_map50 is not None:
            header = f"{'Class':25s} {'ID':>3s} {'mAP50':>10s} {'mAP50-95':>12s}"
        else:
            header = f"{'Class':25s} {'ID':>3s} {'mAP50-95':>12s}"
        print(header)
        print("-" * len(header))

        for cls_id, map95 in enumerate(per_class_map):
            cls_name = id_to_name.get(cls_id, str(cls_id))
            if per_class_map50 is not None and cls_id < len(per_class_map50):
                print(f"{cls_name:25s} {cls_id:>3d} {per_class_map50[cls_id]:>10.3f} {map95:>12.3f}")
            else:
                print(f"{cls_name:25s} {cls_id:>3d} {map95:>12.3f}")
    else:
        print("Per-class mAP values are not available in this Ultralytics version.")

    print("-" * 50)
    print("Done.")

    print(f"Training done. best.pt: {best_pt}")
    if onnx_path:
        print(f"Exported ONNX: {onnx_path}")

if __name__ == "__main__":
    main()
