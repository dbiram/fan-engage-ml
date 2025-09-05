import os, json, tempfile, pathlib, shutil, argparse
import time
from datetime import datetime
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

MINIO_ENDPOINT        = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ROOT_USER       = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_ROOT_PASSWORD   = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
MINIO_SECURE          = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET_MODELS         = os.getenv("BUCKET_MODELS", "models")

def _minio():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=MINIO_SECURE,
    )

def _upload_file(cli: Minio, bucket: str, local_path: str, key: str, content_type="application/octet-stream"):
    cli.fput_object(bucket, key, local_path, content_type=content_type)
    print(f"uploaded s3://{bucket}/{key}")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 pose model for pitch keypoints")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml inside container")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")  # e.g. cuda:0
    parser.add_argument("--project", type=str, default="/artifacts/runs")
    parser.add_argument("--artifact-prefix", type=str, default="pitch_keypoints")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.project, exist_ok=True)

    # ---- train ----
    model = YOLO(args.model)
    epoch_times = {}

    def on_train_epoch_start(trainer):
        epoch = trainer.epoch + 1
        total = trainer.epochs
        epoch_times[epoch] = time.time()
        print(f"\n[Epoch {epoch}/{total}] starting...")

    def on_train_epoch_end(trainer):
        epoch = trainer.epoch + 1
        total = trainer.epochs
        start = epoch_times.get(epoch)
        elapsed = time.time() - start if start else 0.0
        # metrics come in trainer.metrics dict
        metrics = {k: float(v) for k, v in (trainer.metrics or {}).items()}
        print(f"[Epoch {epoch}/{total}] done in {elapsed:.1f}s — metrics: {metrics}")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    run_name = "exp"
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
        amp=True,
        pretrained=True,
        workers=1,
        verbose=True,
        val=True,
        mosaic=0.0
    )
    run_dir = pathlib.Path(results.save_dir)  # e.g. /artifacts/runs/exp
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Ensure best.pt is in weights_dir (Ultralytics puts it there already)
    best_pt = weights_dir / "best.pt"
    if not best_pt.exists():
        # try common fallbacks
        for cand in (run_dir / "weights" / "best.pt", run_dir / "best.pt"):
            if cand.exists():
                shutil.copyfile(cand, best_pt)
                break

    # ---- export ONNX (optional, recommended for API) ----
    onnx_path = weights_dir / "model.onnx"
    if args.export_onnx:
        print(f"[export] reloading trained weights from: {best_pt}")
        export_model = YOLO(str(best_pt))
        cwd = os.getcwd()
        try:
            os.chdir(weights_dir)
            print(f"[export] exporting ONNX (imgsz={args.imgsz}, dynamic=True, opset=12) ...")
            export_model.export(format="onnx", imgsz=args.imgsz, dynamic=True, opset=12)

            # Known possible filenames depending on Ultralytics version
            candidates = [
                "best.onnx",                 # common when exporting from best.pt
                "model.onnx",                # some versions
                "yolov8s-pose.onnx",         # depends on base model name
                "yolov8n-pose.onnx",
                os.path.join("weights", "best.onnx"),
            ]

            found = None
            for c in candidates:
                p = pathlib.Path(c)
                if p.exists():
                    found = p
                    break

            if found is None:
                # final sweep: any .onnx in weights_dir
                for p in pathlib.Path(".").glob("*.onnx"):
                    found = p
                    break

            if found is None:
                print("[export] WARNING: ONNX file not found after export step.")
            else:
                print(f"[export] found ONNX at: {found} -> moving to {onnx_path}")
                # ensure parent exists
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(found), str(onnx_path))

        finally:
            os.chdir(cwd)

    # ---- upload to MinIO: models/pitch_keypoints/<version>/{best.pt, model.onnx} ----
    cli = _minio()
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    obj_prefix = f"{args.artifact_prefix}/{version}"

    if best_pt.exists():
        _upload_file(cli, BUCKET_MODELS, str(best_pt), f"{obj_prefix}/best.pt")
    if onnx_path.exists():
        _upload_file(cli, BUCKET_MODELS, str(onnx_path), f"{obj_prefix}/model.onnx")

    # ---- write latest.json pointer ----
    latest_key = f"{args.artifact_prefix}/latest.json"
    with tempfile.TemporaryDirectory() as td:
        latest_path = os.path.join(td, "latest.json")
        payload = {
            "version": version,
            "pt": f"{obj_prefix}/best.pt",
            "onnx": f"{obj_prefix}/model.onnx",
            "imgsz": args.imgsz,
            "artifact_prefix": args.artifact_prefix,
        }
        with open(latest_path, "w") as f:
            json.dump(payload, f)
        _upload_file(cli, BUCKET_MODELS, latest_path, latest_key, content_type="application/json")

    print("Done. Latest pointer updated:", latest_key)

if __name__ == "__main__":
    main()
