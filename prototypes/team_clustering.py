import os, io, sys, math, json, tempfile, pathlib
from typing import List
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

from minio import Minio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import open_clip

# -------------------
# Config via env vars
# -------------------
MATCH_ID = int(os.getenv("MATCH_ID", "14"))
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

BUCKET_DETECTIONS = os.getenv("BUCKET_DETECTIONS", "detections")
BUCKET_FRAMES     = os.getenv("BUCKET_FRAMES",     "frames")
BUCKET_TRACKS     = os.getenv("BUCKET_TRACKS",     "tracks")

MAX_SAMPLES_PER_TRACK = int(os.getenv("MAX_SAMPLES_PER_TRACK", "8"))
USE_UMAP = os.getenv("USE_UMAP", "1") == "1"  # set to 0 to skip UMAP
UMAP_COMPONENTS = int(os.getenv("UMAP_COMPONENTS", "16"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_COLOR = float(os.getenv("ALPHA_COLOR", "0.3"))  # weight for HSV hist in concat feature

# -------------------
# MinIO helpers
# -------------------
def _minio() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)

def read_parquet_via_temp(bucket: str, key: str) -> pd.DataFrame:
    """
    Keep consistency with earlier code: fetch to a temp file and let pandas read from path.
    Avoid BytesIO for Parquet.
    """
    cli = _minio()
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, pathlib.Path(key).name)
        cli.fget_object(bucket, key, dst)
        return pd.read_parquet(dst, engine="pyarrow")

def write_parquet_via_temp(bucket: str, key: str, df: pd.DataFrame):
    cli = _minio()
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, pathlib.Path(key).name)
        df.to_parquet(dst, engine="pyarrow", index=False)
        cli.fput_object(bucket, key, dst, content_type="application/octet-stream")

def read_frame(match_id: int, filename: str) -> np.ndarray:
    """
    For frames we keep the efficient bytes approach (many small JPEGs): consistent with previous code.
    """
    cli = _minio()
    key = f"{filename}"
    obj = cli.get_object(BUCKET_FRAMES, key)
    arr = np.frombuffer(obj.read(), dtype=np.uint8)
    obj.close(); obj.release_conn()
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

# -------------------
# Vision helpers
# -------------------
def crop_jersey(img_bgr: np.ndarray, x1, y1, x2, y2) -> Image.Image:
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w, int(x2)); y2 = min(h, int(y2))
    top = y1
    bot = y1 + max(1, int(0.6 * (y2 - y1)))  # top 60% of bbox
    crop = img_bgr[top:bot, x1:x2]
    if crop.size == 0:
        crop = img_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def hsv_hist(img_bgr: np.ndarray, x1, y1, x2, y2, bins=(12,6,6)) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w, int(x2)); y2 = min(h, int(y2))
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((bins[0]*bins[1]*bins[2],), dtype=np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist

# -------------------
# Embeddings (OpenCLIP)
# -------------------
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
    model.eval()
    return model, preprocess

def embed_batch(model, preprocess, pil_images: List[Image.Image]) -> np.ndarray:
    with torch.no_grad():
        batch = torch.stack([preprocess(im) for im in pil_images]).to(DEVICE)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy()

# -------------------
# Main
# -------------------
def main():
    # 1) Load detections parquet via temp file (no bytes)
    
    det_key = f"match_{MATCH_ID}/match_{MATCH_ID}.parquet"
    df = read_parquet_via_temp(BUCKET_DETECTIONS, det_key)
    df = df[(df["class_name"] == "player") & df["object_id"].notna()].copy()
    if df.empty:
        print("No player detections with object_id; nothing to do.")
        return 0
    df["object_id"] = df["object_id"].astype(int)

    # 2) Sample frames per object_id
    samples = []  # (oid, frame_id, pil_crop, hsv_hist)
    model, preprocess = load_clip()
    print(f"Sampling up to {MAX_SAMPLES_PER_TRACK} frames per track...")
    for oid, grp in df.groupby("object_id"):
        g = grp.sort_values("frame_id")
        step = max(1, len(g)//MAX_SAMPLES_PER_TRACK)
        g = g.iloc[::step][:MAX_SAMPLES_PER_TRACK]
        for _, r in g.iterrows():
            frame = read_frame(MATCH_ID, r["filename"])
            pil = crop_jersey(frame, r["x1"], r["y1"], r["x2"], r["y2"])
            hist = hsv_hist(frame, r["x1"], r["y1"], r["x2"], r["y2"])
            samples.append((oid, r["frame_id"], pil, hist))
    if not samples:
        print("No samples collected; exiting.")
        return 0

    # 3) Embed jersey crops
    print("Embedding crops with OpenCLIP...")
    pil_list = [s[2] for s in samples]
    feats_e = []
    B = 32
    for i in tqdm(range(0, len(pil_list), B)):
        feats_e.append(embed_batch(model, preprocess, pil_list[i:i+B]))
    feats_e = np.concatenate(feats_e, axis=0)

    # 4) Aggregate per track (mean embedding + mean HSV hist)
    rows = []
    oids = sorted(set(s[0] for s in samples))
    for oid in oids:
        idx = [i for i,(o,_,_,_) in enumerate(samples) if o == oid]
        e = feats_e[idx].mean(axis=0)
        h = np.stack([samples[i][3] for i in idx]).mean(axis=0)
        h = h / (np.linalg.norm(h) + 1e-8)
        feat = np.concatenate([e, ALPHA_COLOR * h], axis=0)  # concat embedding + color
        rows.append((oid, len(idx), feat))
    X = np.stack([r[2] for r in rows], axis=0)
    ns = [r[1] for r in rows]

    # 5) Dimensionality reduction (optional UMAP → else PCA)
    if USE_UMAP:
        try:
            from umap import UMAP
            print(f"Reducing with UMAP to {UMAP_COMPONENTS} dims...")
            reducer = UMAP(n_components=UMAP_COMPONENTS, random_state=42,
                           n_neighbors=10, min_dist=0.05)
            Xr = reducer.fit_transform(X)
        except Exception as e:
            print(f"UMAP failed ({e}); falling back to PCA(16).")
            Xr = PCA(n_components=min(16, X.shape[1]-1), random_state=42).fit_transform(X)
    else:
        print("UMAP disabled; using PCA(16).")
        Xr = PCA(n_components=min(16, X.shape[1]-1), random_state=42).fit_transform(X)

    # 6) KMeans to 2 teams
    print("Clustering with KMeans(k=2)...")
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(Xr)
    team_ids = (labels + 1).astype(int)  # map to {1,2}

    # 7) Save tracks parquet via temp file (no bytes)
    out = pd.DataFrame({
        "match_id": MATCH_ID,
        "object_id": [r[0] for r in rows],
        "n_samples": ns,
        "team_id": team_ids,
    }).sort_values(["team_id","object_id"])

    write_parquet_via_temp(BUCKET_TRACKS, f"match_{MATCH_ID}/match_{MATCH_ID}.parquet", out)
    summary = out.groupby("team_id")["object_id"].count().to_dict()
    print("Assignment summary:", summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())