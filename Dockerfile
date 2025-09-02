FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# === PyTorch NIGHTLY with CUDA 12.8 (includes SM 12.0 / Blackwell) ===
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
RUN pip install --no-cache-dir --pre --index-url $TORCH_INDEX_URL \
    torch torchvision torchaudio 

# deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Ultralytics checks for this dir to save runs
RUN mkdir -p /artifacts
ENV YOLO_VERBOSE=1 PYTHONUNBUFFERED=1

COPY train_yolov8.py ./
COPY prototypes/ ./prototypes/
CMD ["python", "train_yolov8.py", "--help"]
