FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# These runtime libraries are needed by OpenCV and the CPU build of PyTorch
# used by Ultralytics.  The local test mode can show a preview window, while
# Docker normally runs with SHOW_PREVIEW=false and only outputs alerts to the
# terminal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt .
# This service is CPU-only.  The constraints force CPU PyTorch wheels and
# prevent dependency resolution from selecting CUDA runtime packages.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt -c constraints.txt

COPY app ./app

# The production model path is fixed at /app/yolo26m.pt. Ultralytics downloads
# yolo26m.pt there on first start if the file is not already present.
CMD ["python", "-m", "app.main"]
