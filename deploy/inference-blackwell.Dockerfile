# Roboflow inference server for NVIDIA Blackwell (RTX 50-series, compute sm_120).
#
# The stock roboflow/roboflow-inference-server-gpu ships a PyTorch built only up to sm_90
# (Hopper), so on an RTX 5090 every model fails with
#   "CUDA error: no kernel image is available for execution on the device".
# Blackwell needs CUDA 12.8 (cu128) builds. This image layers the cu128 PyTorch and a
# matching onnxruntime-gpu on top of the stock image, which is what actually runs on the
# 5090.
#
# Proven on an RTX 5090 (driver 595.79, WSL2 + Docker Desktop) on 2026-06-15: CLIP
# embeddings and ONNX detection both run on the GPU; warm CLIP ~0.06s.
#
# Build (on a machine with the GPU):
#   docker build -f deploy/inference-blackwell.Dockerfile -t roboflow-inference-blackwell .
# Run:
#   docker run -d --name inference --restart unless-stopped --gpus all -p 9001:9001 \
#       roboflow-inference-blackwell
#
# Only needed for RTX 50-series / Blackwell GPUs. On older GPUs the stock image works as-is.

FROM roboflow/roboflow-inference-server-gpu:latest

RUN pip install --no-cache-dir --upgrade \
        torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir --upgrade onnxruntime-gpu==1.23.2
