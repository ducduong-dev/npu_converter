# Multi-format -> i.MX 95 Neutron NPU converter.
#
# This image bundles the heavy/conflicting conversion stack (tensorflow + torch
# + onnx2tf) so you don't have to resolve it on the host. It does NOT contain
# the proprietary NXP eIQ Neutron SDK (non-redistributable) -- mount the
# extracted SDK at runtime and point the tool at it:
#
#   docker build -t imx95-convert .
#   docker run --rm \
#     -v "$PWD/models:/data" \
#     -v /opt/eiq-neutron-sdk:/opt/sdk:ro \
#     -e NEUTRON_SDK_DIR=/opt/sdk \
#     imx95-convert /data/model.onnx --rep-data /data/calib.npy --input-shape 3,224,224
#
# With the SDK present the tool uses the nxp quantization backend automatically.
# Without it, the tflite (TFLiteConverter/onnx2tf PTQ) backend is used; for the
# final neutron step you can still mount just the converter
# (-e NEUTRON_CONVERTER=/opt/sdk/bin/neutron-converter) or the bundled dev stub
# (-e NEUTRON_CONVERTER=/app/tools/neutron-converter-stub.py) for testing.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libGL/glib are needed by some image/onnx ops pulled in transitively.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU torch wheel first (smaller, avoids CUDA), then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
    && pip install -r requirements.txt

# TF 2.16 hard-pins ml-dtypes~=0.3.1, but ai-edge-litert (pulled by onnx2tf) needs
# the newer ml_dtypes with float4_e2m1fn. TF tolerates the newer one at runtime,
# so force the upgrade (pip warns about the TF pin; that's expected and benign).
RUN pip install --upgrade "ml_dtypes>=0.5.1"

COPY . .
RUN pip install --no-deps .

ENTRYPOINT ["imx95-convert"]
CMD ["--help"]
