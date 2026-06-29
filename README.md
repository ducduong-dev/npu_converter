# imx95-npu-converter

Convert **PyTorch, TensorFlow, ONNX, or TFLite** models into a
`*_neutron.tflite` that runs on the **i.MX 95 eIQ Neutron NPU** and is
auto-detected by the GoPoint demos.

> **New here? Start with the step-by-step walkthrough: [`docs/USAGE.md`](docs/USAGE.md)** —
> setup, a command per format, preparing calibration data, Docker/compose, and
> deploying to the board.

The Neutron NPU only accelerates **int8-quantized** graphs. This tool normalizes
any of the four source formats to a quantized TFLite, then compiles it with NXP's
`neutron-converter`. There are two quantization backends:

**`nxp` backend (preferred, needs the eIQ Neutron SDK)** — uses NXP's own tools:

```text
source ──to float TFLite──► tflite-profiler ──► profile.csv ──► tflite-quantizer ──► int8 ──► neutron-converter ──► *_neutron.tflite
```

**`tflite` backend (fallback, no SDK)** — quantizes with TFLiteConverter/onnx2tf:

```text
source ──normalize+PTQ──► int8 TFLite ──► neutron-converter ──► *_neutron.tflite
 .pt/.pth   torch → onnx → onnx2tf (int8 PTQ)
 tf/keras   tf.lite.TFLiteConverter (int8 PTQ)
 .onnx      onnx2tf (int8 PTQ w/ calibration)
 .tflite    pass-through if int8, else error
```

`--quant-backend auto` (default) picks `nxp` when the SDK is found, else `tflite`.

## Important constraints

- **Host tool, not on-board.** Run on an x86_64 Linux dev host. The NXP SDK
  tools (`neutron-converter`, `tflite-profiler`, `tflite-quantizer`) ship in the
  eIQ Neutron SDK — they are **not** pip-installable and **not** on the board.
- **Converter version must match the target Neutron runtime** (driver, firmware,
  delegate — all from the same SDK). With SDK ≥3.1 a mismatch is reported at load
  time as `Microcode version mismatch!` (older runtimes fell back to CPU
  silently). Tested here with SDK `3.1.3`; check the board with
  `dpkg -s neutron | grep Version`.
- **Quantization needs calibration data.** Any non-int8 source must be
  post-training-quantized with a representative dataset (`--rep-data`). A bare
  **float `.tflite`** can be quantized by the **`nxp` backend** (the SDK
  quantizer takes a float `.tflite`), but **not** by the `tflite` backend
  (TFLiteConverter needs the source graph) — there, pass the original model.

## Install

```bash
# Everything (large: tensorflow + torch + onnx2tf)
pip install ".[full]"

# Or just the path you need
pip install ".[tensorflow]"   # TF/Keras sources
pip install ".[onnx]"         # ONNX sources (also used by torch)
pip install ".[torch]"        # adds the torch export step
```

Or use Docker (bundles the whole stack):

```bash
docker build -t imx95-convert .
```

### Provision the eIQ Neutron SDK (for the `nxp` backend)

The SDK (`eiq-neutron-sdk-linux-*.zip`, downloaded from NXP) carries the real
`neutron-converter`, `tflite-profiler` and `tflite-quantizer`. It is proprietary
and **not redistributable**, so it is never committed or baked into the image —
extract it and point the tool at it:

```bash
scripts/setup_sdk.sh eiq-neutron-sdk-linux-3.1.3.zip
export NEUTRON_SDK_DIR="$PWD/vendor/eiq-neutron-sdk"   # or pass --sdk-dir
```

The converter version **must match the board's Neutron runtime** (driver,
firmware, delegate from the same SDK); a mismatch is reported at load time as
`Microcode version mismatch!`.

## Usage

```bash
imx95-convert MODEL [MODEL ...] [options] [-- EXTRA_NEUTRON_FLAGS]
```

| Option | Purpose |
| --- | --- |
| `--format` | `auto` (default) or `torch`/`tensorflow`/`onnx`/`tflite` |
| `--rep-data PATH` | Calibration data: a `.npy`, an image directory, or a `.py` exposing `representative_dataset()` |
| `--rep-norm` | Image normalization: `none` / `0to1` / `-1to1` / `imagenet` |
| `--input-shape` | Input dims without batch, e.g. `3,224,224`. **Required for torch** and dynamic-shape ONNX |
| `--max-samples` | Calibration sample cap (default 200) |
| `--fail-on-float-ops` | Error (not warn) if any op can't be int8-quantized |
| `--target` | Neutron target (default `imx95`) |
| `--output-dir` | Where to write `*_neutron.tflite` (default: beside each input) |
| `--quant-backend` | `auto` (default) / `nxp` / `tflite` |
| `--sdk-dir` | Path to the extracted eIQ Neutron SDK (or `$NEUTRON_SDK_DIR`) |
| `--calibration-method` | `MinMax` (default) / `Percentile` (nxp backend) |
| `--percentile-val` | clip percentile when `--calibration-method=Percentile` |
| `--float-io` | Keep **float32** model inputs/outputs (int8 internals). Required by the GoPoint object-detection demos (nxp backend) |
| `--no-neutron` | Emit `<name>_int8.tflite` (int8, CPU/XNNPACK) and skip NPU compilation. Use when the NPU mis-quantizes a head — e.g. YOLO class scores (see note below) |
| `--yolo-raw-head` | YOLOv8/11: export the head before box decoding (DFL + class logits) and decode on the host — accurate boxes on the NPU (see note below) |
| `--optimization-level` | `OFast` (fast) / `OOpt` (optimal, slower) for the converter |
| `--validate` | dry-run the converted model (`--run-after-generate`) |
| `--converter-cmd` | neutron-converter path for the **tflite** backend (or `$NEUTRON_CONVERTER`) |

Anything after `--` is forwarded verbatim to `neutron-converter`
(e.g. `--use-sequencer`, `--fetch-constants-to-sram`).

### Examples

```bash
# NXP backend (SDK provisioned): ONNX -> float -> SDK quantizer -> neutron
export NEUTRON_SDK_DIR="$PWD/vendor/eiq-neutron-sdk"
imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224 --validate

# Already-quantized TFLite: just the last-mile neutron compile
imx95-convert model.tflite

# ONNX with a numpy calibration set
imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224

# Keras model, calibrate from a folder of JPEGs scaled to [0,1]
imx95-convert model.h5 --rep-data ./calib_images/ --rep-norm 0to1

# For the GoPoint object-detection demos: keep float32 I/O (they feed/read float)
imx95-convert yolov8s.onnx --rep-data ./coco/ --rep-norm 0to1 --float-io

# PyTorch (whole-module or TorchScript .pt; state_dicts are rejected)
imx95-convert model.pt --rep-data calib.npy --input-shape 3,224,224

# Forward extra flags to the neutron converter
imx95-convert model.tflite -- --use-sequencer

# YOLO that the NPU mis-quantizes -> emit a CPU/XNNPACK int8 model instead
imx95-convert yolo11.onnx --rep-data ./imgs/ --rep-norm 0to1 --float-io --no-neutron

# Best for YOLO on the NPU: raw-head export (decode on the host) -- accurate boxes
imx95-convert yolov8s.onnx --rep-data ./coco/ --rep-norm 0to1 --float-io --yolo-raw-head
```

### YOLO detection heads on the Neutron NPU

YOLOv8/YOLO11 export **one** output that concatenates boxes (pixel units) with
class scores (0–1). int8 quantization with a single shared scale collapses the
class scores to 0, so the converter auto-splits that into **separate box/score
outputs** (`note: split YOLO head ...`), each with its own scale.

Even split, the **NPU's int8 kernels are imprecise for YOLO's decode math**:

- **YOLO11** — the class head/neck (depthwise convs, C3k2) loses precision: class
  confidences collapse on the NPU (box survives). Verified: same int8 model scores
  ~0.85 on CPU vs ~0.01–0.3 on the NPU. YOLOv8's regular-conv head does **not**
  have this problem — prefer **YOLOv8** for NPU deployment.
- **YOLOv8** — class scores survive, but the box **w/h** (computed as `x2 − x1`,
  a difference of large pixel values) collapses to int8 catastrophic cancellation
  (`cx/cy` survive).

**Fix: `--yolo-raw-head`.** It exports the head *before* decoding — raw box DFL
logits + class logits — so the NPU runs only convolutions (accurate) and the host
does DFL + `dist2bbox` + sigmoid in float. This gives accurate boxes *and* classes
on the NPU (yolov8s_640: ~28 FPS on i.MX95, boxes matching CPU). The patched demo
auto-detects raw-head outputs. Falls back cleanly: `--no-neutron` for a CPU model,
or quantization-aware training for the highest NPU accuracy.

### Representative dataset

`--rep-data` accepts:

- **`.npy`** — array shaped `(N, *input_shape)`, already preprocessed.
- **directory of images** — resized to the model input and normalized per
  `--rep-norm`.
- **`.py` file** — defines `representative_dataset()` yielding `[sample]`
  batches (the TFLite generator protocol); full control over preprocessing.

## Running with Docker

```bash
docker run --rm \
  -v "$PWD/models:/data" \
  -v "$PWD/vendor/eiq-neutron-sdk:/opt/sdk:ro" \
  -e NEUTRON_SDK_DIR=/opt/sdk \
  imx95-convert /data/model.onnx --rep-data /data/calib.npy --input-shape 3,224,224
```

### docker compose

`docker-compose.yml` runs a **long-lived** container you `exec` into, so repeated
conversions reuse one warm container. Paths are relative to `/data` (`MODELS_DIR`):

```bash
cp .env.example .env                       # set MODELS_DIR / NEUTRON_SDK_DIR
docker compose up -d --build               # start once
docker compose exec convert imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
docker compose exec convert imx95-convert model.tflite
docker compose exec convert imx95-convert --help
docker compose down                        # stop when done
```

Files added to `MODELS_DIR` appear live in the container (bind mount), so you can
keep dropping in models and re-running `exec` without a restart. For a one-off run
without a persistent container: `docker compose run --rm --entrypoint imx95-convert convert --help`.

## Development & testing without the NXP binary

A stub converter lets you exercise the whole pipeline (and the test suite)
without the proprietary `neutron-converter`:

```bash
export NEUTRON_CONVERTER=tools/neutron-converter-stub.py
imx95-convert model.tflite
pip install ".[dev]" && pytest
```

The stub copies the input and appends a marker — the output is **not**
board-loadable; it only proves the staging/compile wiring works.

## Legacy script

`convert_neutron_model.py` still works for the last-mile (quantized TFLite →
neutron) and now wraps the same shared core (`imx95_npu/neutron.py`).
