# Using imx95-convert — a walkthrough

This guide takes you from a trained model to a `*_neutron.tflite` running on the
i.MX 95 board. For the design/architecture see `../CLAUDE.md`; for the SDK
integration record see `SDK_INTEGRATION_PLAN.md`.

---

## 0. The mental model (read this first)

The i.MX 95 Neutron NPU only runs **int8-quantized** TFLite. This tool takes any
of **PyTorch / TensorFlow / ONNX / TFLite** and produces the deployable
`<name>_neutron.tflite` in three stages:

```text
your model ──►  int8 TFLite  ──►  neutron-converter  ──►  <name>_neutron.tflite
              (quantization)        (NPU compile)            (copy to board)
```

Two things decide whether a run succeeds:

1. **Quantization needs calibration data** (`--rep-data`) unless your model is
   *already* int8. ~100–1000 real input samples.
2. **The converter version must match the board runtime.** Use the *same* eIQ
   Neutron SDK for the converter (here) and the board's driver/firmware/delegate.
   This build is tested with SDK **3.1.3**.

---

## 1. One-time setup

### 1a. Provision the eIQ Neutron SDK (enables the best results)

The SDK is downloaded from NXP (`eiq-neutron-sdk-linux-3.1.3.zip`). It is
proprietary, so it is never committed — extract it locally:

```bash
scripts/setup_sdk.sh eiq-neutron-sdk-linux-3.1.3.zip
export NEUTRON_SDK_DIR="$PWD/vendor/eiq-neutron-sdk"   # add to your shell rc
```

Without the SDK the tool still runs (the `tflite` fallback backend), but the
`nxp` backend (NXP's own profiler + quantizer) is preferred.

### 1b. Choose how to run it

**Option A — Docker (recommended; no Python deps to wrangle):**

```bash
docker build -t imx95-convert .
```

**Option B — local Python:**

```bash
pip install ".[full]"                       # tensorflow + onnx2tf + torch
pip install --upgrade "ml_dtypes>=0.5.1"    # see requirements.txt note
```

---

## 2. Quick start — one command per format

Put your model and calibration data in a folder, then:

| Source | Command (local CLI) |
| --- | --- |
| TFLite (already int8) | `imx95-convert model.tflite` |
| TensorFlow (SavedModel/.keras/.h5) | `imx95-convert model.keras --rep-data calib/` |
| ONNX | `imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224` |
| PyTorch (.pt, whole-module/TorchScript) | `imx95-convert model.pt --rep-data calib.npy --input-shape 3,224,224` |

Output `model_neutron.tflite` is written next to the input (or use
`--output-dir`). Exit codes: **0** all converted, **1** backend unavailable,
**2** some models failed.

> **PyTorch / dynamic ONNX need `--input-shape`** (dims without batch, e.g.
> `3,224,224`) so the model can be exported/traced. A PyTorch *state_dict* is
> rejected — save the whole module (`torch.save(model, ...)`) or TorchScript.

---

## 3. Preparing a representative dataset (`--rep-data`)

Calibration data must reflect real inputs in the model's **training input
range**. Three accepted forms:

**A. NumPy array** — shape `(N, *input_shape)`, already preprocessed.
```python
import numpy as np
# 200 samples, model-native layout (NCHW for onnx/torch, NHWC for tf/tflite)
np.save("calib.npy", my_preprocessed_samples.astype("float32"))
```
```bash
imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
```

**B. A directory of images** — resized to the model input and normalized with
`--rep-norm` (`none` / `0to1` / `-1to1` / `imagenet`):
```bash
imx95-convert model.keras --rep-data ./calib_images/ --rep-norm 0to1
```

**C. A Python hook** — full control over preprocessing. Create `rep.py`:
```python
def representative_dataset():
    import numpy as np
    for _ in range(200):
        yield [np.random.rand(1, 224, 224, 3).astype("float32")]  # your real data
```
```bash
imx95-convert model.keras --rep-data rep.py
```

Tune sample count with `--max-samples` (default 200).

---

## 4. Choosing / tuning the backend

`--quant-backend auto` (default) uses `nxp` when the SDK is found, else `tflite`.

- **`nxp`** (NXP profiler + quantizer): force with `--quant-backend nxp`.
  Tuning knobs:
  - `--calibration-method MinMax|Percentile` (+ `--percentile-val 99.9` to clip
    outliers — often improves accuracy on models with rare extreme activations).
  - `--optimization-level OFast|OOpt` (OOpt = best NPU schedule, slower convert).
  - `--float-io` keeps **float32** model inputs/outputs (int8 weights/activations
    inside). The GoPoint object-detection demos require this — they feed and read
    float32 tensors via `set_tensor`/`get_tensor` and do not pre-quantize. Without
    it the model has int8 I/O and the demo crashes with `Cannot set tensor: Got
    value of type FLOAT32 but expected type INT8 ... name: images`. Matches the
    reference `..._uint8_float32_...` models.
  - `--validate` dry-runs the converted model in the converter as a sanity check.
- **`tflite`** (no SDK): force with `--quant-backend tflite`. For the final
  neutron step it needs a converter binary (`--converter-cmd` / `$NEUTRON_CONVERTER`),
  or the bundled dev stub for testing.

Pass any extra neutron-converter flag verbatim after `--`:
```bash
imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224 -- --dump-statistics
```

---

## 5. Running with Docker

```bash
docker run --rm -w /data \
  -v "$PWD/models:/data" \
  -v "$PWD/vendor/eiq-neutron-sdk:/opt/sdk:ro" \
  -e NEUTRON_SDK_DIR=/opt/sdk \
  imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
```

**For repeated conversions, use the long-lived compose container** (start once,
`exec` many times):

```bash
cp .env.example .env          # set MODELS_DIR and NEUTRON_SDK_DIR
docker compose up -d --build
docker compose exec convert imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
docker compose exec convert imx95-convert another.keras --rep-data calib/
docker compose down
```

Drop new models into `MODELS_DIR` anytime — it's a live bind mount at `/data`.

---

## 6. Deploying to the board

On success the tool prints the sha1 and next steps. Copy the result onto the
i.MX 95:

```bash
scp model_neutron.tflite root@<board>:/run/media/mmcblk1p1/gopoint-apps/downloads/
```

The GoPoint demos auto-detect `*_neutron.tflite` on i.MX 95. Or register it in
`downloads.json` with the printed sha1.

**Keep versions aligned.** The board's Neutron runtime (from the *same* SDK)
lives at:

| File | Board path |
| --- | --- |
| `NeutronFirmware.elf` | `/lib/firmware/` |
| `libNeutronDriver.so` | `/lib/` |
| `libneutron_delegate.so` | `/lib/` |

A mismatch is reported at load time as `Microcode version mismatch!`. Check the
board with `dpkg -s neutron | grep Version`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `... need int8 quantization ... pass --rep-data` | Source is float and no calibration given. Provide `--rep-data`. |
| `PyTorch models need --input-shape` | Add `--input-shape C,H,W` (no batch). |
| `'...' is a state_dict, not a model` | Re-save the whole module: `torch.save(model, path)` or TorchScript. |
| `float TFLite model ... cannot be re-quantized` (tflite backend) | A bare float `.tflite` has no source graph. Use the **nxp** backend (it can quantize a float `.tflite`), or pass the original TF/ONNX/Torch model. |
| `--quant-backend nxp requires the eIQ Neutron SDK` | Set `--sdk-dir` / `$NEUTRON_SDK_DIR` (run `scripts/setup_sdk.sh`). |
| `Cannot set tensor: Got value of type FLOAT32 but expected type INT8 ... name: images` (in a GoPoint demo) | Model has int8 I/O but the demo feeds float32. Re-convert with `--float-io` (keeps float32 I/O, int8 internals). |
| YOLO detects nothing / wrong boxes on the NPU (class ~0, or box w/h tiny/negative) | NPU int8 is imprecise for YOLO decode (YOLO11: class collapses; YOLOv8: box w/h collapses). Re-convert with **`--yolo-raw-head`** (decode on the host — accurate boxes+classes on the NPU). Fallbacks: `--no-neutron` (CPU), or QAT. Prefer YOLOv8 over YOLO11 for the NPU. |
| `Neutron Converter '...' not found` (tflite backend) | No converter binary. Set `$NEUTRON_CONVERTER`/`--converter-cmd`, or use the nxp backend with the SDK. |
| `model not found: model.onnx` under `docker run` | Pass `-w /data` (or absolute `/data/model.onnx`). Compose already sets the working dir. |
| `Microcode version mismatch!` on the board | Converter and board runtime are from different SDKs. Re-convert with the SDK matching the board, or update the board runtime (§6). |
| Output runs on CPU, not NPU | Model wasn't fully int8, or some ops are unsupported. Re-run with `--fail-on-float-ops` to surface which, and check `docs/SupportedOperatorsS.md` in the SDK. |

---

## 8. Develop / test without the real converter

The bundled stub stands in for `neutron-converter` so the whole pipeline and the
test suite run with no SDK (output is **not** board-loadable — it only proves the
wiring):

```bash
export NEUTRON_CONVERTER=tools/neutron-converter-stub.py
pip install ".[dev]" && pytest          # SDK-backed tests auto-skip without the SDK
```
