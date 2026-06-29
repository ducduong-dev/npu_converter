# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A host-side tool that converts **PyTorch / TensorFlow / ONNX / TFLite** models into a
`<name>_neutron.tflite` for the **i.MX 95 eIQ Neutron NPU** — the file the GoPoint demos
(`object_detection_test.py`, `rtsp_object_detection.py`, `face_recognition.py`) auto-detect.

The public entry point is the `imx95-convert` CLI (`imx95_npu.cli:main`,
`python -m imx95_npu.cli`). The original `convert_neutron_model.py` still exists as a thin
backward-compatible wrapper for the last-mile (quantized TFLite → neutron) step only.

User-facing docs: `README.md` (overview/reference) and `docs/USAGE.md` (step-by-step
walkthrough — point users there). `docs/SDK_INTEGRATION_PLAN.md` is the SDK integration record.

## Architecture

The pipeline normalizes any source format to an **int8 TFLite**, then compiles it with NXP's
`neutron-converter`. There are **two quantization backends** (selected by `--quant-backend`):

- **`nxp`** (preferred; needs the eIQ Neutron SDK): source → *float* TFLite →
  `tflite-profiler` → `tflite-quantizer` → `neutron-converter`. Uses NXP's own tools.
- **`tflite`** (fallback; no SDK): source converters quantize to int8 themselves
  (TFLiteConverter / onnx2tf PTQ) → `neutron-converter` (or the dev stub).
- **`auto`** (default): `nxp` if the SDK toolchain is found, else `tflite`.

The whole flow lives in `imx95_npu/`:

- `cli.py` — argparse front end; splits `-- extra` neutron flags, builds `ConvertOptions`.
- `pipeline.py` — `_select_backend()` then per-model `_convert_nxp` / `_convert_tflite`
  (`detect → _to_tflite → quantize → neutron`). Converters are imported **lazily** so a
  TFLite/ONNX job never imports torch+tf.
- `sdk.py` — `NeutronSDK.resolve()` locates the SDK tools (`$NEUTRON_SDK_DIR`/`--sdk-dir`/PATH);
  wraps `tflite-profiler` (`profile`), `tflite-quantizer` (`quantize`), and the converter
  (`convert`, which reuses `neutron.convert_one`). `has_full_toolchain()` gates the nxp backend.
- `detect.py` — format detection (content sniff: `TFL3` magic, ZIP for torch, protobuf for
  onnx; falls back to extension).
- `converters/` — one module per source family, each exposing
  `to_tflite(input, output, *, rep_data, input_shape, fail_on_float_ops, float_only) -> ConvertResult`.
  `float_only=True` (nxp backend) emits a **float** TFLite (no PTQ); otherwise they int8-quantize.
  `ConvertResult.input_shape` carries the model's **native** layout (NHWC for tf/tflite, NCHW for
  onnx/torch) so the nxp backend lays out calibration data correctly.
  - `tf_to_tflite` — Keras/SavedModel; `quantize.tf_to_float_tflite` or int8 PTQ.
  - `onnx_to_tflite` — `onnx2tf`; float (`*_float32.tflite`) or int8 with calibration `.npy`.
  - `torch_to_tflite` — `torch.onnx.export` (`dynamo=False`) → delegates to `onnx_to_tflite`.
  - `tflite_passthrough` — int8 passes through; float is accepted only on the nxp backend
    (it gets quantized downstream), else a hard error.
- `quantize.py` — `is_int8_quantized()` / `tflite_input_shape()` (inspect TFLite) + TF PTQ/float helpers.
- `repdata.py` — representative-dataset loaders (`.npy` / image dir / `.py` hook). `RepDataSpec`
  is the unbound CLI request; converters call `.build(input_shape)`. Emits NHWC `tf_generator`,
  NCHW `to_calibration_npy` (onnx2tf), and raw-float32 `to_profiler_dataset` (NXP profiler).
- `neutron.py` — the shared last-mile core (see below).

## Key design decisions (and why)

- **Two backends.** The NXP SDK quantizer is the official, NPU-optimal path; we prefer it when
  present and keep the in-house PTQ as a no-SDK fallback. The SDK is proprietary
  (`LA_OPT_NXP` license) → never committed/baked in; provisioned via `scripts/setup_sdk.sh`
  into `vendor/` (gitignored) and referenced by `$NEUTRON_SDK_DIR`.
- **Quantization can't run on a finished float `.tflite`** with `TFLiteConverter` (needs the
  source graph) — but the **NXP backend can** (`tflite-quantizer` takes a float `.tflite`).
- **One representative-dataset loader, multiple layouts.** TF path → NHWC `tf_generator`;
  onnx2tf int8 → NCHW `to_calibration_npy`; NXP profiler → NHWC raw-float32 `to_profiler_dataset`.
  The loader transposes from the model's native layout (`ConvertResult.input_shape`).
  Image normalization (`--rep-norm`) is applied once in `repdata`.
- **Torch requires `--input-shape`** (ONNX export needs an example input); state_dicts are
  rejected (no model class to rebuild). Whole-module / TorchScript `.pt` only.

## Critical operational constraints

These domain facts aren't obvious from the code and drive most decisions:

- **Host tool, NOT on-board.** Runs on x86_64 Linux. The NXP `neutron-converter` ships with
  the eIQ Toolkit / meta-imx SDK, is **not** pip-installable, and is **not** on the board.
  Workflow: convert on host → copy `*_neutron.tflite` to the board's
  `/run/media/mmcblk1p1/gopoint-apps/downloads/` (or register it in `downloads.json` with the
  printed sha1).
- **Converter version must match the target Neutron runtime** (driver, firmware, delegate —
  all from the same SDK), or execution is unreliable. SDK ≥3.1 reports a `Microcode version
  mismatch!` at load time (older runtimes fell back to CPU silently). The board runtime files
  live at `/lib/firmware/NeutronFirmware.elf`, `/lib/libNeutronDriver.so`,
  `/lib/libneutron_delegate.so` (see `docs/NeutronSDKUserGuide.md` BSP table). Verify with
  `dpkg -s neutron` on target.
- **Only int8/uint8 graphs are accelerated.** Non-int8 sources need `--rep-data` calibration.
- **The `_neutron.tflite` suffix is a contract** with the demos — don't change `NEUTRON_SUFFIX`.
- **sha1 (not sha256)** — matches `downloads.json`'s `sha` field.

## Commands

```bash
# Provision the proprietary SDK (enables the nxp backend); never committed
scripts/setup_sdk.sh eiq-neutron-sdk-linux-3.1.3.zip
export NEUTRON_SDK_DIR="$PWD/vendor/eiq-neutron-sdk"

# CLI (auto-detects format and backend)
imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
python -m imx95_npu.cli model.tflite                 # already-quantized: last mile only

# Install (heavy framework deps are optional extras)
pip install ".[full]"        # tensorflow + onnx2tf + torch
pip install ".[tensorflow]"  # or just one path: tensorflow | onnx | torch | dev

# Tests. SDK tests auto-skip if vendor/eiq-neutron-sdk is absent.
pip install ".[dev]" && pytest
NEUTRON_CONVERTER=tools/neutron-converter-stub.py pytest   # tflite backend w/o real binary
pytest tests/test_sdk.py -q                                # nxp backend (needs SDK)

# Docker (bundles the conversion stack; mount the SDK in at runtime)
docker build -t imx95-convert .

# docker compose: long-lived container, exec in for repeated conversions
cp .env.example .env            # MODELS_DIR (-> /data), NEUTRON_SDK_DIR (-> /opt/sdk:ro)
docker compose up -d --build
docker compose exec convert imx95-convert model.onnx --rep-data calib.npy --input-shape 3,224,224
```

The compose service overrides the image's CLI entrypoint with `sleep infinity` so it idles;
real work goes through `docker compose exec`. The SDK is bind-mounted read-only (never baked in).

Exit codes: `0` all converted, `1` backend/converter unavailable, `2` some failed.

Two host-environment gotchas (already handled in code): `TF_USE_LEGACY_KERAS=1` is set in
`__init__.py` because TF≥2.16 Keras 3 crashes the TFLite MLIR converter; and `numpy<2` is
required (TF 2.16 aborts under numpy 2). onnx2tf also fetches sample data from the network for
an accuracy check — `onnx_to_tflite` seeds a local file to keep it offline/deterministic.

## The neutron converter dependency

`neutron.resolve_converter` locates it three ways, in order: `PATH` lookup, an explicit
executable/`.py` path, then a `neutron_converter` Python module (`python3 -m neutron_converter`).
Default command comes from `$NEUTRON_CONVERTER` or `neutron-converter`. `tools/neutron-converter-stub.py`
is a dev/CI stand-in that copies the input + appends a marker (output is **not** board-loadable)
so the full pipeline and test suite run without the proprietary binary.
