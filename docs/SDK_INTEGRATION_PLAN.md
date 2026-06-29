# eIQ Neutron SDK 3.1.3 — Integration Plan

> **Status: IMPLEMENTED.** This plan has been built and verified end-to-end with the real SDK
> binaries (see `imx95_npu/sdk.py`, the `nxp` backend in `imx95_npu/pipeline.py`, and
> `tests/test_sdk.py`). The SDK is provisioned via `scripts/setup_sdk.sh` into `vendor/`
> (gitignored) and consumed at runtime through `$NEUTRON_SDK_DIR`. Docker mounts it read-only;
> `docker-compose.yml` runs a long-lived container you `exec` into. This document is kept as
> the design record. See `README.md` / `CLAUDE.md` for usage.

## What the SDK actually contains (verified by extracting & running it)

`eiq-neutron-sdk-linux-3.1.3.zip` (x86_64 Linux) ships real tools, not just the converter:

| Tool (`bin/`) | Role |
| --- | --- |
| `neutron-converter` | quantized TFLite → `*_neutron.tflite` (the last mile we already wrap) |
| `tflite-profiler` | float TFLite + calibration data → **profile CSV** (per-tensor min/max + histograms) |
| `tflite-quantizer` | float TFLite + profile CSV → **int8 TFLite** (NXP's own PTQ) |
| `tflite-optimizer` | graph-level TFLite optimizations |
| `tflite-profiler`/`neutron-runner` | on-host dry-run / golden validation, cycle estimates |
| `tflite-extractor` | inspect/extract NeutronGraph artifacts |

Plus `include/` + `lib/libNeutronConverter.a` (C API), `scripts/` (`convert_tensorflow.py`,
`resize_image.py`, `generate_random.py`, …), and `docs/`.

**Verified facts (ran the binaries on this host):**
- `neutron-converter --version` → `3.1.3+0X1e788118`; `--show-targets` lists **`imx95`** (our
  string is correct). Real flags: `--input` (required), `--target` (default `mcxn94x`),
  `--output` (optional, defaults to `<input>_converted`). Many extras: `--run-after-import`,
  `--run-after-generate`, `--dump-statistics`, `--optimization-level OFast|OOpt`,
  `--force-determinism`, `--verbose`.
- **End-to-end chain runs locally**: tiny float `.tflite` → `tflite-profiler --use-random-data`
  → `profile.csv` → `tflite-quantizer` → int8 `.tflite` (confirmed int8 by our
  `is_int8_quantized`) → `neutron-converter --target imx95` → valid `TFL3` neutron model.
- Microcode mismatch is **reported at runtime** on this version (not silent):
  `Microcode version mismatch! Neutron Driver is 3.1.0-… but model converted with 3.0.1-…`.

## The official quantization flow (this is the big change)

NXP's recommended path quantizes with **its own tools**, not TFLiteConverter/onnx2tf int8:

```
source ──to FLOAT tflite──► tflite-profiler ──► profile.csv ──► tflite-quantizer ──► int8 tflite ──► neutron-converter ──► *_neutron.tflite
                              (calibration data)                 (NXP PTQ)
```

This is strictly better for us than the current design because:
- Our source converters only need to emit **float** TFLite — no fragile int8 calibration
  inside onnx2tf, and no NHWC-vs-NCHW calibration-layout hazard.
- Quantization is done by the vendor tool that matches the NPU's expectations (per-channel
  weights, configurable schemas, layernorm exclusion, percentile calibration, etc.).
- The dev stub is replaced by the **real** converter for true end-to-end output.

We keep the existing TFLiteConverter/onnx2tf int8 path as a **fallback** for when the SDK
isn't present.

## Changes to the service

### 1. New module `imx95_npu/sdk.py` — wrap the SDK binaries
- `resolve_sdk()` — locate the SDK dir from `--sdk-dir` / `$NEUTRON_SDK_DIR`, else fall back
  to individual tools on `PATH` / `$NEUTRON_CONVERTER` (keeps stub support).
- `profile(float_tflite, dataset_dir|random, out_csv, histogram_bins)` → runs `tflite-profiler`.
- `quantize(float_tflite, profile_csv, out_tflite, **opts)` → runs `tflite-quantizer`
  (expose `--quantization-calibration-method`, `--quantization-percentile-val`,
  `--quantize-operator-types-except`, granularity/schema knobs as needed).
- `convert(quant_tflite, out, target, extra)` → reuse/move `neutron.convert_one`; add optional
  `--run-after-generate` sanity dry-run and `--dump-statistics` capture.
- `version()` / `show_targets()` — validate the target string and record the converter version
  in output (sha1 + version) for the `downloads.json`/board-sync story.

### 2. Quantization backend selector in `pipeline.py`
- New option `--quant-backend nxp|tflite|auto` (default `auto`: use `nxp` when the SDK is
  found, else `tflite`).
- `nxp` path:
  1. source → **float** TFLite (converters, below),
  2. build a calibration directory from `--rep-data` (see #4),
  3. `tflite-profiler` → profile CSV,
  4. `tflite-quantizer` → int8 TFLite,
  5. `neutron-converter`.
- `tflite` path: today's behavior (TFLiteConverter / onnx2tf int8 PTQ), unchanged.
- Pre-quantized int8 `.tflite` input still skips straight to `neutron-converter`.

### 3. Source converters: add a "float TFLite" mode
Each converter already produces TFLite; parameterize for float output (no PTQ):
- `tf_to_tflite` — drop optimizations/representative dataset when `quantize=False`.
- `onnx_to_tflite` — call onnx2tf **without** `output_integer_quantized_tflite`; grab
  `*_float32.tflite`. (Removes the onnx2tf int8 download/calibration hazards entirely on the
  NXP path.)
- `torch_to_tflite` — unchanged (torch → onnx → float onnx2tf).
- `tflite_passthrough` — float now allowed when backend is `nxp` (it can be quantized);
  still rejected on the `tflite`-only backend.

### 4. Calibration dataset for the profiler (`repdata.py` extension)
`tflite-profiler --dataset <dir>` consumes **raw float32 binary** files (one flattened sample
each, pre-normalized to the model's training input range). Add
`RepData.to_profiler_dataset(dir, layout="nhwc")` that writes each sample via
`ndarray.astype(float32).tofile(...)` — reusing all existing loaders (.npy / image dir / .py
hook) and the NCHW↔NHWC handling. If no `--rep-data`, fall back to
`tflite-profiler --use-random-data` with a loud accuracy warning.

### 5. CLI additions (`cli.py`)
`--sdk-dir`, `--quant-backend`, `--calibration-method MinMax|Percentile`,
`--percentile-val`, `--optimization-level OFast|OOpt`, `--validate` (turn on the converter/
quantizer dry-runs), and pass-through of advanced quantizer/converter flags. Keep all existing
flags working.

### 6. SDK provisioning (licensing-aware)
- **Do NOT commit or bake the SDK into git or the image** — `LA_OPT_NXP_Software_License.txt`
  is proprietary/non-redistributable. Add `eiq-neutron-sdk-*.zip` and `vendor/` to
  `.gitignore` and `.dockerignore`.
- Provide `scripts/setup_sdk.sh <zip> [dest]` that extracts to `vendor/eiq-neutron-sdk/` and
  prints the `NEUTRON_SDK_DIR` to export.
- Docker: mount at runtime (`-v /opt/eiq-neutron-sdk:/opt/sdk:ro -e NEUTRON_SDK_DIR=/opt/sdk`),
  same pattern as today's converter mount. Document a build-arg variant for CI that has a
  licensed copy.

### 7. Versioning / board sync
- Replace the hard-coded `REFERENCE_FIRMWARE = "3.0.0"` with the **actual converter version**
  queried from the binary, and surface it next to the sha1 in the "next steps" output.
- Document that the board's `NeutronFirmware.elf` / `libNeutronDriver.so` /
  `libneutron_delegate.so` must come from the **same** SDK (table is in
  `docs/NeutronSDKUserGuide.md`); optionally add a `scripts/deploy_runtime.sh` helper.

## Tests & verification
- New `tests/test_sdk.py`: skip if `$NEUTRON_SDK_DIR` unset; otherwise run the real
  profiler→quantizer→converter chain on a generated tiny float TFLite and assert an int8
  intermediate + a `TFL3` neutron output (mirrors the manual run already proven).
- Extend pipeline tests with `--quant-backend nxp` (SDK-gated) and keep the `tflite`-backend
  tests as the always-on path (stub converter).
- Keep the dev stub for the `tflite` backend / no-SDK CI.

## Out of scope / open questions
- Bundling the proprietary SDK anywhere public (licensing) — must stay mounted/provided.
- `tflite-optimizer` and `neutron-runner` golden-output validation: useful, propose as a
  follow-up `--validate=golden` once the core flow lands.
- The C API (`libNeutronConverter.a`) — not needed; the CLIs cover our use case.
