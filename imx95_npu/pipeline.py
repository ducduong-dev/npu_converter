"""Orchestration: source model -> int8 TFLite -> *_neutron.tflite.

Two quantization backends:

* ``nxp`` (preferred when the eIQ Neutron SDK is available): each source is
  normalized to a FLOAT TFLite, then quantized with the SDK's official tools
  (tflite-profiler + tflite-quantizer), then compiled with neutron-converter.
* ``tflite`` (fallback): the source converters quantize to int8 themselves
  (TFLiteConverter / onnx2tf PTQ), then neutron-converter compiles it.

``auto`` picks ``nxp`` when the full SDK toolchain is found, else ``tflite``.
Converters are imported lazily inside :func:`_to_tflite` so a TFLite/ONNX job
never pays to import torch + tensorflow.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from . import neutron, detect
from .repdata import RepDataSpec, RepDataError
from .sdk import NeutronSDK

NXP = "nxp"
TFLITE = "tflite"
AUTO = "auto"


@dataclass
class ConvertOptions:
    """Everything one run needs, assembled from CLI args."""

    target: str = neutron.DEFAULT_TARGET
    output_dir: Optional[str] = None
    converter_cmd: str = field(default_factory=neutron.default_converter_cmd)
    fmt: str = "auto"  # 'auto' or a detect.* name
    rep_data: Optional[RepDataSpec] = None
    input_shape: Optional[tuple] = None  # no batch dim
    fail_on_float_ops: bool = False
    extra: list = field(default_factory=list)  # passed verbatim to neutron-converter
    # NXP-backend options
    quant_backend: str = AUTO  # nxp | tflite | auto
    sdk_dir: Optional[str] = None
    calibration_method: str = "MinMax"  # MinMax | Percentile
    percentile_val: Optional[float] = None
    float_io: bool = False  # keep float32 graph I/O (int8 internals) for the demos
    no_neutron: bool = False  # emit the int8 TFLite (CPU/XNNPACK), skip neutron compile
    yolo_raw_head: bool = False  # emit pre-decode YOLO head (box DFL + class logits)
    optimization_level: Optional[str] = None  # OFast | OOpt
    validate: bool = False  # run converter/quantizer dry-runs


@dataclass
class JobResult:
    input_path: str
    output_path: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None


def _to_tflite(fmt, input_path, output_path, options, *, float_only):
    """Dispatch to the format-specific converter (imported lazily)."""
    kwargs = dict(
        rep_data=options.rep_data,
        input_shape=options.input_shape,
        fail_on_float_ops=options.fail_on_float_ops,
        float_only=float_only,
    )
    if fmt == detect.TFLITE:
        from .converters import tflite_passthrough

        return tflite_passthrough.to_tflite(input_path, output_path, **kwargs)
    if fmt == detect.TENSORFLOW:
        from .converters import tf_to_tflite

        return tf_to_tflite.to_tflite(input_path, output_path, **kwargs)
    if fmt == detect.ONNX:
        from .converters import onnx_to_tflite

        return onnx_to_tflite.to_tflite(
            input_path, output_path, yolo_raw_head=options.yolo_raw_head, **kwargs)
    if fmt == detect.TORCH:
        from .converters import torch_to_tflite

        return torch_to_tflite.to_tflite(
            input_path, output_path, yolo_raw_head=options.yolo_raw_head, **kwargs)
    raise ValueError(f"unknown format '{fmt}'")


def _converter_extra(options):
    """Build the neutron-converter passthrough flags from options."""
    flags = list(options.extra)
    if options.optimization_level:
        flags += ["--optimization-level", options.optimization_level]
    return flags


# -- NXP backend ------------------------------------------------------------


def _quantize_with_sdk(sdk, float_path, work_dir, stem, native_shape, options):
    """Profiler + quantizer -> int8 TFLite path (named <stem>.tflite)."""
    int8_path = os.path.join(work_dir, f"{stem}.tflite")

    dataset_dir = None
    if options.rep_data is not None:
        shape = options.input_shape or native_shape
        rep = options.rep_data.build(shape)
        dataset_dir = rep.to_profiler_dataset(
            os.path.join(work_dir, "calib"), layout="nhwc"
        )[0]

    profile_csv = sdk.profile(
        float_path, os.path.join(work_dir, f"{stem}_profile.csv"),
        dataset_dir=dataset_dir,
    )
    quant_extra = None
    if options.float_io:
        # Keep float32 graph placeholders so the GoPoint demos can feed/read
        # float tensors (the converter inserts quantize/dequantize at the edges).
        # abseil bool flags need the '=' form; '--flag false' leaves it true.
        quant_extra = ["--quantize-inputs=false", "--quantize-outputs=false"]
    sdk.quantize(
        float_path, profile_csv, int8_path,
        calibration_method=options.calibration_method,
        percentile_val=options.percentile_val,
        extra=quant_extra,
    )
    return int8_path


def _convert_nxp(input_path, options, sdk):
    from .converters import ConversionError

    try:
        fmt = options.fmt if options.fmt != "auto" else detect.detect_format(input_path)
    except detect.DetectionError as exc:
        return JobResult(input_path, error=str(exc))
    print(f"  format: {fmt} | backend: nxp")

    work_dir = tempfile.mkdtemp(prefix="imx95_")
    stem = os.path.splitext(os.path.basename(input_path.rstrip("/")))[0]
    float_path = os.path.join(work_dir, f"{stem}_float.tflite")

    try:
        result = _to_tflite(fmt, input_path, float_path, options, float_only=True)
        if result.quantized:
            # Source was already int8 -> skip profiler/quantizer.
            int8_path = os.path.join(work_dir, f"{stem}.tflite")
            os.replace(result.tflite_path, int8_path)
        else:
            int8_path = _quantize_with_sdk(
                sdk, result.tflite_path, work_dir, stem, result.input_shape, options
            )
    except (ConversionError, RepDataError) as exc:
        _cleanup(work_dir)
        return JobResult(input_path, error=str(exc))
    except Exception as exc:  # surface unexpected SDK/converter errors
        _cleanup(work_dir)
        return JobResult(input_path, error=f"unexpected error: {exc}")

    out_dir = options.output_dir or os.path.dirname(os.path.abspath(input_path))

    if options.no_neutron:
        # Emit the int8 TFLite without compiling for the NPU. It runs on
        # CPU/XNNPACK (the demos' --backend cpu) at full int8 accuracy -- the
        # escape hatch when the NPU mis-quantizes a head (e.g. YOLO classes).
        os.makedirs(out_dir, exist_ok=True)
        cpu_path = os.path.join(out_dir, f"{stem}_int8.tflite")
        shutil.copyfile(int8_path, cpu_path)
        _cleanup(work_dir)
        print(f"  OK -> {cpu_path}  (int8, CPU/XNNPACK; not for the NPU delegate)")
        print(f"     sha1: {neutron.sha1(cpu_path)}")
        return JobResult(input_path, output_path=cpu_path, ok=True)

    try:
        output_path = sdk.convert(
            int8_path, out_dir, options.target,
            extra=_converter_extra(options),
            run_after_generate=options.validate,
        )
    except Exception as exc:
        _cleanup(work_dir)
        return JobResult(input_path, error=f"neutron-converter failed: {exc}")
    _cleanup(work_dir)

    if output_path is None:
        return JobResult(input_path, error="neutron converter failed")
    return JobResult(input_path, output_path=output_path, ok=True)


# -- TFLite (fallback) backend ----------------------------------------------


def _convert_tflite(input_path, options, converter):
    from .converters import ConversionError

    try:
        fmt = options.fmt if options.fmt != "auto" else detect.detect_format(input_path)
    except detect.DetectionError as exc:
        return JobResult(input_path, error=str(exc))
    print(f"  format: {fmt} | backend: tflite")

    work_dir = tempfile.mkdtemp(prefix="imx95_")
    stem = os.path.splitext(os.path.basename(input_path.rstrip("/")))[0]
    staged = os.path.join(work_dir, f"{stem}.tflite")

    try:
        result = _to_tflite(fmt, input_path, staged, options, float_only=False)
    except (ConversionError, RepDataError) as exc:
        _cleanup(work_dir)
        return JobResult(input_path, error=str(exc))
    except Exception as exc:
        _cleanup(work_dir)
        return JobResult(input_path, error=f"unexpected conversion error: {exc}")

    if not result.quantized:
        msg = result.notes or "model is not int8-quantized"
        print(f"  WARNING: {msg}; Neutron acceleration may be partial.")

    out_dir = options.output_dir or os.path.dirname(os.path.abspath(input_path))
    output_path = neutron.convert_one(
        converter, staged, out_dir, options.target, _converter_extra(options)
    )
    _cleanup(work_dir)

    if output_path is None:
        return JobResult(input_path, error="neutron converter failed")
    return JobResult(input_path, output_path=output_path, ok=True)


# -- driver -----------------------------------------------------------------


def _select_backend(options):
    """Return ('nxp', sdk) or ('tflite', converter_argv), or (None, error_msg)."""
    requested = options.quant_backend
    sdk = NeutronSDK.resolve(options.sdk_dir) if requested in (NXP, AUTO) else None

    if requested == NXP:
        if not sdk.has_full_toolchain():
            return None, (
                "--quant-backend nxp requires the eIQ Neutron SDK "
                "(tflite-profiler, tflite-quantizer, neutron-converter). Set "
                "--sdk-dir / $NEUTRON_SDK_DIR (run scripts/setup_sdk.sh)."
            )
        return NXP, sdk

    if requested == AUTO and sdk is not None and sdk.has_full_toolchain():
        return NXP, sdk

    # tflite backend (explicit, or auto without a full SDK)
    converter = neutron.resolve_converter(options.converter_cmd)
    if converter is None:
        return None, neutron.converter_not_found_message(options.converter_cmd)
    return TFLITE, converter


def run(models, options):
    """Convert every model. Returns (results, exit_code)."""
    backend, handle = _select_backend(options)
    if backend is None:
        print(handle)  # error message
        return [], 1

    if backend == NXP:
        sdk = handle
        print(f"Backend:   nxp ({sdk.version() or 'eIQ Neutron SDK'})")
        print(f"Target:    {options.target}")
        convert = lambda m: _convert_nxp(m, options, sdk)  # noqa: E731
    else:
        converter = handle
        print(f"Backend:   tflite (converter: {' '.join(converter)})")
        print(f"Target:    {options.target}")
        convert = lambda m: _convert_tflite(m, options, converter)  # noqa: E731

    results = []
    for model in models:
        print(f"\n==> {model}")
        res = convert(model)
        if res.ok:
            print(f"  done: {res.output_path}")
        else:
            print(f"  FAILED: {res.error}")
        results.append(res)

    ok = sum(1 for r in results if r.ok)
    print(f"\nConverted {ok}/{len(models)} model(s).")
    if ok and options.no_neutron:
        print("Next: copy the *_int8.tflite file(s) to the board and run them on "
              "CPU (e.g. the GoPoint demo's --backend cpu). They are NOT for the "
              "Neutron delegate.")
    elif ok:
        print(neutron.next_steps_message())
    return results, (0 if ok == len(models) else 2)


def _cleanup(work_dir):
    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)
