"""``imx95-convert`` -- compile PyTorch / TensorFlow / ONNX / TFLite models into
a ``*_neutron.tflite`` for the i.MX 95 eIQ Neutron NPU.

Run on an x86 host with the NXP Neutron Converter installed (or point at the dev
stub with --converter-cmd / $NEUTRON_CONVERTER). Source models that aren't
already int8 are post-training-quantized using --rep-data.

Examples:
    imx95-convert model.tflite                      # already-quantized, last mile
    imx95-convert model.onnx  --rep-data calib.npy  --input-shape 3,224,224
    imx95-convert model.h5    --rep-data ./images/  --rep-norm 0to1
    imx95-convert model.pt    --rep-data calib.npy  --input-shape 3,224,224
    imx95-convert m.tflite -- --use-sequencer       # extra neutron flags after --
"""

import sys
import argparse

from . import neutron, sdk
from .repdata import RepDataSpec, NORMS, DEFAULT_MAX_SAMPLES
from .detect import ALL_FORMATS
from .pipeline import ConvertOptions, run


def _parse_input_shape(text):
    if text is None:
        return None
    try:
        dims = tuple(int(p) for p in text.replace("x", ",").split(",") if p != "")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--input-shape '{text}' must be comma-separated ints, e.g. 3,224,224"
        )
    if not dims:
        raise argparse.ArgumentTypeError("--input-shape is empty")
    return dims


def _split_extra(argv):
    """Split argv on the first standalone '--'; the tail is forwarded verbatim
    to the neutron converter (e.g. --use-sequencer, --fetch-constants-to-sram)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def build_parser():
    parser = argparse.ArgumentParser(
        prog="imx95-convert",
        description="Convert models to a *_neutron.tflite for the i.MX 95 NPU.",
        epilog="Anything after '--' is passed verbatim to the neutron converter.",
    )
    parser.add_argument("models", nargs="+", help="Input model file(s)/dir(s).")
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", *ALL_FORMATS),
        help="Source format (default: auto-detect).",
    )
    parser.add_argument(
        "--rep-data",
        default=None,
        help="Representative dataset for int8 PTQ: a .npy, an image directory, "
        "or a .py file exposing representative_dataset().",
    )
    parser.add_argument(
        "--rep-norm",
        default="none",
        choices=NORMS,
        help="Pixel normalization for an image-directory --rep-data (default: none).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Max calibration samples (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_input_shape,
        default=None,
        help="Input dims without batch, e.g. 3,224,224. Required for torch and "
        "dynamic-shape onnx.",
    )
    parser.add_argument(
        "--fail-on-float-ops",
        action="store_true",
        help="Error instead of warn if any op can't be int8-quantized.",
    )
    parser.add_argument(
        "--target",
        default=neutron.DEFAULT_TARGET,
        help=f"Neutron target (default: {neutron.DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write *_neutron.tflite (default: alongside each input).",
    )
    parser.add_argument(
        "--converter-cmd",
        default=neutron.default_converter_cmd(),
        help=f"Neutron converter executable/path for the tflite backend "
        f"(default: {neutron.DEFAULT_CONVERTER}; or set ${neutron.CONVERTER_ENV}).",
    )

    nxp = parser.add_argument_group("NXP eIQ Neutron SDK backend")
    nxp.add_argument(
        "--quant-backend",
        default="auto",
        choices=("auto", "nxp", "tflite"),
        help="Quantization backend: 'nxp' uses the SDK's tflite-profiler + "
        "tflite-quantizer; 'tflite' uses TFLiteConverter/onnx2tf PTQ; 'auto' "
        "(default) picks nxp when the SDK is available.",
    )
    nxp.add_argument(
        "--sdk-dir",
        default=None,
        help=f"Path to the extracted eIQ Neutron SDK (or set ${sdk.SDK_DIR_ENV}).",
    )
    nxp.add_argument(
        "--calibration-method",
        default="MinMax",
        choices=("MinMax", "Percentile"),
        help="tflite-quantizer calibration method (nxp backend). Default: MinMax.",
    )
    nxp.add_argument(
        "--percentile-val",
        type=float,
        default=None,
        help="Percentile clip value when --calibration-method=Percentile (e.g. 99.9).",
    )
    nxp.add_argument(
        "--float-io",
        action="store_true",
        help="Keep float32 graph inputs/outputs (int8 internals). Required by the "
             "GoPoint object-detection demos, which feed/read float32 tensors.",
    )
    nxp.add_argument(
        "--no-neutron",
        action="store_true",
        help="Emit the int8 TFLite ('<name>_int8.tflite') without NPU compilation. "
             "Runs on CPU/XNNPACK (demos' --backend cpu) at full int8 accuracy -- "
             "use when the NPU mis-quantizes a head (e.g. YOLO class scores).",
    )
    nxp.add_argument(
        "--yolo-raw-head",
        action="store_true",
        help="For YOLOv8/11: emit the head BEFORE box decoding (box DFL logits + "
             "class logits) and decode (DFL+dist2bbox+sigmoid) on the host. Keeps "
             "the NPU to convolutions only -- avoids the int8 box-decode precision "
             "loss. The patched demo auto-detects and decodes raw-head outputs.",
    )
    nxp.add_argument(
        "--optimization-level",
        default=None,
        choices=("OFast", "OOpt"),
        help="neutron-converter optimization level (OFast=fast, OOpt=optimal/slow).",
    )
    nxp.add_argument(
        "--validate",
        action="store_true",
        help="Dry-run the converted model in the converter (--run-after-generate).",
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    head, extra = _split_extra(argv)
    args = build_parser().parse_args(head)

    rep_data = None
    if args.rep_data is not None:
        rep_data = RepDataSpec(
            spec=args.rep_data,
            norm=args.rep_norm,
            max_samples=args.max_samples,
            input_shape=args.input_shape,
        )

    options = ConvertOptions(
        target=args.target,
        output_dir=args.output_dir,
        converter_cmd=args.converter_cmd,
        fmt=args.format,
        rep_data=rep_data,
        input_shape=args.input_shape,
        fail_on_float_ops=args.fail_on_float_ops,
        extra=extra,
        quant_backend=args.quant_backend,
        sdk_dir=args.sdk_dir,
        calibration_method=args.calibration_method,
        percentile_val=args.percentile_val,
        float_io=args.float_io,
        no_neutron=args.no_neutron,
        yolo_raw_head=args.yolo_raw_head,
        optimization_level=args.optimization_level,
        validate=args.validate,
    )

    _, exit_code = run(args.models, options)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
