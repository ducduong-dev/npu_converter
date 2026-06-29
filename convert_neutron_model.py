#!/usr/bin/env python3

"""
Copyright 2025 NXP

SPDX-License-Identifier: BSD-3-Clause

Host-side helper to compile a quantized TFLite model for the i.MX 95 eIQ Neutron
NPU using NXP's Neutron Converter, producing a '<name>_neutron.tflite' file that
the GoPoint demos auto-detect (object_detection_test.py, rtsp_object_detection.py
and face_recognition.py all prefer a '*_neutron.tflite' on i.MX 95).

IMPORTANT: this is a *development host* tool (x86_64 Linux). The Neutron Converter
is part of the NXP eIQ Toolkit / Yocto meta-imx SDK and is NOT present on the
target board -- run this on the machine where the converter is installed, then
copy the result into the board's downloads/ directory (or add it to
downloads.json).

The converter version MUST match the target's Neutron runtime (driver, firmware,
delegate -- all from the same SDK). With SDK >=3.1 a mismatch is reported at load
time as "Microcode version mismatch!"; older runtimes silently fell back to the
CPU. Confirm the board's version with:
    dpkg -s neutron | grep Version          # on the target

This script only handles the last mile (quantized TFLite -> neutron). To convert
from PyTorch / TensorFlow / ONNX, or to int8-quantize a float model first, use
the `imx95-convert` CLI (python -m imx95_npu.cli), which wraps this same step.

Examples:
    python3 convert_neutron_model.py model.tflite
    python3 convert_neutron_model.py a.tflite b.tflite --output-dir ./out
    python3 convert_neutron_model.py model.tflite --target imx95 \\
        --converter-cmd /opt/eiq/neutron-converter
"""

import sys
import argparse

from imx95_npu import neutron
from imx95_npu.quantize import warn_if_not_quantized


def main():
    parser = argparse.ArgumentParser(
        description="Compile quantized TFLite model(s) for the i.MX 95 Neutron NPU.",
        epilog="Run on an x86 host with the eIQ Neutron Converter installed.",
    )
    parser.add_argument("models", nargs="+", help="Input .tflite model file(s).")
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
        help=f"Converter executable or path (default: {neutron.DEFAULT_CONVERTER}).",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra args passed verbatim to the converter (after '--').",
    )
    args = parser.parse_args()

    converter = neutron.resolve_converter(args.converter_cmd)
    if converter is None:
        print(neutron.converter_not_found_message(args.converter_cmd))
        return 1

    print(f"Converter: {' '.join(converter)}")
    print(f"Target:    {args.target}")

    converted = []
    for model in args.models:
        print(f"\n==> {model}")
        warn_if_not_quantized(model)
        result = neutron.convert_one(
            converter, model, args.output_dir, args.target, args.extra
        )
        if result:
            converted.append(result)

    print(f"\nConverted {len(converted)}/{len(args.models)} model(s).")
    if converted:
        print(neutron.next_steps_message())
    return 0 if len(converted) == len(args.models) else 2


if __name__ == "__main__":
    sys.exit(main())
