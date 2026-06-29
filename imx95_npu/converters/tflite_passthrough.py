"""TFLite source 'conversion': just validate and pass the file through.

A bare float ``.tflite`` cannot be re-quantized -- TFLite int8 PTQ needs the
original graph (SavedModel/Keras), which we no longer have. So an int8 model is
accepted as-is, and a float model is a hard error pointing the user at the
source model or a pre-quantized export.
"""

import os
import shutil

from . import ConvertResult, ConversionError
from .. import quantize


def to_tflite(
    input_path,
    output_path,
    *,
    rep_data=None,
    input_shape=None,
    fail_on_float_ops=False,
    float_only=False,
):
    del input_shape, fail_on_float_ops  # not applicable to pass-through

    if not os.path.isfile(input_path):
        raise ConversionError(f"TFLite model not found: {input_path}")

    quantized = quantize.is_int8_quantized(input_path)
    native_shape = quantize.tflite_input_shape(input_path)
    if not quantized and float_only:
        # NXP backend: a float .tflite is fine here; it gets quantized downstream
        # by tflite-profiler + tflite-quantizer.
        _copy(input_path, output_path, rep_data=None)
        return ConvertResult(output_path, quantized=False, input_shape=native_shape)
    if not quantized:
        dtype = quantize.input_dtype(input_path)
        if dtype is None:
            # No TFLite runtime to inspect with: don't block, but flag it.
            note = (
                "could not verify quantization (no TFLite runtime on host); "
                "assuming the model is already int8. The Neutron converter will "
                "reject it otherwise."
            )
            _copy(input_path, output_path, rep_data)
            return ConvertResult(output_path, quantized=False, notes=note)
        raise ConversionError(
            f"'{input_path}' is a float TFLite model (input dtype '{dtype}'). "
            "A bare .tflite cannot be re-quantized -- the Neutron NPU needs int8. "
            "Provide the original TF/ONNX/Torch model with --rep-data so it can "
            "be int8-quantized, or export a pre-quantized .tflite."
        )

    _copy(input_path, output_path, rep_data)
    return ConvertResult(output_path, quantized=True, input_shape=native_shape)


def _copy(input_path, output_path, rep_data):
    if rep_data is not None:
        print(
            "  note: input is already int8 TFLite; ignoring --rep-data "
            "(re-quantization of a finished .tflite is not supported)."
        )
    if os.path.abspath(input_path) != os.path.abspath(output_path):
        shutil.copyfile(input_path, output_path)
