"""TensorFlow -> int8 TFLite.

Handles SavedModel directories and Keras files (.keras / .h5). Frozen ``.pb``
graphs are not supported directly (TF2 dropped a clean path); the error tells
the user to re-export a SavedModel.

Quantization uses ``tf.lite.TFLiteConverter`` int8 PTQ (see ``quantize.py``),
driven by the shared representative dataset.
"""

import os

from . import ConvertResult, ConversionError
from .. import quantize
from ..repdata import RepDataError


def to_tflite(
    input_path,
    output_path,
    *,
    rep_data=None,
    input_shape=None,
    fail_on_float_ops=False,
    float_only=False,
):
    if rep_data is None and not float_only:
        raise ConversionError(
            "TensorFlow models need int8 quantization for the Neutron NPU; "
            "pass --rep-data (a .npy, an image directory, or a .py hook)."
        )

    import tensorflow as tf

    is_saved_model = os.path.isdir(input_path)
    if is_saved_model:
        model = None
        model_input_shape = _saved_model_input_shape(tf, input_path)
    else:
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pb":
            raise ConversionError(
                f"'{input_path}' looks like a frozen graph. Re-export it as a "
                "TF SavedModel directory (tf.saved_model.save) and pass that."
            )
        model = tf.keras.models.load_model(input_path, compile=False)
        model_input_shape = tuple(d for d in model.input_shape[1:] if d is not None)

    try:
        if float_only:
            # No quantization here: the NXP backend quantizes downstream with
            # tflite-profiler + tflite-quantizer. Just emit a float TFLite.
            tflite_bytes = quantize.tf_to_float_tflite(
                input_path if is_saved_model else model,
                from_saved_model=is_saved_model,
            )
        else:
            rep = rep_data.build(input_shape or model_input_shape)
            # TF/TFLite are NHWC; feed the representative dataset in NHWC.
            rep_fn = rep.tf_generator(layout="nhwc")
            if is_saved_model:
                tflite_bytes = quantize.quantize_saved_model_to_tflite(
                    input_path, rep_fn, fail_on_float_ops=fail_on_float_ops
                )
            else:
                tflite_bytes = quantize.quantize_keras_to_tflite(
                    model, rep_fn, fail_on_float_ops=fail_on_float_ops
                )
    except RepDataError as exc:
        raise ConversionError(str(exc)) from exc
    except Exception as exc:  # converter raises a variety of types
        raise ConversionError(f"TFLite conversion failed: {exc}") from exc

    with open(output_path, "wb") as handle:
        handle.write(tflite_bytes)

    quantized = quantize.is_int8_quantized(output_path)
    note = ""
    if not float_only and not quantized:
        note = "converter did not produce a full-int8 graph"
    native_shape = tuple(input_shape) if input_shape else model_input_shape
    return ConvertResult(output_path, quantized=quantized, notes=note,
                         input_shape=native_shape)


def _saved_model_input_shape(tf, saved_model_dir):
    """Best-effort input shape (no batch dim) from a SavedModel serving sig."""
    try:
        loaded = tf.saved_model.load(saved_model_dir)
        sig = loaded.signatures["serving_default"]
        spec = list(sig.structured_input_signature[1].values())[0]
        return tuple(int(d) for d in spec.shape[1:] if d is not None)
    except (KeyError, IndexError, ValueError, TypeError):
        return None
