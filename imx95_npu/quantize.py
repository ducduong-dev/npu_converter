"""Quantization helpers.

The Neutron NPU only accelerates integer-quantized graphs. This module knows
how to (a) tell whether a TFLite model is already int8/uint8 quantized and
(b) drive ``tf.lite.TFLiteConverter`` int8 post-training quantization for the
TensorFlow path. The ONNX/Torch paths quantize inside onnx2tf instead (see
``converters/onnx_to_tflite.py``); they share the same representative dataset.

TensorFlow is imported lazily so that importing this module (and the legacy
``convert_neutron_model.py`` script) stays cheap and dependency-free.
"""

INT_DTYPES = ("int8", "uint8")


def _load_interpreter(model_path):
    """Return an allocated TFLite Interpreter, trying tflite_runtime first."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def input_dtype(model_path):
    """Return the input tensor dtype name (e.g. 'int8'), or None if unknown."""
    try:
        interpreter = _load_interpreter(model_path)
        return interpreter.get_input_details()[0]["dtype"].__name__
    except ImportError:
        return None
    except (ValueError, RuntimeError):
        return None


def tflite_input_shape(model_path):
    """Return the TFLite model's input shape without the batch dim (NHWC), or
    None if it can't be read. Used to size calibration data for a .tflite source."""
    try:
        interpreter = _load_interpreter(model_path)
        shape = interpreter.get_input_details()[0]["shape"]
        dims = tuple(int(d) for d in shape)
        return dims[1:] if len(dims) > 1 else dims
    except ImportError:
        return None
    except (ValueError, RuntimeError, KeyError, IndexError):
        return None


def is_int8_quantized(model_path):
    """True if the model's input tensor is int8/uint8.

    Returns False when the model is float OR when no TFLite runtime is available
    to inspect it -- callers treat 'unknown' as 'not proven quantized' and act
    conservatively (warn / require a representative dataset).
    """
    return input_dtype(model_path) in INT_DTYPES


def warn_if_not_quantized(path):
    """Best-effort warning that a model isn't integer-quantized. Used by the
    legacy script and the pipeline's pass-through path. Silent if no runtime."""
    dtype = input_dtype(path)
    if dtype is None:
        return  # cannot inspect on this host; let the converter decide
    if dtype not in INT_DTYPES:
        print(
            f"  WARNING: input dtype is '{dtype}', not int8/uint8. The Neutron "
            "NPU only accelerates integer-quantized graphs; float parts stay "
            "on the CPU. Provide a representative dataset (--rep-data) to "
            "int8-quantize, or pass a fully-quantized model."
        )


def tf_to_float_tflite(model_or_dir, *, from_saved_model):
    """Convert a Keras model or SavedModel dir to a FLOAT TFLite (no PTQ).

    Used by the NXP backend, which quantizes downstream with the SDK's
    tflite-profiler + tflite-quantizer rather than TFLiteConverter."""
    import tensorflow as tf

    if from_saved_model:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_or_dir)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model_or_dir)
    return converter.convert()


def quantize_keras_to_tflite(model, rep_dataset_fn, *, fail_on_float_ops=False):
    """Int8 post-training-quantize a Keras model to TFLite bytes.

    ``rep_dataset_fn`` is a zero-arg callable yielding lists of input arrays
    (the TFLite ``representative_dataset`` protocol). Inputs and outputs are
    forced to int8 so the whole graph is integer -- which is what Neutron wants.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return _run_int8_converter(converter, tf, rep_dataset_fn, fail_on_float_ops)


def quantize_saved_model_to_tflite(
    saved_model_dir, rep_dataset_fn, *, fail_on_float_ops=False
):
    """Int8 post-training-quantize a TF SavedModel directory to TFLite bytes."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    return _run_int8_converter(converter, tf, rep_dataset_fn, fail_on_float_ops)


def _run_int8_converter(converter, tf, rep_dataset_fn, fail_on_float_ops):
    """Shared int8 PTQ configuration for TFLiteConverter."""
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_fn
    # Force a full-integer graph. TFLITE_BUILTINS_INT8 (without the float
    # fallback op set) makes the converter error on ops it can't quantize,
    # which is what we want when fail_on_float_ops is set.
    if fail_on_float_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()
