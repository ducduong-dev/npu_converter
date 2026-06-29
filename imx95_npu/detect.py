"""Source-format detection for input models.

The pipeline supports four source families. Detection prefers a content sniff
(magic bytes / structure) and falls back to the file extension, so a
mis-extensioned file is still handled correctly where possible.
"""

import os

# Canonical format names used throughout the package.
TORCH = "torch"
TENSORFLOW = "tensorflow"
ONNX = "onnx"
TFLITE = "tflite"

ALL_FORMATS = (TORCH, TENSORFLOW, ONNX, TFLITE)

_EXT_MAP = {
    ".pt": TORCH,
    ".pth": TORCH,
    ".onnx": ONNX,
    ".tflite": TFLITE,
    ".h5": TENSORFLOW,
    ".keras": TENSORFLOW,
    ".pb": TENSORFLOW,
}


class DetectionError(ValueError):
    """Raised when the source format cannot be determined."""


def _is_tflite(path):
    """TFLite/FlatBuffer files carry the 'TFL3' identifier at bytes 4..8."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] == b"TFL3"


def _is_zip(path):
    """PyTorch .pt/.pth (and .keras) are ZIP archives -> 'PK\\x03\\x04'."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _looks_like_onnx(path):
    """ONNX is a protobuf; the first field is usually ir_version (field 1,
    varint -> tag byte 0x08). Cheap heuristic, not a full parse."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(2)
    except OSError:
        return False
    return len(head) >= 1 and head[0] == 0x08


def detect_format(path):
    """Return one of ALL_FORMATS for ``path``.

    A TensorFlow SavedModel is a directory containing ``saved_model.pb``; that
    is checked first. For files we sniff content, then fall back to extension.
    """
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "saved_model.pb")):
            return TENSORFLOW
        raise DetectionError(
            f"'{path}' is a directory but not a TF SavedModel "
            "(no saved_model.pb). Pass a model file instead."
        )

    if not os.path.isfile(path):
        raise DetectionError(f"model not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    # Content sniff wins over extension where it is reliable.
    if _is_tflite(path):
        return TFLITE
    if ext in (".pt", ".pth") and _is_zip(path):
        return TORCH
    if ext == ".onnx" or (ext not in _EXT_MAP and _looks_like_onnx(path)):
        return ONNX

    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    # Keras v3 / torch share the ZIP container; disambiguate by extension above.
    if _is_zip(path):
        raise DetectionError(
            f"'{path}' is a zip-based model but the extension is ambiguous; "
            "pass --format torch|tensorflow."
        )

    raise DetectionError(
        f"could not determine model format for '{path}'. "
        "Pass --format torch|tensorflow|onnx|tflite."
    )
