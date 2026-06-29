"""Per-format converters that normalize a source model into a TFLite model.

Each converter exposes ``to_tflite(input_path, output_path, *, rep_data,
input_shape, fail_on_float_ops) -> ConvertResult``. They are imported lazily by
``pipeline.py`` so that a job for one format never imports another framework's
heavy dependencies (importing torch + tensorflow eagerly is slow and can clash).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ConvertResult:
    """Outcome of a source -> TFLite conversion."""

    tflite_path: str
    quantized: bool  # True if the resulting TFLite is int8/uint8
    notes: str = ""
    # The model's NATIVE input shape (no batch dim): NHWC for tf/tflite sources,
    # NCHW for onnx/torch. The NXP backend uses this to lay out calibration data.
    input_shape: Optional[Tuple[int, ...]] = None


class ConversionError(RuntimeError):
    """Raised when a source model cannot be converted to TFLite."""
