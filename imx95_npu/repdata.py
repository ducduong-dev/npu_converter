"""Representative dataset loading for int8 post-training quantization.

PTQ needs a few hundred real input samples to calibrate activation ranges. One
loader feeds both quantization backends:

* the TensorFlow path consumes a ``representative_dataset`` generator
  (``tf_generator``) yielding ``[sample]`` batches;
* the ONNX/Torch path (onnx2tf) consumes a calibration ``.npy`` file
  (``to_calibration_npy``).

Three input forms are accepted via ``--rep-data``:

* ``*.npy``      -- an array shaped ``(N, *input_shape)``, already preprocessed.
* a directory   -- images loaded, resized to the model input, and normalized
                    per ``--rep-norm``.
* ``*.py``       -- a Python file exposing ``representative_dataset()`` (the
                    TFLite generator protocol); used verbatim, an escape hatch
                    for custom preprocessing.

NumPy is the only hard dependency here; Pillow is imported lazily for images.
"""

import os
import glob
import importlib.util
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
DEFAULT_MAX_SAMPLES = 200

# Supported pixel normalizations for image-directory rep data. Must match the
# preprocessing the model was trained with, or calibration ranges will be off.
NORMS = ("none", "0to1", "-1to1", "imagenet")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_image(arr, norm):
    """Apply ``norm`` to an HWC float32 image in 0..255 units."""
    if norm == "none":
        return arr
    if norm == "0to1":
        return arr / 255.0
    if norm == "-1to1":
        return (arr - 127.5) / 127.5
    if norm == "imagenet":
        return (arr / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    raise RepDataError(f"unknown normalization '{norm}'")


class RepDataError(ValueError):
    """Raised for malformed representative-dataset inputs."""


@dataclass
class RepDataSpec:
    """User-supplied representative-dataset request, not yet bound to a shape.

    The input shape often comes from the model (Keras input, ONNX graph input),
    which converters only know after loading it. The pipeline builds this from
    CLI args; each converter calls :meth:`build` with the resolved shape.
    """

    spec: str  # path: .npy | image dir | .py
    norm: str = "none"
    max_samples: int = DEFAULT_MAX_SAMPLES
    input_shape: Optional[Sequence[int]] = None  # CLI override (no batch dim)

    def build(self, input_shape=None):
        """Resolve to a :class:`RepData`. A CLI-provided shape wins over the
        model-derived ``input_shape``."""
        shape = self.input_shape or input_shape
        if shape is None and not self._is_py():
            raise RepDataError(
                "could not determine the model input shape for the "
                "representative dataset; pass --input-shape."
            )
        return RepData(
            self.spec,
            tuple(shape) if shape is not None else (),
            norm=self.norm,
            max_samples=self.max_samples,
        )

    def _is_py(self):
        return isinstance(self.spec, str) and self.spec.endswith(".py")


def _channels_position(shape):
    """Return 'nchw' or 'nhwc' for a 3-D (C,H,W)/(H,W,C) input shape, else None."""
    if len(shape) == 3:
        if shape[0] in (1, 3):
            return "nchw"
        if shape[2] in (1, 3):
            return "nhwc"
    return None


class RepData:
    """A resolved representative dataset bound to a model input shape/layout.

    ``input_shape`` excludes the batch dimension (e.g. ``(3, 224, 224)`` or
    ``(224, 224, 3)``). ``layout`` is the layout the *consumer* wants samples
    in ('nchw' for the ONNX/torch graph, 'nhwc' for TFLite); images are
    transposed accordingly.
    """

    def __init__(self, spec, input_shape, norm="none", max_samples=DEFAULT_MAX_SAMPLES):
        self.spec = spec
        self.input_shape = tuple(int(d) for d in input_shape)
        self.norm = norm
        self.max_samples = max_samples
        if norm not in NORMS:
            raise RepDataError(
                f"unknown --rep-norm '{norm}'; choose from {list(NORMS)}"
            )

    # -- public API ---------------------------------------------------------

    def py_generator(self):
        """If spec is a .py file exposing representative_dataset(), return it."""
        if isinstance(self.spec, str) and self.spec.endswith(".py"):
            return _load_py_generator(self.spec)
        return None

    def samples(self, layout="nhwc"):
        """Yield up to ``max_samples`` float32 arrays of shape ``input_shape``
        (no batch dim), in the requested ``layout``."""
        custom = self.py_generator()
        if custom is not None:
            for batch in custom():
                arr = np.asarray(batch[0], dtype=np.float32)
                yield np.squeeze(arr, axis=0) if arr.ndim == len(self.input_shape) + 1 else arr
            return

        if os.path.isfile(self.spec) and self.spec.endswith(".npy"):
            yield from self._from_npy(layout)
            return

        if os.path.isdir(self.spec):
            yield from self._from_image_dir(layout)
            return

        raise RepDataError(
            f"--rep-data '{self.spec}' is not a .npy file, a directory of "
            "images, or a .py file exposing representative_dataset()."
        )

    def tf_generator(self, layout="nhwc"):
        """A TFLite ``representative_dataset`` callable (yields ``[sample]``)."""
        custom = self.py_generator()
        if custom is not None:
            return custom

        def gen():
            for sample in self.samples(layout=layout):
                yield [sample[np.newaxis, ...].astype(np.float32)]

        return gen

    def to_calibration_npy(self, out_path, layout="nchw"):
        """Write a ``(N, *input_shape)`` calibration array for onnx2tf."""
        stack = [s for s in self.samples(layout=layout)]
        if not stack:
            raise RepDataError("representative dataset produced no samples.")
        arr = np.stack(stack, axis=0).astype(np.float32)
        np.save(out_path, arr)
        return out_path, arr.shape

    def to_profiler_dataset(self, out_dir, layout="nhwc"):
        """Write one raw float32 binary file per sample for tflite-profiler.

        The NXP profiler's ``--dataset <dir>`` reads flattened float32 sample
        files (alphabetical order) in the model's input range. Samples are
        emitted in ``layout`` (NHWC for a TFLite model). Returns (dir, count).
        """
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        for i, sample in enumerate(self.samples(layout=layout)):
            path = os.path.join(out_dir, f"sample_{i:05d}.bin")
            sample.astype(np.float32).tofile(path)
            count += 1
        if count == 0:
            raise RepDataError("representative dataset produced no samples.")
        return out_dir, count

    # -- loaders ------------------------------------------------------------

    def _from_npy(self, layout):
        """Yield rows of a ``(N, *input_shape)`` array, transposed to ``layout``.

        The array is assumed to be in the model's native input layout (NCHW for
        ONNX/Torch, NHWC for TF), inferred from ``input_shape``; we transpose to
        whatever the consuming backend wants."""
        arr = np.load(self.spec)
        if arr.ndim == len(self.input_shape):
            arr = arr[np.newaxis, ...]  # a single un-batched sample
        src = _channels_position(self.input_shape)
        for i, sample in enumerate(arr):
            if i >= self.max_samples:
                break
            sample = sample.astype(np.float32)
            if src is not None:
                sample = _to_layout(sample, src, layout)
            yield sample

    def _from_image_dir(self, layout):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RepDataError(
                "Pillow is required to use an image directory as --rep-data; "
                "install it or pass a .npy / .py representative dataset."
            ) from exc

        pos = _channels_position(self.input_shape)
        if pos == "nchw":
            channels, height, width = self.input_shape
        elif pos == "nhwc":
            height, width, channels = self.input_shape
        else:
            raise RepDataError(
                f"cannot infer H/W/C from input shape {self.input_shape}; "
                "use a .npy representative dataset instead."
            )

        files = sorted(
            f
            for f in glob.glob(os.path.join(self.spec, "**", "*"), recursive=True)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        if not files:
            raise RepDataError(f"no images found under '{self.spec}'.")

        for path in files[: self.max_samples]:
            mode = "L" if channels == 1 else "RGB"
            img = Image.open(path).convert(mode).resize((width, height))
            arr = np.asarray(img, dtype=np.float32)  # HWC (or HW for L)
            if channels == 1:
                arr = arr[..., np.newaxis]
            arr = _normalize_image(arr, self.norm)
            yield _to_layout(arr, "nhwc", layout)


def _to_layout(arr, src, dst):
    """Transpose a single 3-D sample between 'nhwc' and 'nchw'."""
    if src == dst or arr.ndim != 3:
        return arr.astype(np.float32)
    if src == "nhwc" and dst == "nchw":
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    if src == "nchw" and dst == "nhwc":
        return np.transpose(arr, (1, 2, 0)).astype(np.float32)
    return arr.astype(np.float32)


def _load_py_generator(py_path):
    """Import ``py_path`` and return its ``representative_dataset`` callable."""
    spec = importlib.util.spec_from_file_location("_rep_data_hook", py_path)
    if spec is None or spec.loader is None:
        raise RepDataError(f"could not import representative dataset from {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "representative_dataset", None)
    if fn is None or not callable(fn):
        raise RepDataError(
            f"{py_path} must define a callable 'representative_dataset()'."
        )
    return fn
