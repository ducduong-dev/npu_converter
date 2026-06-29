"""PyTorch -> int8 TFLite, via ONNX.

PyTorch has no direct TFLite path, so we export to ONNX (``torch.onnx.export``)
and hand off to the onnx2tf-based converter. Export needs a concrete example
input, so ``--input-shape`` is REQUIRED for this path (an ONNX graph then
carries the shape for the rest of the pipeline).

The loaded object must be something exportable: a scripted/traced module, or a
state-dict-free ``nn.Module`` saved whole (``torch.save(model)``). A bare
state_dict can't be exported without the model class, which we don't have.
"""

import os
import tempfile

import numpy as np

from . import ConvertResult, ConversionError
from . import onnx_to_tflite


def to_tflite(
    input_path,
    output_path,
    *,
    rep_data=None,
    input_shape=None,
    fail_on_float_ops=False,
    float_only=False,
    yolo_raw_head=False,
):
    if input_shape is None:
        raise ConversionError(
            "PyTorch models need --input-shape (e.g. 3,224,224) to export to "
            "ONNX. Give the input dims without the batch dimension."
        )
    if rep_data is None and not float_only:
        raise ConversionError(
            "PyTorch models need int8 quantization for the Neutron NPU; pass "
            "--rep-data (a .npy, an image directory, or a .py hook)."
        )

    import torch

    model = _load_module(torch, input_path)
    model.eval()

    example = torch.from_numpy(
        np.zeros((1, *tuple(int(d) for d in input_shape)), dtype=np.float32)
    )

    work_dir = tempfile.mkdtemp(prefix="torch2onnx_")
    onnx_path = os.path.join(work_dir, "model.onnx")
    try:
        try:
            _export_onnx(torch, model, example, onnx_path)
        except Exception as exc:
            raise ConversionError(f"torch.onnx.export failed: {exc}") from exc

        # Reuse the ONNX path for the heavy lifting (onnx2tf float or int8 PTQ).
        return onnx_to_tflite.to_tflite(
            onnx_path,
            output_path,
            rep_data=rep_data,
            input_shape=input_shape,
            fail_on_float_ops=fail_on_float_ops,
            float_only=float_only,
            yolo_raw_head=yolo_raw_head,
        )
    finally:
        # Keep nothing but the final output; onnx2tf cleans its own temp dir.
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


def _export_onnx(torch, model, example, onnx_path):
    """Export to ONNX via the stable TorchScript path.

    Newer torch defaults to the dynamo exporter, which needs the optional
    'onnxscript' package; we force the legacy TorchScript exporter (dynamo=False)
    for portability and fall back for torch versions without that kwarg.
    """
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, example, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        # Older torch: no 'dynamo' kwarg, legacy exporter is the only path.
        torch.onnx.export(model, example, onnx_path, **kwargs)


def _load_module(torch, input_path):
    """Load a torch model object, rejecting bare state_dicts with guidance."""
    try:
        obj = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ConversionError(f"could not torch.load '{input_path}': {exc}") from exc

    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        raise ConversionError(
            f"'{input_path}' is a state_dict, not a model. Re-save the whole "
            "module (torch.save(model, path)) or a TorchScript export "
            "(torch.jit.script(model).save(path)); the model class isn't "
            "available here to rebuild it."
        )
    # TorchScript modules load as ScriptModule (an nn.Module subclass) -> handled.
    raise ConversionError(
        f"unsupported torch object in '{input_path}': {type(obj).__name__}."
    )
