"""ONNX -> int8 TFLite via onnx2tf.

onnx2tf converts the ONNX graph to a TF/TFLite model and, with a calibration
dataset, emits a full-integer ``*_full_integer_quant.tflite``. The Torch path
reuses this module after exporting to ONNX.

Calibration data is written by the shared representative-dataset loader in the
ONNX (NCHW) layout and passed via onnx2tf's custom-calibration hook with
mean=0/std=1 (we pre-normalize the samples ourselves), so onnx2tf runs the float
model on exactly the values we provide.
"""

import os
import glob
import shutil
import tempfile

from . import ConvertResult, ConversionError
from .. import quantize
from ..repdata import RepDataError


def onnx_input_info(onnx_path):
    """Return (input_op_name, input_shape_without_batch_or_None)."""
    import onnx

    model = onnx.load(onnx_path)
    graph_input = model.graph.input[0]
    name = graph_input.name
    dims = graph_input.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        shape.append(d.dim_value if (d.HasField("dim_value") and d.dim_value > 0) else None)
    body = tuple(shape[1:]) if len(shape) > 1 else tuple(shape)
    if any(d is None for d in body):
        body = None  # dynamic dims -> caller must supply --input-shape
    return name, body


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
    if rep_data is None and not float_only:
        raise ConversionError(
            "ONNX models need int8 quantization for the Neutron NPU; pass "
            "--rep-data (a .npy, an image directory, or a .py hook)."
        )

    try:
        import onnx2tf  # noqa: F401
    except ImportError as exc:
        raise ConversionError(
            "onnx2tf is required for the ONNX/Torch path; pip install onnx2tf."
        ) from exc

    input_name, model_shape = onnx_input_info(input_path)

    work_dir = tempfile.mkdtemp(prefix="onnx2tf_")
    try:
        calib_npy = None
        if not float_only:
            # onnx2tf converts the graph to NHWC, and its calibration data must
            # match that converted input layout (repdata transposes from the
            # model's native NCHW for us).
            try:
                rep = rep_data.build(input_shape or model_shape)
            except RepDataError as exc:
                raise ConversionError(str(exc)) from exc
            calib_npy = os.path.join(work_dir, "calib.npy")
            rep.to_calibration_npy(calib_npy, layout="nhwc")

        onnx_for_tf = _sanitize_onnx_names(os.path.abspath(input_path), work_dir)
        if yolo_raw_head:
            onnx_for_tf = _yolo_raw_head(onnx_for_tf, work_dir)
        else:
            onnx_for_tf = _split_yolo_outputs(onnx_for_tf, work_dir)
        _run_onnx2tf(onnx_for_tf, work_dir, input_name, calib_npy,
                     abs_input_path=onnx_for_tf, float_only=float_only)

        produced = _find_float_tflite(work_dir) if float_only else _find_int8_tflite(work_dir)
        if produced is None:
            kind = "*_float32.tflite" if float_only else "*_full_integer_quant.tflite"
            raise ConversionError(f"onnx2tf ran but produced no {kind} under {work_dir}.")
        shutil.copyfile(produced, output_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    quantized = quantize.is_int8_quantized(output_path)
    note = ""
    if not float_only and not quantized:
        note = "onnx2tf did not produce a full-int8 graph"
        if fail_on_float_ops:
            raise ConversionError("onnx2tf output is not fully int8-quantized.")
    native_shape = tuple(input_shape) if input_shape else model_shape
    return ConvertResult(output_path, quantized=quantized, notes=note,
                         input_shape=native_shape)


# onnx2tf fetches a 20x128x128x3 sample-image array from GitHub for its internal
# onnx-vs-tf accuracy check (any model with a static 4D NHWC input). That network
# call makes conversion non-deterministic and fails in offline/sandboxed builds.
# We seed an equivalent local file in the working dir so onnx2tf skips the
# download; the data only feeds a diagnostic, not the produced model.
_SAMPLE_DATA_NAME = "calibration_image_sample_data_20x128x128x3_float32.npy"


_VALID_NAME = r"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"


def _sanitize_onnx_names(input_path, work_dir):
    """Rename ONNX tensors/nodes whose names onnx2tf's saved_model export rejects.

    TF requires op names to match ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$ -- i.e. the
    first char can't be '/'. Newer exporters (e.g. YOLO11/Ultralytics) name nodes
    '/model.10/...', which onnx2tf can't lower. We strip the leading slash (the
    rest of the name is already legal) consistently across the graph and write a
    temp copy. Graph inputs/outputs are left untouched so tflite I/O names hold.
    Returns the sanitized path, or the original if nothing needed renaming.
    """
    import re
    import onnx

    valid = re.compile(rf"^{_VALID_NAME}$")
    model = onnx.load(input_path)
    graph = model.graph
    protected = {t.name for t in list(graph.input) + list(graph.output)}

    names = set()
    for node in graph.node:
        names.update(node.input)
        names.update(node.output)
        if node.name:
            names.add(node.name)
    names.update(init.name for init in graph.initializer)

    rename = {}
    taken = set(names)
    for name in names:
        if not name or name in protected or valid.match(name):
            continue
        base = name.lstrip("/") or "x"
        if not re.match(r"[A-Za-z0-9.]", base[0]):
            base = "n_" + base
        candidate, i = base, 0
        while candidate in taken:
            i += 1
            candidate = f"{base}_{i}"
        rename[name] = candidate
        taken.add(candidate)

    if not rename:
        return input_path

    def rn(x):
        return rename.get(x, x)

    for node in graph.node:
        node.input[:] = [rn(i) for i in node.input]
        node.output[:] = [rn(o) for o in node.output]
        node.name = rn(node.name)
    for init in graph.initializer:
        init.name = rn(init.name)
    for vi in graph.value_info:
        vi.name = rn(vi.name)

    out_path = os.path.join(work_dir, "sanitized.onnx")
    onnx.save(model, out_path)
    return out_path


def _split_yolo_outputs(input_path, work_dir):
    """Split a YOLO detection head's single output into separate box/score tensors.

    Ultralytics YOLOv8/11 emit one output that concatenates decoded boxes (pixel
    units, 0..imgsz) with class scores (0..1). With a single int8 scale spanning
    ~0..imgsz, every class score (<=1) collapses to ~0 and all detections vanish.
    Normalizing the box branch in-graph doesn't survive the neutron-converter's
    optimizer (it relocates the divide). The robust fix is to remove the final
    concat so boxes and scores are *separate* graph outputs -- each then gets its
    own quantization scale and the scores keep their range (same shape as the
    multi-output SSD models the demos already handle).

    No-op unless the graph matches that head: single output produced by a
    Concat(axis=1) of [box-branch, Sigmoid]. Returns the rewritten path or the
    original if it didn't apply.
    """
    import onnx
    from onnx import helper

    model = onnx.load(input_path)
    graph = model.graph
    if len(graph.output) != 1:
        return input_path
    producers = {o: n for n in graph.node for o in n.output}
    concat = producers.get(graph.output[0].name)
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        return input_path
    if next((a.i for a in concat.attribute if a.name == "axis"), None) != 1:
        return input_path

    # One input is the class branch (Sigmoid); the other is the decoded boxes.
    first, second = concat.input
    if (producers.get(first) or _NULL).op_type == "Sigmoid":
        cls_in, box_in = first, second
    elif (producers.get(second) or _NULL).op_type == "Sigmoid":
        cls_in, box_in = second, first
    else:
        return input_path

    graph.node.remove(concat)
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info(box_in, onnx.TensorProto.FLOAT, None),
        helper.make_tensor_value_info(cls_in, onnx.TensorProto.FLOAT, None),
    ])
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass  # shapes are optional; onnx2tf re-infers them

    out_path = os.path.join(work_dir, "yolo_split.onnx")
    onnx.save(model, out_path)
    print("  note: split YOLO head into separate box/score outputs (int8 fix)")
    return out_path


def _yolo_raw_head(input_path, work_dir):
    """Emit the YOLO head *before* box decoding: raw box (DFL logits) + class logits.

    Even split, the NPU's int8 box decode is imprecise: YOLO computes box w/h as a
    difference of large pixel values (x2-x1), and int8 catastrophic cancellation
    collapses it (cx/cy survive, w/h don't). This cuts the graph before the DFL +
    dist2bbox decode entirely, so the NPU runs only convolutions (accurate) and the
    host does DFL + dist2bbox + sigmoid in float. Outputs, in order:
      [0] box_raw  [1, 4*reg_max, anchors] -- cv2 conv logits, pre-DFL
      [1] cls_raw  [1, num_classes, anchors] -- cv3 conv logits, pre-sigmoid

    No-op unless the head matches: single output = Concat(axis=1) of [decoded-box,
    Sigmoid]. Returns the rewritten path or the original.
    """
    import onnx
    from onnx import helper

    model = onnx.load(input_path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    graph = model.graph
    if len(graph.output) != 1:
        return input_path
    producers = {o: n for n in graph.node for o in n.output}
    concat = producers.get(graph.output[0].name)
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        return input_path
    if next((a.i for a in concat.attribute if a.name == "axis"), None) != 1:
        return input_path
    first, second = concat.input
    if (producers.get(first) or _NULL).op_type == "Sigmoid":
        cls_sig, box_dec = first, second
    elif (producers.get(second) or _NULL).op_type == "Sigmoid":
        cls_sig, box_dec = second, first
    else:
        return input_path
    cls_raw = producers[cls_sig].input[0]  # class logits = Sigmoid input

    shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(graph.value_info) + list(graph.output) + list(graph.input)}
    box_shape = shapes.get(box_dec, [])
    anchors = box_shape[-1] if box_shape else None

    # box_raw = the pre-DFL cv2 conv concat: an ancestor Concat shaped [1, 4*reg_max, anchors].
    ancestors, stack = set(), [box_dec]
    while stack:
        x = stack.pop()
        n = producers.get(x)
        if n is None or x in ancestors:
            continue
        ancestors.add(x)
        stack.extend(n.input)
    box_raw = None
    for t in ancestors:
        n = producers.get(t)
        if n is None or n.op_type != "Concat":
            continue
        s = shapes.get(t, [])
        if len(s) == 3 and s[0] == 1 and s[2] == anchors and s[1] > 4 and s[1] % 4 == 0:
            box_raw = t
            break
    if box_raw is None:
        return input_path

    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info(box_raw, onnx.TensorProto.FLOAT, None),
        helper.make_tensor_value_info(cls_raw, onnx.TensorProto.FLOAT, None),
    ])
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    out_path = os.path.join(work_dir, "yolo_rawhead.onnx")
    onnx.save(model, out_path)
    print("  note: emitting raw YOLO head (box DFL + class logits); host-side decode")
    return out_path


class _Null:
    op_type = None


_NULL = _Null()


def _run_onnx2tf(onnx_path, out_dir, input_name, calib_npy, abs_input_path, float_only=False):
    import numpy as np
    import onnx2tf

    prev_cwd = os.getcwd()
    seed = os.path.join(out_dir, _SAMPLE_DATA_NAME)
    if not os.path.isfile(seed):
        np.save(seed, np.random.rand(20, 128, 128, 3).astype(np.float32))
    # download_test_image_data() looks for the seed in os.getcwd().
    os.chdir(out_dir)
    kwargs = dict(
        input_onnx_file_path=abs_input_path,
        output_folder_path=out_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    if not float_only:
        # Full int8 via onnx2tf's own calibration (NXP-backend bypasses this and
        # quantizes the float output with the SDK tools instead).
        kwargs["output_integer_quantized_tflite"] = True
        # We pre-normalize samples, so neutral mean/std here.
        kwargs["custom_input_op_name_np_data_path"] = [
            [input_name, os.path.abspath(calib_npy), 0.0, 1.0]
        ]
    try:
        try:
            onnx2tf.convert(**kwargs)
        except Exception:
            # Some graphs (e.g. YOLO11 attention layers) have op names the plain
            # saved_model export rejects ("OP name does not match the following
            # pattern"); onnx2tf's own remedy is --output_signaturedefs. Retry
            # once with it before giving up.
            onnx2tf.convert(output_signaturedefs=True, **kwargs)
    except Exception as exc:  # onnx2tf raises assorted exceptions
        raise ConversionError(f"onnx2tf conversion failed: {exc}") from exc
    finally:
        os.chdir(prev_cwd)


def _find_int8_tflite(out_dir):
    matches = glob.glob(os.path.join(out_dir, "*_full_integer_quant.tflite"))
    if matches:
        return matches[0]
    matches = glob.glob(os.path.join(out_dir, "*integer_quant*.tflite"))
    return matches[0] if matches else None


def _find_float_tflite(out_dir):
    matches = glob.glob(os.path.join(out_dir, "*_float32.tflite"))
    if matches:
        return matches[0]
    # Fall back to any tflite that isn't a quantized variant.
    others = [
        f for f in glob.glob(os.path.join(out_dir, "*.tflite"))
        if "quant" not in os.path.basename(f)
    ]
    return others[0] if others else None
