"""NXP eIQ Neutron SDK backend tests.

These run the real SDK binaries (tflite-profiler / tflite-quantizer /
neutron-converter) and are skipped automatically when the SDK isn't provisioned
under vendor/eiq-neutron-sdk (see the sdk_dir fixture).
"""

import os

import numpy as np
import pytest

from imx95_npu.sdk import NeutronSDK, PROFILER, QUANTIZER, CONVERTER
from imx95_npu.cli import main
from imx95_npu import quantize


def _float_tflite(path):
    """Write a tiny float TFLite model (input (8,8,3) -> conv -> dense)."""
    import imx95_npu  # ensures TF_USE_LEGACY_KERAS
    import tensorflow as tf

    inp = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2)(x))
    with open(path, "wb") as handle:
        handle.write(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    return path


def test_resolve_and_version(sdk_dir):
    sdk = NeutronSDK.resolve(sdk_dir)
    assert sdk.has_full_toolchain()
    for tool in (PROFILER, QUANTIZER, CONVERTER):
        assert sdk.has(tool)
    assert "neutron-converter" in (sdk.version() or "")


def test_profiler_quantizer_converter_chain(sdk_dir, tmp_path):
    pytest.importorskip("tensorflow")
    sdk = NeutronSDK.resolve(sdk_dir)
    float_tflite = _float_tflite(str(tmp_path / "f.tflite"))

    csv = sdk.profile(float_tflite, str(tmp_path / "p.csv"))  # random calibration
    assert os.path.isfile(csv)

    int8 = sdk.quantize(float_tflite, csv, str(tmp_path / "q.tflite"))
    assert quantize.is_int8_quantized(int8)

    out = sdk.convert(int8, str(tmp_path), "imx95", run_after_generate=True)
    assert out and out.endswith("q_neutron.tflite")
    assert open(out, "rb").read()[4:8] == b"TFL3"


def test_quantize_with_real_dataset(sdk_dir, tmp_path):
    """End-to-end via the profiler dataset path (not random data)."""
    pytest.importorskip("tensorflow")
    sdk = NeutronSDK.resolve(sdk_dir)
    float_tflite = _float_tflite(str(tmp_path / "f.tflite"))

    # raw float32 NHWC samples for tflite-profiler --dataset
    ds = tmp_path / "calib"
    ds.mkdir()
    for i in range(8):
        np.random.rand(8, 8, 3).astype(np.float32).tofile(ds / f"s{i:03d}.bin")

    csv = sdk.profile(float_tflite, str(tmp_path / "p.csv"), dataset_dir=str(ds))
    int8 = sdk.quantize(float_tflite, csv, str(tmp_path / "q.tflite"))
    assert quantize.is_int8_quantized(int8)


def test_cli_nxp_backend_keras_end_to_end(sdk_dir, tmp_path, tiny_keras_model, calib_npy):
    """Full CLI run through the nxp backend: Keras -> float -> SDK quant -> neutron."""
    pytest.importorskip("tensorflow")
    src = tmp_path / "m.keras"
    tiny_keras_model.save(src)

    code = main([
        str(src),
        "--quant-backend", "nxp",
        "--sdk-dir", sdk_dir,
        "--rep-data", calib_npy((8, 8, 3)),
        "--output-dir", str(tmp_path),
    ])

    assert code == 0
    out = tmp_path / "m_neutron.tflite"
    assert out.is_file()
    assert out.read_bytes()[4:8] == b"TFL3"


def test_cli_nxp_backend_onnx_end_to_end(sdk_dir, tmp_path, tiny_onnx_path, calib_npy):
    """ONNX -> float (onnx2tf) -> SDK quant -> neutron, via the nxp backend.

    Exercises the NCHW(rep-data) -> NHWC(profiler) calibration transpose."""
    pytest.importorskip("onnx2tf")
    code = main([
        tiny_onnx_path,
        "--quant-backend", "nxp",
        "--sdk-dir", sdk_dir,
        "--rep-data", calib_npy((3, 8, 8)),  # NCHW, the onnx-native layout
        "--input-shape", "3,8,8",
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "tiny_neutron.tflite").is_file()


def test_cli_nxp_prequantized_tflite(sdk_dir, tmp_path, stub_converter, monkeypatch):
    """An already-int8 .tflite skips profiler/quantizer and goes straight to neutron."""
    tf = pytest.importorskip("tensorflow")
    import imx95_npu  # noqa: F401  (legacy keras)

    inp = tf.keras.Input(shape=(4,))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2)(inp))

    def rep():
        for _ in range(8):
            yield [np.random.rand(1, 4).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    (tmp_path / "q.tflite").write_bytes(conv.convert())

    code = main([
        str(tmp_path / "q.tflite"),
        "--quant-backend", "nxp",
        "--sdk-dir", sdk_dir,
        "--output-dir", str(tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "q_neutron.tflite").is_file()


def test_nxp_backend_requires_sdk(tmp_path, monkeypatch):
    """--quant-backend nxp with no SDK fails cleanly (exit 1)."""
    monkeypatch.delenv("NEUTRON_SDK_DIR", raising=False)
    (tmp_path / "m.tflite").write_bytes(b"\x00\x00\x00\x00TFL3body")
    code = main([
        str(tmp_path / "m.tflite"),
        "--quant-backend", "nxp",
        "--sdk-dir", str(tmp_path / "nonexistent-sdk"),
    ])
    assert code == 1
