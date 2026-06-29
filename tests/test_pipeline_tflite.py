"""Pipeline tests for the TFLite pass-through path (the lightest, always run)."""

import os

import pytest

from imx95_npu.cli import main
from imx95_npu.converters import tflite_passthrough, ConversionError


def _int8_tflite_bytes():
    """Build a real, tiny, fully-int8 TFLite model (needs tensorflow)."""
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    inp = tf.keras.Input(shape=(4,))
    out = tf.keras.layers.Dense(2)(inp)
    model = tf.keras.Model(inp, out)

    def rep():
        for _ in range(8):
            yield [np.random.rand(1, 4).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def test_passthrough_int8(tmp_path):
    src = tmp_path / "m.tflite"
    src.write_bytes(_int8_tflite_bytes())
    out = tmp_path / "out.tflite"

    res = tflite_passthrough.to_tflite(str(src), str(out))

    assert res.quantized is True
    assert os.path.isfile(out)


def test_passthrough_float_errors(tmp_path):
    """A float .tflite cannot be re-quantized -> hard error."""
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input(shape=(4,))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2)(inp))
    float_tflite = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    src = tmp_path / "float.tflite"
    src.write_bytes(float_tflite)

    with pytest.raises(ConversionError, match="float TFLite"):
        tflite_passthrough.to_tflite(str(src), str(tmp_path / "out.tflite"))


def test_cli_end_to_end_int8(tmp_path, stub_converter, monkeypatch):
    pytest.importorskip("tensorflow")
    src = tmp_path / "m.tflite"
    src.write_bytes(_int8_tflite_bytes())
    monkeypatch.setenv("NEUTRON_CONVERTER", stub_converter)

    code = main([str(src), "--output-dir", str(tmp_path)])

    assert code == 0
    assert (tmp_path / "m_neutron.tflite").is_file()


def test_cli_float_tflite_fails(tmp_path, stub_converter, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input(shape=(4,))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(2)(inp))
    (tmp_path / "f.tflite").write_bytes(
        tf.lite.TFLiteConverter.from_keras_model(model).convert()
    )
    monkeypatch.setenv("NEUTRON_CONVERTER", stub_converter)

    code = main([str(tmp_path / "f.tflite"), "--output-dir", str(tmp_path)])

    assert code == 2  # some/all models failed
    assert not (tmp_path / "f_neutron.tflite").exists()
