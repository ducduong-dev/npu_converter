"""TensorFlow path: Keras + SavedModel -> int8 TFLite -> stub neutron."""

import os

import pytest

from imx95_npu.cli import main
from imx95_npu.converters import tf_to_tflite, ConversionError
from imx95_npu.repdata import RepDataSpec
from imx95_npu import quantize


def test_keras_int8_conversion(tmp_path, tiny_keras_model, calib_npy):
    pytest.importorskip("tensorflow")
    src = tmp_path / "m.keras"
    tiny_keras_model.save(src)
    rep = RepDataSpec(spec=calib_npy((8, 8, 3)))
    out = tmp_path / "out.tflite"

    res = tf_to_tflite.to_tflite(str(src), str(out), rep_data=rep)

    assert res.quantized is True
    assert quantize.is_int8_quantized(str(out))


def test_keras_without_repdata_errors(tmp_path, tiny_keras_model):
    pytest.importorskip("tensorflow")
    src = tmp_path / "m.keras"
    tiny_keras_model.save(src)

    with pytest.raises(ConversionError, match="rep-data"):
        tf_to_tflite.to_tflite(str(src), str(tmp_path / "o.tflite"), rep_data=None)


def test_saved_model_int8_conversion(tmp_path, tiny_keras_model, calib_npy):
    tf = pytest.importorskip("tensorflow")
    sm_dir = tmp_path / "sm"
    tiny_keras_model.export(str(sm_dir))  # writes a SavedModel
    rep = RepDataSpec(spec=calib_npy((8, 8, 3)))
    out = tmp_path / "out.tflite"

    res = tf_to_tflite.to_tflite(str(sm_dir), str(out), rep_data=rep)

    assert res.quantized is True
    del tf


def test_keras_cli_end_to_end(tmp_path, tiny_keras_model, calib_npy, stub_converter, monkeypatch):
    pytest.importorskip("tensorflow")
    src = tmp_path / "m.keras"
    tiny_keras_model.save(src)
    monkeypatch.setenv("NEUTRON_CONVERTER", stub_converter)

    code = main([
        str(src),
        "--rep-data", calib_npy((8, 8, 3)),
        "--output-dir", str(tmp_path),
    ])

    assert code == 0
    assert (tmp_path / "m_neutron.tflite").is_file()
