"""ONNX and Torch paths -> int8 TFLite -> stub neutron.

These import onnx2tf + torch; they ``importorskip`` so the suite still passes on
a host that only has the TF path installed.
"""

import pytest

from imx95_npu.cli import main
from imx95_npu.converters import onnx_to_tflite, torch_to_tflite, ConversionError
from imx95_npu.repdata import RepDataSpec
from imx95_npu import quantize


def test_onnx_input_info(tiny_onnx_path):
    pytest.importorskip("onnx")
    name, shape = onnx_to_tflite.onnx_input_info(tiny_onnx_path)
    assert name == "input"
    assert shape == (3, 8, 8)


def test_onnx_int8_conversion(tmp_path, tiny_onnx_path, calib_npy):
    pytest.importorskip("onnx2tf")
    rep = RepDataSpec(spec=calib_npy((3, 8, 8)))
    out = tmp_path / "out.tflite"

    res = onnx_to_tflite.to_tflite(tiny_onnx_path, str(out), rep_data=rep)

    assert quantize.is_int8_quantized(str(out))
    assert res.quantized is True


def test_torch_requires_input_shape(tmp_path, tiny_torch_path, calib_npy):
    pytest.importorskip("torch")
    rep = RepDataSpec(spec=calib_npy((3, 8, 8)))
    with pytest.raises(ConversionError, match="input-shape"):
        torch_to_tflite.to_tflite(
            tiny_torch_path, str(tmp_path / "o.tflite"), rep_data=rep, input_shape=None
        )


def test_torch_state_dict_rejected(tmp_path, calib_npy):
    torch = pytest.importorskip("torch")
    sd_path = tmp_path / "sd.pt"
    torch.save({"weight": torch.zeros(2, 2)}, sd_path)
    rep = RepDataSpec(spec=calib_npy((3, 8, 8)))

    with pytest.raises(ConversionError, match="state_dict"):
        torch_to_tflite.to_tflite(
            str(sd_path), str(tmp_path / "o.tflite"), rep_data=rep, input_shape=(3, 8, 8)
        )


def test_torch_cli_end_to_end(tmp_path, tiny_torch_path, calib_npy, stub_converter, monkeypatch):
    pytest.importorskip("onnx2tf")
    pytest.importorskip("torch")
    monkeypatch.setenv("NEUTRON_CONVERTER", stub_converter)

    code = main([
        tiny_torch_path,
        "--rep-data", calib_npy((3, 8, 8)),
        "--input-shape", "3,8,8",
        "--output-dir", str(tmp_path),
    ])

    assert code == 0
    assert (tmp_path / "tiny_model_neutron.tflite").is_file()
