"""Format detection tests (no heavy deps)."""

import os

import pytest

from imx95_npu import detect


def _write(path, data):
    with open(path, "wb") as handle:
        handle.write(data)
    return str(path)


def test_detect_tflite_by_magic(tmp_path):
    # TFL3 identifier at bytes 4..8, regardless of extension.
    p = _write(tmp_path / "model.bin", b"\x00\x00\x00\x00TFL3rest-of-buffer")
    assert detect.detect_format(p) == detect.TFLITE


def test_detect_torch_zip(tmp_path):
    p = _write(tmp_path / "m.pt", b"PK\x03\x04" + b"\x00" * 32)
    assert detect.detect_format(p) == detect.TORCH


def test_detect_onnx_by_extension(tmp_path):
    p = _write(tmp_path / "m.onnx", b"\x08\x07rest")
    assert detect.detect_format(p) == detect.ONNX


def test_detect_onnx_by_protobuf_sniff(tmp_path):
    # No known extension, but protobuf field-1 tag -> onnx heuristic.
    p = _write(tmp_path / "model.data", b"\x08\x07more-protobuf")
    assert detect.detect_format(p) == detect.ONNX


def test_detect_keras_extension(tmp_path):
    p = _write(tmp_path / "m.h5", b"\x89HDF\r\n\x1a\n")
    assert detect.detect_format(p) == detect.TENSORFLOW


def test_detect_saved_model_dir(tmp_path):
    d = tmp_path / "sm"
    d.mkdir()
    _write(d / "saved_model.pb", b"\x08\x01")
    assert detect.detect_format(str(d)) == detect.TENSORFLOW


def test_detect_dir_without_saved_model_errors(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(detect.DetectionError):
        detect.detect_format(str(d))


def test_detect_missing_file(tmp_path):
    with pytest.raises(detect.DetectionError):
        detect.detect_format(str(tmp_path / "nope.tflite"))


def test_detect_unknown_errors(tmp_path):
    p = _write(tmp_path / "mystery.xyz", b"\xff\xfe\xfd\xfc random bytes")
    with pytest.raises(detect.DetectionError):
        detect.detect_format(p)
