"""Last-mile neutron core tests, driven by the stub converter (no NXP binary)."""

import os
import sys

from imx95_npu import neutron

# Mirror of the marker the dev stub appends (tools/neutron-converter-stub.py).
STUB_MARKER = b"\nNEUTRON_STUB"


def test_neutron_output_name():
    assert neutron.neutron_output_name("foo.tflite") == "foo_neutron.tflite"
    assert neutron.neutron_output_name("/a/b/m.tflite") == "m_neutron.tflite"
    # Non-.tflite input still gets the suffix appended.
    assert neutron.neutron_output_name("model").endswith(neutron.NEUTRON_SUFFIX)


def test_resolve_converter_py_stub(stub_converter):
    # The stub is executable, so it resolves directly as the command.
    argv = neutron.resolve_converter(stub_converter)
    assert argv == [stub_converter]


def test_resolve_converter_nonexec_py_uses_interpreter(tmp_path):
    # A non-executable .py path is run through the current interpreter.
    script = tmp_path / "conv.py"
    script.write_text("print('hi')\n")  # not chmod +x
    argv = neutron.resolve_converter(str(script))
    assert argv == [sys.executable, str(script)]


def test_resolve_converter_missing_returns_none():
    assert neutron.resolve_converter("definitely-not-a-real-converter-xyz") is None


def test_default_converter_cmd_env(monkeypatch):
    monkeypatch.setenv(neutron.CONVERTER_ENV, "/custom/conv")
    assert neutron.default_converter_cmd() == "/custom/conv"
    monkeypatch.delenv(neutron.CONVERTER_ENV)
    assert neutron.default_converter_cmd() == neutron.DEFAULT_CONVERTER


def test_convert_one_with_stub(tmp_path, stub_converter):
    src = tmp_path / "m.tflite"
    src.write_bytes(b"\x00\x00\x00\x00TFL3quantized-body")
    argv = neutron.resolve_converter(stub_converter)

    out = neutron.convert_one(argv, str(src), str(tmp_path), neutron.DEFAULT_TARGET, [])

    assert out is not None
    assert os.path.basename(out) == "m_neutron.tflite"
    assert open(out, "rb").read().endswith(STUB_MARKER)


def test_convert_one_missing_input(tmp_path, stub_converter):
    argv = neutron.resolve_converter(stub_converter)
    out = neutron.convert_one(argv, str(tmp_path / "nope.tflite"), str(tmp_path),
                              neutron.DEFAULT_TARGET, [])
    assert out is None


def test_sha1_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "f.bin"
    p.write_bytes(b"hello neutron")
    assert neutron.sha1(str(p)) == hashlib.sha1(b"hello neutron").hexdigest()
