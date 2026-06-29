"""Shared fixtures: tiny on-the-fly models + the dev stub converter.

Heavy-framework fixtures (tf/onnx/torch) ``pytest.importorskip`` their
dependency so the suite degrades gracefully on a host that only has some of the
conversion stack installed. The stub converter makes the neutron last-mile run
everywhere.
"""

import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = os.path.join(REPO_ROOT, "tools", "neutron-converter-stub.py")
VENDOR_SDK = os.path.join(REPO_ROOT, "vendor", "eiq-neutron-sdk")


@pytest.fixture(autouse=True)
def _no_ambient_sdk(monkeypatch):
    """Keep the default (auto) backend deterministic: hide any ambient
    $NEUTRON_SDK_DIR so non-SDK tests use the tflite/stub backend. SDK tests
    opt in by passing --sdk-dir explicitly."""
    monkeypatch.delenv("NEUTRON_SDK_DIR", raising=False)


@pytest.fixture(scope="session")
def stub_converter():
    """Path to the dev stub neutron converter."""
    assert os.path.isfile(STUB), STUB
    return STUB


def _sdk_available():
    return all(
        os.path.isfile(os.path.join(VENDOR_SDK, "bin", t))
        for t in ("tflite-profiler", "tflite-quantizer", "neutron-converter")
    )


@pytest.fixture(scope="session")
def sdk_dir():
    """Path to the extracted eIQ Neutron SDK, or skip if not provisioned."""
    if not _sdk_available():
        pytest.skip(
            "eIQ Neutron SDK not found under vendor/eiq-neutron-sdk "
            "(run scripts/setup_sdk.sh)."
        )
    return VENDOR_SDK


@pytest.fixture
def calib_npy(tmp_path):
    """Factory: write an (N, *shape) float32 calibration array, return its path."""

    def _make(shape, n=8, name="calib.npy"):
        arr = np.random.rand(n, *shape).astype(np.float32)
        path = tmp_path / name
        np.save(path, arr)
        return str(path)

    return _make


@pytest.fixture
def tiny_keras_model():
    """A minimal quantizable Keras model: input (8,8,3) -> conv -> dense(2)."""
    tf = pytest.importorskip("tensorflow")
    inp = tf.keras.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(2)(x)
    return tf.keras.Model(inp, out)


def _build_tiny_net():
    """A minimal torch model. Defined via the module-level TinyNet class so it
    can be pickled by torch.save (a locally-scoped class cannot)."""
    import torch

    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    # Re-bind to module scope so pickling by qualified name works.
    TinyNet.__module__ = __name__
    TinyNet.__qualname__ = "TinyNet"
    globals().setdefault("TinyNet", TinyNet)
    return globals()["TinyNet"]().eval()


@pytest.fixture
def tiny_onnx_path(tmp_path):
    """A minimal ONNX model with input (1,3,8,8) -> conv -> output."""
    pytest.importorskip("onnx")
    torch = pytest.importorskip("torch")

    path = str(tmp_path / "tiny.onnx")
    torch.onnx.export(
        _build_tiny_net(),
        torch.zeros(1, 3, 8, 8),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    return path


@pytest.fixture
def tiny_torch_path(tmp_path):
    """A whole-module torch model saved with torch.save(model, path)."""
    torch = pytest.importorskip("torch")
    path = str(tmp_path / "tiny_model.pt")
    torch.save(_build_tiny_net(), path)
    return path
