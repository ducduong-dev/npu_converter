"""imx95_npu -- convert PyTorch / TensorFlow / ONNX / TFLite models into a
``*_neutron.tflite`` that runs on the i.MX 95 eIQ Neutron NPU.

The public entry point is the ``imx95-convert`` CLI (``imx95_npu.cli:main``).
``imx95_npu.neutron`` holds the last-mile logic shared with the legacy
``convert_neutron_model.py`` script.
"""

__version__ = "0.1.0"

import os as _os

# TensorFlow >= 2.16 defaults to Keras 3, whose models crash the TFLite MLIR
# converter ("LLVM ERROR: Failed to infer result type(s)"). Force the Keras 2
# API (provided by the tf-keras package) BEFORE any tensorflow import. We import
# tensorflow lazily inside converters, so setting this at package import time
# guarantees it takes effect. Override by exporting TF_USE_LEGACY_KERAS=0.
_os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# The output-name suffix is a contract with the GoPoint demos -- they look for
# a file ending in this on i.MX 95. Re-exported here so callers don't reach
# into submodules for it.
from .neutron import NEUTRON_SUFFIX, neutron_output_name  # noqa: E402,F401

__all__ = ["NEUTRON_SUFFIX", "neutron_output_name", "__version__"]
