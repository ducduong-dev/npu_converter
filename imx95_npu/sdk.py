"""Thin wrappers around the NXP eIQ Neutron SDK command-line tools.

The SDK (proprietary, ships as eiq-neutron-sdk-linux-*.zip) provides the official
quantization toolchain for the Neutron NPU:

    float TFLite --(tflite-profiler)--> profile.csv
    float TFLite + profile.csv --(tflite-quantizer)--> int8 TFLite
    int8 TFLite --(neutron-converter)--> *_neutron.tflite

This module locates those binaries and runs them. The neutron-converter step
itself reuses :func:`imx95_npu.neutron.convert_one` so the output naming, sha1
and version-mismatch handling stay in one place (and the dev stub keeps working
for the fallback path).

The SDK is found via ``--sdk-dir`` / ``$NEUTRON_SDK_DIR`` (expects a ``bin/``
subdir), or each tool individually on ``PATH``.
"""

import os
import shutil
import subprocess

from . import neutron

SDK_DIR_ENV = "NEUTRON_SDK_DIR"

PROFILER = "tflite-profiler"
QUANTIZER = "tflite-quantizer"
CONVERTER = "neutron-converter"


class SDKError(RuntimeError):
    """Raised when an SDK tool is missing or fails."""


class NeutronSDK:
    """A resolved set of SDK tool paths."""

    def __init__(self, tools, sdk_dir=None):
        # tools: name -> absolute path (or None if not found)
        self._tools = tools
        self.sdk_dir = sdk_dir

    # -- resolution ---------------------------------------------------------

    @classmethod
    def resolve(cls, sdk_dir=None):
        """Locate the SDK tools. ``sdk_dir`` overrides ``$NEUTRON_SDK_DIR``.

        Returns a :class:`NeutronSDK` (possibly with some tools missing — check
        :meth:`has_full_toolchain`). The converter also honours
        ``$NEUTRON_CONVERTER`` / the dev stub via ``neutron.resolve_converter``.
        """
        sdk_dir = sdk_dir or os.environ.get(SDK_DIR_ENV)
        tools = {}
        for name in (PROFILER, QUANTIZER, CONVERTER):
            tools[name] = cls._find_tool(name, sdk_dir)
        return cls(tools, sdk_dir=sdk_dir)

    @staticmethod
    def _find_tool(name, sdk_dir):
        if sdk_dir:
            candidate = os.path.join(sdk_dir, "bin", name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        located = shutil.which(name)
        return located

    def tool(self, name):
        path = self._tools.get(name)
        if not path:
            raise SDKError(
                f"SDK tool '{name}' not found. Set ${SDK_DIR_ENV} to the "
                "extracted eiq-neutron-sdk directory (run scripts/setup_sdk.sh), "
                "or put the tool on PATH."
            )
        return path

    def has(self, name):
        return bool(self._tools.get(name))

    def has_full_toolchain(self):
        """True if profiler + quantizer + converter are all available (the
        prerequisite for the NXP-native quantization backend)."""
        return all(self.has(n) for n in (PROFILER, QUANTIZER, CONVERTER))

    # -- pipeline steps -----------------------------------------------------

    def profile(
        self,
        float_tflite,
        out_csv,
        *,
        dataset_dir=None,
        histogram_bins=256,
        random_samples=100,
        random_min=-1.0,
        random_max=1.0,
    ):
        """Run tflite-profiler to produce a calibration profile CSV.

        Provide ``dataset_dir`` (a directory of raw float32 sample files) for
        real calibration; omit it to fall back to random data (poor accuracy).
        """
        argv = [self.tool(PROFILER), "--input", float_tflite, "--output", out_csv,
                "--histogram-num-bins", str(histogram_bins)]
        if dataset_dir:
            argv += ["--dataset", dataset_dir, "--dataset-source", "dir"]
        else:
            print(
                "  WARNING: no representative dataset; profiling with RANDOM data. "
                "Quantization accuracy will be poor — pass --rep-data."
            )
            argv += ["--use-random-data",
                     "--random-data-samples", str(random_samples),
                     "--random-data-min", str(random_min),
                     "--random-data-max", str(random_max)]
        _run(argv, "tflite-profiler")
        if not os.path.isfile(out_csv):
            raise SDKError("tflite-profiler reported success but wrote no profile.")
        return out_csv

    def quantize(
        self,
        float_tflite,
        profile_csv,
        out_tflite,
        *,
        calibration_method="MinMax",
        percentile_val=None,
        extra=None,
    ):
        """Run tflite-quantizer (profiling-guided int8 PTQ)."""
        argv = [self.tool(QUANTIZER), "--input", float_tflite,
                "--profile", profile_csv, "--output", out_tflite,
                "--quantization-calibration-method", calibration_method]
        if calibration_method == "Percentile" and percentile_val is not None:
            argv += ["--quantization-percentile-val", str(percentile_val)]
        if extra:
            argv += extra
        _run(argv, "tflite-quantizer")
        if not os.path.isfile(out_tflite):
            raise SDKError("tflite-quantizer reported success but wrote no model.")
        return out_tflite

    def convert(self, quant_tflite, output_dir, target, *, extra=None,
                run_after_generate=False, dump_statistics=False):
        """Run neutron-converter via the shared neutron.convert_one helper.

        Returns the produced ``*_neutron.tflite`` path, or None on failure.
        """
        converter = [self.tool(CONVERTER)]
        flags = list(extra or [])
        if run_after_generate:
            flags.append("--run-after-generate")
        if dump_statistics:
            flags.append("--dump-statistics")
        return neutron.convert_one(converter, quant_tflite, output_dir, target, flags)

    # -- introspection ------------------------------------------------------

    def version(self):
        """neutron-converter version string, or None if unavailable."""
        if not self.has(CONVERTER):
            return None
        try:
            out = subprocess.run(
                [self.tool(CONVERTER), "--version"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip().splitlines()[0] if out.stdout else None
        except (subprocess.CalledProcessError, OSError, IndexError):
            return None


def _run(argv, label):
    """Run an SDK tool, raising SDKError with a captured stderr tail on failure."""
    print("  running: " + " ".join(argv))
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        tail = (error.stderr or error.stdout or "").strip().splitlines()[-8:]
        raise SDKError(f"{label} failed (exit {error.returncode}):\n  " +
                       "\n  ".join(tail)) from error
    except OSError as error:
        raise SDKError(f"could not run {label}: {error}") from error
    return proc
