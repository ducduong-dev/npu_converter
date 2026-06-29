"""Last-mile Neutron compilation.

Shells out to NXP's ``neutron-converter`` to turn a quantized TFLite model into
a ``*_neutron.tflite`` that the GoPoint demos auto-detect on i.MX 95. This is
the same logic that the standalone ``convert_neutron_model.py`` script used to
contain; it lives here so both the script and the multi-format pipeline share
one implementation.

The converter ships with the NXP eIQ Neutron SDK / meta-imx SDK and is NOT pip
installable. Its version MUST match the target's Neutron runtime (driver,
firmware, delegate from the same SDK); SDK >=3.1 reports a "Microcode version
mismatch!" at load time, older runtimes silently fell back to the CPU.
"""

import os
import sys
import shutil
import hashlib
import subprocess

NEUTRON_SUFFIX = "_neutron.tflite"
DEFAULT_TARGET = "imx95"
DEFAULT_CONVERTER = "neutron-converter"

# Reference default for the "converter not found" hint only. The nxp backend
# reports the real binary version via sdk.version(); this is just a reminder of
# the runtime the converted model must stay aligned with on the board.
REFERENCE_FIRMWARE = "3.1.3"

# Env var that overrides the default converter command. Lets dev/CI point at the
# stub (tools/neutron-converter-stub.py) without passing --converter-cmd.
CONVERTER_ENV = "NEUTRON_CONVERTER"


def default_converter_cmd():
    """The converter command to use when the caller passes nothing."""
    return os.environ.get(CONVERTER_ENV) or DEFAULT_CONVERTER


def neutron_output_name(input_path):
    """Map 'foo.tflite' -> 'foo_neutron.tflite' (the name the demos look for)."""
    base = os.path.basename(input_path)
    if base.endswith(".tflite"):
        return base[: -len(".tflite")] + NEUTRON_SUFFIX
    return base + NEUTRON_SUFFIX


def sha1(path):
    """Return the SHA-1 hex digest of a file (matches downloads.json 'sha')."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_converter(converter_cmd):
    """Return an argv prefix for the converter, or None if it cannot be found.

    Tries, in order: a PATH lookup, an explicit executable path, and finally a
    ``neutron_converter`` Python module entry point (some eIQ distributions
    expose it that way).
    """
    located = shutil.which(converter_cmd)
    if located:
        return [located]
    if os.path.isfile(converter_cmd) and os.access(converter_cmd, os.X_OK):
        return [converter_cmd]
    # A path to a non-executable .py file (e.g. the dev stub) is still runnable.
    if os.path.isfile(converter_cmd) and converter_cmd.endswith(".py"):
        return [sys.executable, converter_cmd]
    if shutil.which("python3"):
        try:
            import importlib.util

            if importlib.util.find_spec("neutron_converter") is not None:
                return [sys.executable, "-m", "neutron_converter"]
        except (ImportError, ValueError):
            pass
    return None


def convert_one(converter, input_path, output_dir, target, extra_args):
    """Compile a single *quantized* TFLite model with the Neutron converter.

    Returns the output path on success, else None. ``converter`` is the argv
    prefix from :func:`resolve_converter`.
    """
    if not os.path.isfile(input_path):
        print(f"ERROR: input model not found: {input_path}")
        return None

    out_name = neutron_output_name(input_path)
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_path))
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, out_name)

    command = converter + [
        "--input",
        input_path,
        "--output",
        output_path,
        "--target",
        target,
    ]
    if extra_args:
        command += extra_args
    print("  running: " + " ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"  ERROR: converter failed (exit {error.returncode}).")
        print(
            "  Check the flags for your converter version with "
            f"'{converter[-1]} --help' and pass overrides after '--'."
        )
        return None
    except OSError as error:
        print(f"  ERROR: could not run converter ({error}).")
        return None

    if not os.path.isfile(output_path):
        print("  ERROR: converter reported success but no output file was written.")
        return None

    print(f"  OK -> {output_path}")
    print(f"     sha1: {sha1(output_path)}")
    return output_path


def converter_not_found_message(converter_cmd):
    """Shared error text for a missing converter."""
    return (
        f"ERROR: Neutron Converter '{converter_cmd}' not found.\n"
        "Install it from the NXP eIQ Toolkit / meta-imx SDK on this host "
        "(it is NOT on the target board), point at it with --converter-cmd, or "
        f"set ${CONVERTER_ENV}.\n"
        "Its version must match the target Neutron runtime "
        f"(reference board: {REFERENCE_FIRMWARE})."
    )


def next_steps_message():
    """Shared 'what to do with the output' guidance."""
    return (
        "Next: copy the *_neutron.tflite file(s) into the board's "
        "/run/media/mmcblk1p1/gopoint-apps/downloads/ directory (the demos pick "
        "them up automatically on i.MX 95), or add them to downloads.json with "
        "the sha1 printed above."
    )
