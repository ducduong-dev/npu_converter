#!/usr/bin/env python3
"""Dev/CI stand-in for NXP's ``neutron-converter``.

The real converter ships with the eIQ Toolkit and isn't on most dev hosts. This
stub accepts the same ``--input/--output/--target`` flags so the full pipeline,
sha1 reporting, and tests run end-to-end without the proprietary binary. It does
NOT produce a board-loadable model -- it copies the input and appends a marker,
purely so downstream steps have a file to operate on.

Point at it with:  --converter-cmd tools/neutron-converter-stub.py
              or:  NEUTRON_CONVERTER=tools/neutron-converter-stub.py
"""

import sys
import argparse

STUB_MARKER = b"\nNEUTRON_STUB"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neutron converter (dev stub).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="imx95")
    # Accept (and ignore) the real converter's optional flags so callers can
    # pass them through '--' without the stub choking.
    parser.add_argument("--use-sequencer", action="store_true")
    parser.add_argument("--fetch-constants-to-sram", action="store_true")
    parser.add_argument("--dump-header-file-output", default=None)
    parser.add_argument("--dump-header-file-input", default=None)
    args, _unknown = parser.parse_known_args(argv)

    try:
        with open(args.input, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        print(f"stub: cannot read input: {exc}", file=sys.stderr)
        return 1

    if not data:
        print("stub: empty input model", file=sys.stderr)
        return 1

    with open(args.output, "wb") as handle:
        handle.write(data)
        handle.write(STUB_MARKER)
    print(f"stub: wrote {args.output} (target={args.target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
