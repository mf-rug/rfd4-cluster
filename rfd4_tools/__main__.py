"""Cluster-side entry point: `python3 -m rfd4_tools log <job_dir>`.

The package is copied into each job directory and invoked with
PYTHONPATH="$_job_dir", so the cluster needs no install of this tool and the
job can never run a different version than the one that generated it.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python3 -m rfd4_tools log [JOB_DIR]", file=sys.stderr)
        return 2
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "log":
        from rfd4_tools.logparse import main as log_main

        return log_main(rest)
    print(f"unknown command {command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
