# verify

`test_spec.py` covers the contig and selection parsers -- the two places
where a parsing bug silently changes a resource decision rather than
producing an obvious error.

    python3 verify/test_spec.py

A GPU-tier sweep belongs here too, once someone runs one. The shape to copy
is rfd3-cluster's `verify/bisect.sh`: ascending sizes, one process per probe
(peak VRAM is per-forward-pass, so probes must not share a process), early
stop on OOM.
