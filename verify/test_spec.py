#!/usr/bin/env python3
"""Unit checks for the spec parser. Run: python3 verify/test_spec.py

These are the cases that cost something when they are wrong: a mis-parsed
contig under-sizes the GPU tier, and a mis-parsed selection reaches the
cluster as a runtime failure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rfd4_tools.spec import SpecError, check_selection, contig_max_tokens, parse_contig

FAILED = []


def check(label, got, want):
    if got != want:
        FAILED.append(f"{label}: got {got!r}, want {want!r}")


def raises(label, fn):
    try:
        fn()
    except SpecError:
        return
    FAILED.append(f"{label}: expected SpecError, none raised")


# Sizing. The regression that motivated this file: a greedy chain id made
# "A17-131" parse as one residue, routing a 235-token job to a MIG slice.
check("binder+target", contig_max_tokens("50-120, /0, A17-131"), 235)
check("target alone", contig_max_tokens("A17-131"), 115)
check("single residue", contig_max_tokens("A52"), 1)
check("two-letter chain", contig_max_tokens("AB1-10"), 10)
check("named expandable", contig_max_tokens("E1:2-20, A1-5"), 25)
check("fixed expandable", contig_max_tokens("A1-5, 10"), 15)
check("break is free", contig_max_tokens("A1-10, /0, B1-10"), 20)

# Expandable ranges size at their UPPER bound: a tier is an OOM cliff and
# must hold the largest design the spec can produce.
check("worst case not average", contig_max_tokens("10-500, A1-1"), 501)

# Structural rejections
raises("all-expandable rejected", lambda: parse_contig("50-120"))
raises("empty contig", lambda: parse_contig("  "))
raises("backwards range", lambda: parse_contig("A131-17"))
raises("garbage token", lambda: parse_contig("A1-10, !!"))

# Selections. The documented trap: trailing fields default to '*', so a
# 3-field path puts an atom name where res_id belongs.
check("atom name in res_id slot", len(check_selection("sel('B/*/N')")), 1)
check("correct 4-field form", check_selection("sel('B/*/*/N')"), [])
check("numeric res_id fine", check_selection("sel('A/*/45')"), [])
check("bracket list fine", check_selection("sel('A/*/[45, 46]')"), [])
check("range fine", check_selection("sel('A/*/10-20')"), [])
check("too many fields", len(check_selection("sel('A/B/C/D/E')")), 1)
check("query syntax untouched", check_selection("is_ligand and not is_protein_backbone"), [])

if FAILED:
    print("FAILED:")
    for f in FAILED:
        print("  " + f)
    sys.exit(1)
print("all spec checks passed")
