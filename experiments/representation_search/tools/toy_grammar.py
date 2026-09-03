#!/usr/bin/env python3
"""Enumerate all 24 precedence grammars over a token sequence and collapse them
into exact equivalence classes.

    python3 tools/toy_grammar.py "2*t+1-t/2"
    python3 tools/toy_grammar.py "2*x*x+1-x/2"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch.grammar import report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    print(report(sys.argv[1]))
