#!/usr/bin/env python3
"""Count AST node types across many Python files."""

import ast
from collections import Counter

counter = Counter()
files_ok = 0
files_err = 0

with open("/tmp/pyfiles.txt") as f:
    files = [line.strip() for line in f if line.strip()]

for path in files:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            counter[type(node).__name__] += 1
        files_ok += 1
    except Exception:
        files_err += 1

print(f"# Parsed {files_ok} files, {files_err} errors")
print(f"# Total nodes: {sum(counter.values())}")
print()
for name, count in counter.most_common():
    print(f"{count:>8}  {name}")
