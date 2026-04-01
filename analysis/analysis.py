#!/usr/bin/env python3
"""Analyze coverage of Python AST nodes by fluent-codegen.

Updated analysis based on 47,341 Python files (84M total nodes).
"""

# AST node counts from 47,341 files (84M total nodes)
# Only including nodes that represent user-visible syntax
# (excluding Load, Store, Del which are internal context nodes,
#  and structural nodes like arg, keyword, arguments, alias, etc.)

nodes = {
    # Expressions
    "Name": 16424441,
    "Constant": 12008911,
    "Attribute": 5422959,
    "Call": 4533640,
    "Tuple": 1487982,
    "Subscript": 934211,
    "List": 875399,
    "BinOp": 858747,
    "Compare": 741530,
    "UnaryOp": 461327,
    "FormattedValue": 196242,
    "BoolOp": 164820,
    "Slice": 149122,
    "Dict": 139477,
    "JoinedStr": 133051,
    "AugAssign": 94096,
    "IfExp": 45986,
    "ListComp": 43289,
    "Starred": 39534,
    "Yield": 32950,
    "GeneratorExp": 26347,
    "Lambda": 22185,
    "Set": 21174,
    "DictComp": 6599,
    "YieldFrom": 5504,
    "NamedExpr": 3575,
    "SetComp": 2400,
    "Await": 2338,
    # Statements
    "Assign": 2508267,
    "Expr": 1646915,  # expression-as-statement
    "If": 824659,
    "FunctionDef": 733484,
    "Return": 550664,
    "ImportFrom": 232794,
    "For": 175484,
    "Assert": 132863,
    "Raise": 131948,
    "ClassDef": 115384,
    "Import": 105591,
    "AnnAssign": 94027,
    "Try": 70201,
    "With": 60190,
    "Pass": 35714,
    "Continue": 24008,
    "While": 17209,
    "Break": 15590,
    "Delete": 12272,
    "Global": 2404,
    "AsyncFunctionDef": 2317,
    "Match": 675,
    "Nonlocal": 445,
    "TypeAlias": 395,
    "AsyncWith": 168,
    "AsyncFor": 127,
    "TryStar": 47,
}

# What fluent-codegen covers (current as of post-0.5.0 development)
covered = {
    # Expressions
    "Name",  # Name class
    "Constant",  # String, Number, Bool, Bytes, NoneExpr
    "Attribute",  # Attr class
    "Call",  # Call class
    "Tuple",  # Tuple class
    "List",  # List class
    "Subscript",  # Subscript class
    "BinOp",  # BinaryOperator (Add, Sub, Mul, etc.)
    "Compare",  # CompareOp subclasses (Equals, Lt, etc.)
    "UnaryOp",  # Not, UAdd, USub, Invert
    "BoolOp",  # And, Or
    "FormattedValue",  # via FStringJoin
    "JoinedStr",  # FStringJoin
    "Dict",  # Dict class
    "Starred",  # Starred class
    "Set",  # Set class
    "Slice",  # Slice class (added in 0.5.0)
    "ListComp",  # ListComp class
    "SetComp",  # SetComp class
    "DictComp",  # DictComp class
    "GeneratorExp",  # GeneratorExpr class
    "Lambda",  # Lambda class
    # Statements
    "Assign",  # Assignment class
    "Expr",  # Expressions used as statements in Block
    "If",  # If class
    "FunctionDef",  # Function class
    "Return",  # Return class
    "ImportFrom",  # ImportFrom class
    "For",  # For class
    "ClassDef",  # Class class
    "Import",  # Import class
    "Assert",  # Assert class
    "AnnAssign",  # Annotation / Assignment with type_hint
    "Raise",  # Raise class
    "Try",  # Try class
    "With",  # With class
    "Pass",  # used as empty block filler
    "Break",  # Break class
    "Continue",  # Continue class
}

not_covered = {k: v for k, v in nodes.items() if k not in covered}
covered_items = {k: v for k, v in nodes.items() if k in covered}

total = sum(nodes.values())
covered_total = sum(covered_items.values())
not_covered_total = sum(not_covered.values())

print(f"Total syntax nodes analyzed: {total:,}")
print(f"Covered by fluent-codegen:  {covered_total:,} ({covered_total / total * 100:.1f}%)")
print(f"Not covered:                {not_covered_total:,} ({not_covered_total / total * 100:.1f}%)")
print()
print("=" * 70)
print("MISSING AST NODES — ranked by frequency in real Python code")
print("=" * 70)
print()
print(f"{'Rank':>4}  {'AST Node':<22} {'Count':>10}  {'% of total':>10}  Notes")
print(f"{'----':>4}  {'--------':<22} {'-----':>10}  {'----------':>10}  -----")

notes = {
    "AugAssign": "x += 1, x -= 1, etc.",
    "IfExp": "ternary: x if cond else y",
    "Yield": "generator yield",
    "While": "while loops",
    "Delete": "del statement",
    "YieldFrom": "yield from expr",
    "NamedExpr": "walrus operator :=",
    "Await": "await expression",
    "Global": "global statement",
    "AsyncFunctionDef": "async def",
    "Match": "match/case (3.10+)",
    "Nonlocal": "nonlocal statement",
    "TypeAlias": "type X = ... (3.12+)",
    "AsyncWith": "async with",
    "AsyncFor": "async for",
    "TryStar": "try/except* (ExceptionGroup, 3.11+)",
}

ranked = sorted(not_covered.items(), key=lambda x: -x[1])
for i, (name, count) in enumerate(ranked, 1):
    pct = count / total * 100
    note = notes.get(name, "")
    print(f"{i:>4}  {name:<22} {count:>10,}  {pct:>9.2f}%  {note}")

print()
print("=" * 70)
print("COVERAGE CHANGE SINCE LAST ANALYSIS")
print("=" * 70)
print()
newly_covered = {
    "For": "For class — for loops",
    "Raise": "Raise class — raise statements",
    "Slice": "Slice class — slice notation",
    "Break": "Break class — loop break",
    "Continue": "Continue class — loop continue",
    "ListComp": "ListComp class — list comprehensions",
    "SetComp": "SetComp class — set comprehensions",
    "DictComp": "DictComp class — dict comprehensions",
    "GeneratorExp": "GeneratorExpr class — generator expressions",
    "Lambda": "Lambda class — lambda expressions",
}
for name, desc in newly_covered.items():
    count = nodes.get(name, 0)
    print(f"  ✅  {name:<18} {count:>10,} occurrences — {desc}")

print()
print("=" * 70)
print("TOP SUGGESTIONS FOR ADDING TO FLUENT-CODEGEN")
print("=" * 70)
print()
print("High priority (very common syntax, broadly useful):")
print("  1. AugAssign  (94,096) — augmented assignment (+=, -=, *=, etc.)")
print("  2. IfExp      (45,986) — inline ternary expressions")
print("  3. Yield      (32,950) — generator yield / yield from")
print("  4. While      (17,209) — while loops")
print()
print("Medium priority (useful but more niche):")
print("  5. Delete     (12,272) — del statement")
print("  6. YieldFrom   (5,504) — yield from")
print("  7. NamedExpr   (3,575) — walrus operator :=")
print("  8. Await       (2,338) — await expression (+ async ecosystem)")
print()
print("Lower priority (rare or very specialized):")
print("  9. Global      (2,404) — global statement")
print(" 10. AsyncFunctionDef (2,317) — async function definitions")
print(" 11. Match         (675) — structural pattern matching")
print(" 12. Nonlocal      (445) — nonlocal statement")
print(" 13. TypeAlias     (395) — type alias statement")
print(" 14. AsyncWith     (168) — async with")
print(" 15. AsyncFor      (127) — async for")
print(" 16. TryStar        (47) — try/except* for ExceptionGroups")
