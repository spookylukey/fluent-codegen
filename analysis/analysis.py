#!/usr/bin/env python3
"""Analyze coverage of Python AST nodes by fluent-codegen."""

# AST node counts from 1975 files (3.3M total nodes)
# Only including nodes that represent user-visible syntax
# (excluding Load, Store, Del which are internal context nodes)

nodes = {
    # Expressions
    "Name": 601345,
    "Constant": 590377,
    "Attribute": 208669,
    "Call": 175320,
    "Tuple": 71141,
    "List": 40369,
    "Subscript": 33677,
    "BinOp": 29100,
    "Compare": 27114,
    "UnaryOp": 19585,
    "BoolOp": 6288,
    "FormattedValue": 7785,
    "JoinedStr": 5278,
    "Dict": 4936,
    "Starred": 1445,
    "IfExp": 1669,
    "Set": 729,
    "Lambda": 699,
    "ListComp": 1472,
    "GeneratorExp": 945,
    "Yield": 1353,
    "DictComp": 256,
    "YieldFrom": 216,
    "NamedExpr": 187,
    "Await": 146,
    "SetComp": 82,
    "Slice": 5235,
    # Statements
    "Assign": 88821,
    "Expr": 72863,  # expression-as-statement
    "If": 33251,
    "FunctionDef": 25775,
    "Return": 21398,
    "ImportFrom": 9763,
    "For": 6672,
    "Raise": 5191,
    "ClassDef": 4193,
    "Import": 4271,
    "Assert": 3833,
    "AnnAssign": 3707,
    "AugAssign": 3022,
    "Try": 2935,
    "With": 1762,
    "Pass": 1490,
    "Continue": 985,
    "While": 752,
    "Break": 642,
    "Delete": 494,
    "AsyncFunctionDef": 59,
    "Global": 116,
    "Nonlocal": 14,
    "AsyncFor": 10,
    "AsyncWith": 2,
    "Match": 24,
    "TypeAlias": 22,
}

# What fluent-codegen covers
covered = {
    # Expressions
    "Name",  # Name class
    "Constant",  # String, Number, Bool, Bytes, NoneExpr
    "Attribute",  # Attr class
    "Call",  # Call class
    "Tuple",  # Tuple class
    "List",  # List class
    "Subscript",  # Subscript class
    "BinOp",  # ArithOp subclasses (Add, Sub, Mul, etc.)
    "Compare",  # CompareOp subclasses (Equals, Lt, etc.)
    "UnaryOp",  # Not, UAdd, USub, Invert
    "BoolOp",  # And, Or
    "FormattedValue",  # via FStringJoin
    "JoinedStr",  # FStringJoin
    "Dict",  # Dict class
    "Starred",  # Starred class
    "Set",  # Set class
    # Statements
    "Assign",  # Assignment class
    "Expr",  # Expressions used as statements in Block
    "If",  # If class
    "FunctionDef",  # Function class
    "Return",  # Return class
    "ImportFrom",  # ImportFrom class
    "ClassDef",  # Class class
    "Import",  # Import class
    "Assert",  # Assert class
    "AnnAssign",  # Annotation / Assignment with type_hint
    "Try",  # Try class
    "With",  # With class
    "Pass",  # used as empty block filler
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
print(f"{'Rank':>4}  {'AST Node':<22} {'Count':>8}  {'% of total':>10}  Notes")
print(f"{'----':>4}  {'--------':<22} {'-----':>8}  {'----------':>10}  -----")

notes = {
    "For": "for loops — very common control flow",
    "Raise": "raise statements — essential for error handling",
    "Slice": "slice notation a[1:3] — used in Subscript",
    "AugAssign": "x += 1, x -= 1, etc.",
    "IfExp": "ternary: x if cond else y",
    "ListComp": "[x for x in items]",
    "Yield": "generator yield",
    "Continue": "loop continue",
    "While": "while loops",
    "GeneratorExp": "(x for x in items)",
    "Lambda": "lambda expressions",
    "Break": "loop break",
    "Delete": "del statement",
    "DictComp": "{k: v for k, v in items}",
    "YieldFrom": "yield from expr",
    "NamedExpr": "walrus operator :=",
    "Await": "await expression",
    "Global": "global statement",
    "SetComp": "{x for x in items}",
    "AsyncFunctionDef": "async def",
    "Match": "match/case (3.10+)",
    "TypeAlias": "type X = ... (3.12+)",
    "Nonlocal": "nonlocal statement",
    "AsyncFor": "async for",
    "AsyncWith": "async with",
}

ranked = sorted(not_covered.items(), key=lambda x: -x[1])
for i, (name, count) in enumerate(ranked, 1):
    pct = count / total * 100
    note = notes.get(name, "")
    print(f"{i:>4}  {name:<22} {count:>8,}  {pct:>9.2f}%  {note}")

print()
print("=" * 70)
print("TOP SUGGESTIONS FOR ADDING TO FLUENT-CODEGEN")
print("=" * 70)
print()
print("High priority (very common syntax):")
print("  1. For       — for loops are the 7th most common statement")
print("  2. Raise     — essential for any error-handling code generation")
print("  3. Slice     — needed for proper a[start:stop:step] support")
print("  4. AugAssign — augmented assignment (+=, -=, etc.)")
print("  5. IfExp     — inline ternary expressions")
print()
print("Medium priority (commonly used):")
print("  6. ListComp  — list comprehensions")
print("  7. Yield     — generator functions")
print("  8. Continue  — loop control")
print("  9. While     — while loops")
print(" 10. GeneratorExp — generator expressions")
print(" 11. Lambda    — lambda expressions")
print(" 12. Break     — loop control")
print()
print("Lower priority (less common):")
print(" 13. Delete    — del statement")
print(" 14. DictComp  — dict comprehensions")
print(" 15. YieldFrom — yield from")
print(" 16. NamedExpr — walrus operator")
print(" 17. Await / AsyncFunctionDef / AsyncFor / AsyncWith — async support")
print(" 18. Global / Nonlocal")
print(" 19. Match     — structural pattern matching")
print(" 20. SetComp   — set comprehensions")
print(" 21. TypeAlias — type alias statement")
