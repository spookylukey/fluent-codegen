"""
Utility to remove unused assignments from a Function body.

Uses ``rewriting_traverse`` to walk the codegen AST.  Only handles a single
function scope — raises ``AssertionError`` if a nested ``Scope`` (e.g. inner
function or class) is encountered.
"""

from __future__ import annotations

from .codegen import (
    Assignment,
    Block,
    CodeGenAst,
    CodeGenAstList,
    CodeGenAstType,
    Function,
    Name,
    Scope,
    rewriting_traverse,
)


def _collect_blocks(func: Function) -> list[Block]:
    """Return every Block reachable from *func*.body, checking for nested scopes."""
    blocks: list[Block] = []
    seen: set[int] = set()

    def _walk(obj: object) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, Scope) and obj is not func:
            if type(obj) is not Scope:
                raise AssertionError(f"remove_unused_assignments does not handle nested Scopes ({type(obj).__name__})")
            return

        if isinstance(obj, Block):
            blocks.append(obj)

        # Recurse into attributes, finding Block children generically.
        if isinstance(obj, (CodeGenAst, CodeGenAstList)):
            for value in obj.__dict__.values():
                _walk_value(value)

    def _walk_value(value: object) -> None:
        if isinstance(value, (CodeGenAst, CodeGenAstList, Scope)):
            _walk(value)
        elif isinstance(value, (list, tuple)):
            for item in value:  # type: ignore[reportUnknownVariableType]
                _walk_value(item)  # type: ignore[reportUnknownVariableType]

    # Start from statements inside the body (not the Function itself, which is
    # a Scope we need to skip).
    for stmt in func.body.statements:
        _walk_value(stmt)
    # Also include the body block itself.
    blocks.insert(0, func.body)
    return blocks


def _collect_name_references(func: Function) -> set[str]:
    """Return the set of all name strings referenced as ``Name`` expressions."""
    names: set[str] = set()

    def _visitor(node: CodeGenAstType) -> CodeGenAstType:
        if isinstance(node, Name):
            names.add(node.name)
        return node

    rewriting_traverse(func.body, _visitor)
    return names


def _assigned_names(blocks: list[Block]) -> set[str]:
    """Return all names that appear on the LHS of an ``_Assignment``."""
    result: set[str] = set()
    for block in blocks:
        for stmt in block.statements:
            if isinstance(stmt, Assignment):
                result.add(stmt.name)
    return result


def _remove_once(func: Function, blocks: list[Block]) -> bool:
    """Remove unused assignment statements.  Returns True if anything changed."""
    referenced = _collect_name_references(func)
    assigned = _assigned_names(blocks)
    unused = assigned - referenced
    if not unused:
        return False

    for block in blocks:
        block.statements = [
            stmt for stmt in block.statements if not (isinstance(stmt, Assignment) and stmt.name in unused)
        ]
    return True


def remove_unused_assignments(func: Function) -> None:
    """
    Remove statements that assign to a name which is never read.

    Works recursively: if removing ``x = y`` makes ``y`` unused, it will be
    removed too.  Raises ``AssertionError`` on nested scopes.
    """
    blocks = _collect_blocks(func)
    while _remove_once(func, blocks):
        pass
