"""
Utility to remove unused assignments from a Function body.

Uses ``rewriting_traverse`` to walk the codegen AST.  Only handles a single
function scope — raises ``AssertionError`` if a nested ``Scope`` (e.g. inner
function or class) is encountered.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .codegen import (
    Assignment,
    Block,
    CodeGenAst,
    CodeGenAstList,
    CodeGenAstType,
    Function,
    Name,
    Scope,
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


def traverse(
    node: CodeGenAstType | Sequence[CodeGenAstType],
    func: Callable[[CodeGenAstType], CodeGenAstType],
    *,
    _visited: set[int] | None = None,
    exclude_assignment_target: bool = False,
):
    """
    Apply 'func' to node and all sub CodeGenAst nodes.

    Discovers child nodes by introspecting instance attributes rather than
    relying on a manually-maintained list.  A *visited* set (keyed by
    ``id()``) prevents infinite recursion through circular references
    (e.g. Block.scope → Function → body → Block).

    if `include_func` is not None, it should be a callable that
    decides whether or not to visit a node and it's descendants
    """
    if _visited is None:
        _visited = set()
    node_id = id(node)
    if node_id in _visited:
        return
    _visited.add(node_id)
    if isinstance(node, (CodeGenAst, CodeGenAstList)):
        func(node)
        if exclude_assignment_target and isinstance(node, Assignment):
            exclude_keys = ("target",)
        else:
            exclude_keys = ()

        for key, value in node.__dict__.items():
            if key in exclude_keys:
                continue
            traverse(value, func, _visited=_visited, exclude_assignment_target=exclude_assignment_target)
    elif isinstance(node, (list, tuple)):
        for i in node:
            traverse(i, func, _visited=_visited, exclude_assignment_target=exclude_assignment_target)
    elif isinstance(node, dict):
        for v in node.values():  # type: ignore[reportUnknownVariableType]
            traverse(v, func, _visited=_visited, exclude_assignment_target=exclude_assignment_target)  # type: ignore[reportUnknownVariableType]


def _collect_name_references(func: Function) -> set[str]:
    """Return the set of all name strings referenced as ``Name`` expressions,
    apart from those in LHS of an Assignment
    """
    names: set[str] = set()

    def _visitor(node: CodeGenAstType) -> CodeGenAstType:
        if isinstance(node, Name):
            names.add(node.name)
        return node

    traverse(func.body, _visitor, exclude_assignment_target=True)
    return names


def _assigned_names(blocks: list[Block]) -> set[str]:
    """Return all names that appear on the LHS of an ``Assignment``."""
    result: set[str] = set()
    for block in blocks:
        for stmt in block.statements:
            if isinstance(stmt, Assignment):
                result.update(stmt.names)
    return result


def _remove_once(func: Function, blocks: list[Block]) -> bool:
    """Remove unused assignment statements.  Returns True if anything changed."""
    referenced = _collect_name_references(func)
    assigned = _assigned_names(blocks)
    unused = assigned - referenced
    if not unused:
        return False

    any_changed = False
    for block in blocks:
        original_length = len(block.statements)
        new_statements = [
            stmt for stmt in block.statements if not (isinstance(stmt, Assignment) and set(stmt.names) <= unused)
        ]
        if len(new_statements) < original_length:
            block.statements = new_statements
            any_changed = True

    return any_changed


def remove_unused_assignments(func: Function) -> None:
    """
    Remove statements that assign to a name which is never read.

    Works recursively: if removing ``x = y`` makes ``y`` unused, it will be
    removed too.  Raises ``AssertionError`` on nested scopes.
    """
    blocks = _collect_blocks(func)
    while _remove_once(func, blocks):
        pass
