"""Compatibility module for generating Python AST.

Provides a curated subset of the stdlib `ast` module used by the codegen module.
"""

import ast
from typing import TypedDict

# This is a very limited subset of Python AST:
# - only the things needed by codegen.py

Add = ast.Add
And = ast.And
Assert = ast.Assert
Assign = ast.Assign
AnnAssign = ast.AnnAssign
BoolOp = ast.BoolOp
BinOp = ast.BinOp
Compare = ast.Compare
Dict = ast.Dict
Div = ast.Div
Eq = ast.Eq
ExceptHandler = ast.ExceptHandler
Expr = ast.Expr
FloorDiv = ast.FloorDiv
Gt = ast.Gt
GtE = ast.GtE
If = ast.If
In = ast.In
List = ast.List
Load = ast.Load
Lt = ast.Lt
LtE = ast.LtE
Mod = ast.Mod
Module = ast.Module
Mult = ast.Mult
MatMult = ast.MatMult
NotEq = ast.NotEq
NotIn = ast.NotIn
Or = ast.Or
Pow = ast.Pow
Sub = ast.Sub
boolop = ast.boolop
cmpop = ast.cmpop
operator = ast.operator
Pass = ast.Pass
Return = ast.Return
Set = ast.Set
Starred = ast.Starred
Store = ast.Store
Subscript = ast.Subscript
Tuple = ast.Tuple
arguments = ast.arguments
JoinedStr = ast.JoinedStr
FormattedValue = ast.FormattedValue
Attribute = ast.Attribute
Call = ast.Call
FunctionDef = ast.FunctionDef
Name = ast.Name
Try = ast.Try
With = ast.With
withitem = ast.withitem
arg = ast.arg
keyword = ast.keyword
ClassDef = ast.ClassDef
walk = ast.walk
fix_missing_locations = ast.fix_missing_locations
unparse = ast.unparse
Constant = ast.Constant
AST = ast.AST
stmt = ast.stmt
expr = ast.expr
Import = ast.Import
ImportFrom = ast.ImportFrom
alias = ast.alias

# `compile` builtin needs these attributes on AST nodes.
# It's hard to get something sensible we can put for line/col numbers so we put arbitrary values.


class DefaultAstArgs(TypedDict):
    lineno: int
    col_offset: int


DEFAULT_AST_ARGS: DefaultAstArgs = {"lineno": 1, "col_offset": 1}
# Some AST types have different requirements:
DEFAULT_AST_ARGS_MODULE: dict[str, object] = dict()
DEFAULT_AST_ARGS_ADD: dict[str, object] = dict()
DEFAULT_AST_ARGS_ARGUMENTS: dict[str, object] = dict()


def subscript_slice_object[T](value: T) -> T:
    return value


class CommentNode(ast.stmt):
    """Custom AST statement node representing a comment.

    This is not a standard Python AST node. It is ignored by ``compile()``
    (callers must strip it first, which ``as_ast()`` does automatically),
    but is rendered by :func:`unparse_with_comments`.
    """

    _fields = ("text",)

    def __init__(self, text: str, **kwargs: int) -> None:
        self.text = text
        super().__init__(**kwargs)  # type: ignore[reportCallIssue]


class _CommentUnparser(ast._Unparser):  # type: ignore[reportAttributeAccessIssue]
    """An unparser that knows how to render :class:`CommentNode`."""

    def visit_CommentNode(self, node: CommentNode) -> None:
        self.fill("# " + node.text)  # type: ignore[reportAttributeAccessIssue]


def unparse_with_comments(node: ast.AST) -> str:
    """Like :func:`ast.unparse`, but also renders :class:`CommentNode` nodes."""
    unparser = _CommentUnparser()
    return unparser.visit(node)  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
