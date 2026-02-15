"""Compatibility module for generating Python AST.

Provides a curated subset of the stdlib `ast` module used by the codegen module.
"""

import ast
from typing import TypedDict

# This is a very limited subset of Python AST:
# - only the things needed by codegen.py

Add = ast.Add
Assign = ast.Assign
BoolOp = ast.BoolOp
BinOp = ast.BinOp
Compare = ast.Compare
Dict = ast.Dict
Eq = ast.Eq
ExceptHandler = ast.ExceptHandler
Expr = ast.Expr
If = ast.If
List = ast.List
Load = ast.Load
Module = ast.Module
Or = ast.Or
boolop = ast.boolop
Pass = ast.Pass
Return = ast.Return
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
arg = ast.arg
keyword = ast.keyword
walk = ast.walk
Constant = ast.Constant
AST = ast.AST
stmt = ast.stmt
expr = ast.expr


# `compile` builtin needs these attributes on AST nodes.
# It's hard to get something sensible we can put for line/col numbers so we put arbitrary values.


class DefaultAstArgs(TypedDict):
    lineno: int
    col_offset: int


DEFAULT_AST_ARGS: DefaultAstArgs = {"lineno": 1, "col_offset": 1}
# Some AST types have different requirements:
DEFAULT_AST_ARGS_MODULE = dict()
DEFAULT_AST_ARGS_ADD = dict()
DEFAULT_AST_ARGS_ARGUMENTS = dict()


def subscript_slice_object[T](value: T) -> T:
    return value
