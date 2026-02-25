"""
Utilities for doing Python code generation
"""

from __future__ import annotations

import builtins
import enum
import keyword
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, assert_never, overload, runtime_checkable

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs  # pragma: no cover

from . import ast_compat as py_ast
from .ast_compat import (
    DEFAULT_AST_ARGS,
    DEFAULT_AST_ARGS_ADD,
    DEFAULT_AST_ARGS_ARGUMENTS,
    DEFAULT_AST_ARGS_MODULE,
    CommentNode,
    unparse_with_comments,
)
from .utils import allowable_keyword_arg_name, allowable_name

#: Type alias for comment strings stored in :attr:`Block.statements`.
Comment = str

# This module provides simple utilities for building up Python source code.
# The design originally came from fluent-compiler, so had the following aims
# and constraints:
#
# 1. Performance.
#
#    The resulting Python code should do as little as possible, especially for
#    simple cases.
#
# 2. Correctness (obviously)
#
#    In particular, we should try to make it hard to generate code that is
#    syntactically correct and therefore compiles but doesn't work. We try to
#    make it hard to generate accidental name clashes, or use variables that are
#    not defined.
#
#    Correctness also has a security implication, since the result of this code
#    might be 'exec'ed. To that end:
#     * We build up AST, rather than strings. This eliminates many
#       potential bugs caused by wrong escaping/interpolation.
#     * the `as_ast()` methods are paranoid about input, and do many asserts.
#       We do this even though other layers will usually have checked the
#       input, to allow us to reason locally when checking these methods. These
#       asserts must also have 100% code coverage.
#
# 3. Simplicity
#
#    The resulting Python code should be easy to read and understand.
#
# 4. Predictability
#
#    Since we want to test the resulting source code, we have made some design
#    decisions that aim to ensure things like function argument names are
#    consistent and so can be predicted easily.

# Outside of fluent-compiler, this code will likely be useful for situations
# which have similar aims.


SENSITIVE_FUNCTIONS = {
    # builtin functions that we should never be calling from our code
    # generation. This is a defense-in-depth mechansim to stop our code
    # generation becoming a code execution vulnerability. There should also be
    # higher level code that ensures we are not generating calls to arbitrary
    # Python functions. This is not a comprehensive list of functions we are not
    # using, but functions we definitely don't need and are most likely to be
    # used to execute remote code or to get around safety mechanisms.
    "__import__",
    "__build_class__",
    "apply",
    "compile",
    "eval",
    "exec",
    "execfile",
    "exit",
    "file",
    "globals",
    "locals",
    "open",
    "object",
    "reload",
    "type",
}


class CodeGenAst(ABC):
    """
    Base class representing a simplified Python AST (not the real one).
    Generates real `ast.*` nodes via `as_ast()` method.
    """

    @abstractmethod
    def as_ast(self, *, include_comments: bool = False) -> py_ast.AST: ...

    def as_python_source(self) -> str:
        """Return the Python source code for this AST node."""
        node = self.as_ast(include_comments=True)
        py_ast.fix_missing_locations(node)
        return unparse_with_comments(node)


class CodeGenAstList(ABC):
    """
    Alternative base class to CodeGenAst when we have code that wants to return a
    list of AST objects. These must also be `stmt` objects.
    """

    @abstractmethod
    def as_ast_list(self, allow_empty: bool = True, *, include_comments: bool = False) -> list[py_ast.stmt]: ...

    def as_python_source(self) -> str:
        """Return the Python source code for this AST list."""
        mod = py_ast.Module(body=self.as_ast_list(include_comments=True), type_ignores=[], **DEFAULT_AST_ARGS_MODULE)
        py_ast.fix_missing_locations(mod)
        return unparse_with_comments(mod)


CodeGenAstType = CodeGenAst | CodeGenAstList


class Scope:
    """Track name reservations and assignments within a lexical scope."""

    def __init__(self, parent_scope: Scope | None = None):
        self.parent_scope = parent_scope
        self.names: set[str] = set()
        self._function_arg_reserved_names: set[str] = set()
        self._assignments: set[str] = set()

    def is_name_in_use(self, name: str) -> bool:
        """Return whether *name* is already reserved in this scope or any parent."""
        if name in self.names:
            return True

        if self.parent_scope is None:
            return False

        return self.parent_scope.is_name_in_use(name)

    def is_name_reserved_function_arg(self, name: str) -> bool:
        """Return whether *name* is reserved for use as a function argument."""
        if name in self._function_arg_reserved_names:
            return True

        if self.parent_scope is None:
            return False

        return self.parent_scope.is_name_reserved_function_arg(name)

    def is_name_reserved(self, name: str) -> bool:
        """Return whether *name* is reserved for any purpose in this scope."""
        return self.is_name_in_use(name) or self.is_name_reserved_function_arg(name)

    def reserve_name(
        self,
        requested: str,
        function_arg: bool = False,
        is_builtin: bool = False,
    ):
        """
        Reserve a name as being in use in a scope.

        Pass function_arg=True if this is a function argument.
        """

        def _add(final: str):
            self.names.add(final)
            return final

        if function_arg:
            if self.is_name_reserved_function_arg(requested):
                assert not self.is_name_in_use(requested)
                return _add(requested)
            if self.is_name_reserved(requested):
                raise AssertionError(f"Cannot use '{requested}' as argument name as it is already in use")

        cleaned = cleanup_name(requested)

        attempt = cleaned
        count = 2  # instance without suffix is regarded as 1
        # To avoid shadowing of global names in local scope, we
        # take into account parent scope when assigning names.

        def _is_name_allowed(name: str) -> bool:
            # We need to also protect against using keywords ('class', 'def' etc.)
            # i.e. count all keywords as 'used'.
            # However, some builtins are also keywords (e.g. 'None'), and so
            # if a builtin is being reserved, don't check against the keyword list
            if (not is_builtin) and keyword.iskeyword(name):
                return False

            return not self.is_name_reserved(name)

        while not _is_name_allowed(attempt):
            attempt = cleaned + "_" + str(count)
            count += 1

        return _add(attempt)

    def reserve_function_arg_name(self, name: str):
        """
        Reserve a name for *later* use as a function argument. This does not result
        in that name being considered 'in use' in the current scope, but will
        avoid the name being assigned for any use other than as a function argument.
        """
        # To keep things simple, and the generated code predictable, we reserve
        # names for all function arguments in a separate scope, and insist on
        # the exact names
        if self.is_name_reserved(name):
            raise AssertionError(f"Can't reserve '{name}' as function arg name as it is already reserved")
        self._function_arg_reserved_names.add(name)

    def has_assignment(self, name: str) -> bool:
        """Return whether *name* has been assigned a value in this scope."""
        return name in self._assignments

    def register_assignment(self, name: str) -> None:
        """Record that *name* has been assigned a value in this scope."""
        self._assignments.add(name)

    def create_name(self, name: str) -> Name:
        """Reserve *name* (or a variant) and return a :class:`Name` expression for it."""
        reserved = self.reserve_name(name)
        return Name(reserved, self)

    def name(self, name: str) -> Name:
        """Return a :class:`Name` expression for an already-reserved name."""
        return Name(name, self)


_IDENTIFIER_SANITIZER_RE = re.compile("[^a-zA-Z0-9_]")
_IDENTIFIER_START_RE = re.compile("^[a-zA-Z_]")


def cleanup_name(name: str) -> str:
    """
    Convert name to a allowable identifier
    """
    # See https://docs.python.org/2/reference/lexical_analysis.html#grammar-token-identifier
    name = _IDENTIFIER_SANITIZER_RE.sub("", name)
    if not _IDENTIFIER_START_RE.match(name):
        name = "n" + name
    return name


class Statement(CodeGenAst):
    """Base class for code-generation nodes that represent Python statements."""

    pass


@runtime_checkable
class SupportsNameAssignment(Protocol):
    """Protocol for nodes that can report whether they assign to a given name."""

    def has_assignment_for_name(self, name: str) -> bool: ...


class Annotation(Statement):
    """A bare type annotation without a value, e.g. ``x: int``."""

    def __init__(self, name: str, annotation: Expression):
        self.name = name
        self.annotation = annotation

    def as_ast(self, *, include_comments: bool = False):
        if not allowable_name(self.name):
            raise AssertionError(f"Expected {self.name} to be a valid Python identifier")
        return py_ast.AnnAssign(
            target=py_ast.Name(id=self.name, ctx=py_ast.Store(), **DEFAULT_AST_ARGS),
            annotation=self.annotation.as_ast(),
            simple=1,
            value=None,
            **DEFAULT_AST_ARGS,
        )

    def has_assignment_for_name(self, name: str) -> bool:
        return self.name == name


class Assignment(Statement):
    """A variable assignment statement, optionally with a type annotation."""

    def __init__(self, target: str | Target, value: Expression, /, *, type_hint: Expression | None = None):
        if isinstance(target, str):
            self.name: str | None = target
            self.target: Target | None = None
        elif is_target(target):
            self.name = _target_name(target)
            self.target = target
            if type_hint is not None and not isinstance(target, Name):
                raise AssertionError("Type hints are only supported for simple name assignment targets")
        else:
            raise AssertionError(f"Invalid assignment target: {target!r}")
        self.value = value
        self.type_hint = type_hint

    def as_ast(self, *, include_comments: bool = False):
        if self.target is not None:
            target_ast = _target_as_store_ast(self.target)
            if self.type_hint is None:
                return py_ast.Assign(
                    targets=[target_ast],
                    value=self.value.as_ast(),
                    **DEFAULT_AST_ARGS,
                )
            else:
                return py_ast.AnnAssign(
                    target=target_ast,
                    annotation=self.type_hint.as_ast(),
                    simple=1,
                    value=self.value.as_ast(),
                    **DEFAULT_AST_ARGS,
                )
        # Legacy str-based path
        assert self.name is not None
        if not allowable_name(self.name):
            raise AssertionError(f"Expected {self.name} to be a valid Python identifier")
        if self.type_hint is None:
            return py_ast.Assign(
                targets=[py_ast.Name(id=self.name, ctx=py_ast.Store(), **DEFAULT_AST_ARGS)],
                value=self.value.as_ast(),
                **DEFAULT_AST_ARGS,
            )
        else:
            return py_ast.AnnAssign(
                target=py_ast.Name(id=self.name, ctx=py_ast.Store(), **DEFAULT_AST_ARGS),
                annotation=self.type_hint.as_ast(),
                simple=1,
                value=self.value.as_ast(),
                **DEFAULT_AST_ARGS,
            )

    def has_assignment_for_name(self, name: str) -> bool:
        return self.name == name


class Block(CodeGenAstList):
    """An ordered sequence of statements sharing a common :class:`Scope`."""

    def __init__(self, scope: Scope, parent_block: Block | None = None):
        self.scope = scope
        # We allow `Expression` here for things like MethodCall which
        # are bare expressions that are still useful for side effects.
        # `Comment` (str) entries are rendered as ``# text`` comments.
        self.statements: list[Block | Statement | Expression | Comment] = []
        self.parent_block = parent_block

    def as_ast_list(self, allow_empty: bool = True, *, include_comments: bool = False) -> list[py_ast.stmt]:
        retval: list[py_ast.stmt] = []
        for s in self.statements:
            if isinstance(s, str):
                # Comment
                if include_comments:
                    retval.append(CommentNode(s, **DEFAULT_AST_ARGS))  # type: ignore[reportArgumentType]
                continue
            if isinstance(s, CodeGenAstList):
                retval.extend(s.as_ast_list(allow_empty=True, include_comments=include_comments))
            elif isinstance(s, Statement):
                ast_obj = s.as_ast(include_comments=include_comments)
                assert isinstance(ast_obj, py_ast.stmt), (
                    "Statement object return {ast_obj} which is not a subclass of py_ast.stmt"
                )
                retval.append(ast_obj)
            else:
                # Things like bare function/method calls need to be wrapped
                # in `Expr` to match the way Python parses.
                retval.append(py_ast.Expr(s.as_ast(include_comments=include_comments), **DEFAULT_AST_ARGS))

        if len(retval) == 0 and not allow_empty:
            return [py_ast.Pass(**DEFAULT_AST_ARGS)]
        return retval

    def add_comment(self, text: str, *, wrap: int | None = None) -> None:
        """Add a ``# text`` comment line at the current position in the block.

        If *wrap* is given as an integer, long lines are wrapped at word
        boundaries so that no comment line exceeds *wrap* characters
        (including the ``#`` prefix and space).
        """
        if wrap is not None:
            # Account for the "# " prefix (2 chars) that will be added during rendering.
            effective_width = max(wrap - 2, 1)
            lines = textwrap.wrap(text, width=effective_width)
            if not lines:
                # Empty or whitespace-only text – preserve as a single blank comment.
                lines = [""]
            for line in lines:
                self.statements.append(line)
        else:
            self.statements.append(text)

    def add_statement(self, statement: Statement | Block | Expression) -> None:
        """Append a statement, block, or bare expression to this block."""
        self.statements.append(statement)
        if isinstance(statement, Block):
            if statement.parent_block is None:
                statement.parent_block = self
            else:
                if statement.parent_block != self:
                    raise AssertionError(
                        f"Block {statement} is already child of {statement.parent_block}, can't reassign to {self}"
                    )

    def create_import(self, module: str, as_: str | None = None) -> tuple[Import, Name]:
        """Create an ``import`` statement, reserve the resulting name, and add it to this block."""
        return_name_object: Name
        if as_ is not None:
            # "import foo as bar" results in `bar` name being assigned.
            if not allowable_name(as_):
                raise AssertionError(f"{as_!r} is not an allowable 'as' name")
            if self.scope.is_name_in_use(as_):
                raise AssertionError(f"{as_!r} is already assigned in the scope")
            as_name_object = self.scope.create_name(as_)
            return_name_object = as_name_object
        else:
            as_name_object = None
            # "import foo" results in `foo` name being assigned
            # "import foo.bar" also results in `foo` being reserved.
            dotted_parts = module.split(".")
            for part in dotted_parts:
                if not allowable_name(part):
                    raise AssertionError(f"{module!r} not an allowable 'import' name")
            name_to_assign = dotted_parts[0]
            # We can't rename, so don't use `reserve_name` or `create_name`.
            # We also need to allow for multiple imports, like `import foo.bar` then `import foo.baz`

            if not self.scope.is_name_in_use(name_to_assign):
                self.scope.reserve_name(name_to_assign)
            return_name_object = self.scope.name(name_to_assign)

        import_statement = Import(module=module, as_=as_name_object)
        self.add_statement(import_statement)
        return import_statement, return_name_object

    def create_import_from(self, *, from_: str, import_: str, as_: str | None = None) -> tuple[ImportFrom, Name]:
        """Create a ``from ... import`` statement, reserve the resulting name, and add it to this block."""
        return_name_object: Name
        if as_ is not None:
            # "from foo import bar as baz" results in `baz` name being assigned.
            if not allowable_name(as_):
                raise AssertionError(f"{as_!r} is not an allowable 'as' name")
            if self.scope.is_name_in_use(as_):
                raise AssertionError(f"{as_!r} is already assigned in the scope")
            as_name_object = self.scope.create_name(as_)
            return_name_object = as_name_object
        else:
            as_name_object = None
            # Check the dotted bit.
            dotted_parts = from_.split(".")
            for part in dotted_parts:
                if not allowable_name(part):
                    raise AssertionError(f"{from_!r} not an allowable 'import' name")

            # Check the `import_` for clashes.
            name_to_assign = import_

            if self.scope.is_name_in_use(name_to_assign):
                raise AssertionError(f"{name_to_assign!r} is already assigned in the scope")
            return_name_object = self.scope.create_name(name_to_assign)

        import_statement = ImportFrom(from_module=from_, import_=import_, as_=as_name_object)
        self.add_statement(import_statement)
        return import_statement, return_name_object

    # Safe alternatives to Block.statements being manipulated directly:
    def create_assignment(
        self, name: str | Name, value: Expression, *, type_hint: Expression | None = None, allow_multiple: bool = False
    ):
        """
        Adds an assigment of the form:

           x = value
        """
        if isinstance(name, Name):
            name = name.name
        if not self.scope.is_name_in_use(name):
            raise AssertionError(f"Cannot assign to unreserved name '{name}'")

        if self.scope.has_assignment(name):
            if not allow_multiple:
                raise AssertionError(f"Have already assigned to '{name}' in this scope")
        else:
            self.scope.register_assignment(name)

        self.add_statement(Assignment(name, value, type_hint=type_hint))

    def create_annotation(self, name: str, annotation: Expression) -> Name:
        """
        Adds a bare type annotation of the form::

            x: int

        Reserves the name and adds the annotation statement to the block.
        """
        name_obj = self.scope.create_name(name)
        self.scope.register_assignment(name_obj.name)
        self.add_statement(Annotation(name_obj.name, annotation))
        return name_obj

    def create_field(self, name: str, annotation: Expression, *, default: Expression | None = None) -> Name:
        """
        Create a typed field, typically used in dataclass bodies.

        If *default* is provided, creates an annotated assignment::

            x: int = 0

        Otherwise, creates a bare annotation::

            x: int
        """
        if default is not None:
            name_obj = self.scope.create_name(name)
            self.scope.register_assignment(name_obj.name)
            self.add_statement(Assignment(name_obj.name, default, type_hint=annotation))
            return name_obj
        else:
            return self.create_annotation(name, annotation)

    def create_function(
        self,
        name: str,
        args: Sequence[str | FunctionArg],
        decorators: Sequence[Expression] | None = None,
        return_type: Expression | None = None,
    ) -> tuple[Function, Name]:
        """
        Reserve a name for a function, create the Function and add the function statement
        to the block.
        """
        name_obj = self.scope.create_name(name)
        func = Function(
            name_obj.name, args=args, parent_scope=self.scope, decorators=decorators, return_type=return_type
        )
        self.add_statement(func)
        return func, name_obj

    def create_class(
        self,
        name: str,
        bases: Sequence[Expression] | None = None,
        decorators: Sequence[Expression] | None = None,
    ) -> tuple[Class, Name]:
        """
        Reserve a name for a class, create the Class and add the class statement
        to the block.
        """
        name_obj = self.scope.create_name(name)
        cls = Class(name_obj.name, parent_scope=self.scope, bases=bases, decorators=decorators)
        self.add_statement(cls)
        return cls, name_obj

    def create_return(self, value: Expression) -> None:
        """Add a ``return`` statement to this block."""
        self.add_statement(Return(value))

    def create_assert(self, test: Expression, msg: Expression | None = None) -> None:
        """Add an ``assert`` statement to this block."""
        self.add_statement(Assert(test, msg))

    def create_if(self) -> If:
        """
        Create an If statement, add it to this block, and return it.

        Usage::

            if_stmt = block.create_if()
            if_block = if_stmt.add_if(condition)
            if_block.create_return(value)
        """
        if_statement = If(self.scope, parent_block=self)
        self.add_statement(if_statement)
        return if_statement

    def create_with(self, context_expr: Expression, target: Name | None = None) -> With:
        """
        Create a With statement, add it to this block, and return it

        Usage::

            with_stmt = block.create_with(expr, "f")
            with_stmt.body.create_return(value)
        """
        with_statement = With(context_expr, target=target, parent_scope=self.scope, parent_block=self)
        self.add_statement(with_statement)
        return with_statement

    def has_assignment_for_name(self, name: str) -> bool:
        """Return whether *name* is assigned anywhere in this block or its parents."""
        for s in self.statements:
            if isinstance(s, SupportsNameAssignment) and s.has_assignment_for_name(name):
                return True
        if self.parent_block is not None:
            return self.parent_block.has_assignment_for_name(name)
        return False


class Module(Block, CodeGenAst):
    """Top-level module block with its own :class:`Scope` pre-loaded with builtins."""

    def __init__(self, reserve_builtins: bool = True):
        scope = Scope(parent_scope=None)
        if reserve_builtins:
            for name in dir(builtins):
                scope.reserve_name(name, is_builtin=True)
        Block.__init__(self, scope)

    def as_ast(self, *, include_comments: bool = False) -> py_ast.Module:
        return py_ast.Module(
            body=self.as_ast_list(include_comments=include_comments), type_ignores=[], **DEFAULT_AST_ARGS_MODULE
        )

    def as_python_source(self) -> str:
        mod = self.as_ast(include_comments=True)
        py_ast.fix_missing_locations(mod)
        return unparse_with_comments(mod)


class ArgKind(enum.Enum):
    """The kind of a function argument."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    KEYWORD_ONLY = "keyword_only"


@dataclass(frozen=True)
class FunctionArg:
    """A function argument with a name, kind, and optional default value."""

    name: str
    kind: ArgKind = ArgKind.POSITIONAL_OR_KEYWORD
    default: Expression | None = None
    annotation: Expression | None = None

    @classmethod
    def positional(
        cls, name: str, *, default: Expression | None = None, annotation: Expression | None = None
    ) -> FunctionArg:
        """Create a positional-only argument."""
        return cls(name=name, kind=ArgKind.POSITIONAL_ONLY, default=default, annotation=annotation)

    @classmethod
    def keyword(
        cls, name: str, *, default: Expression | None = None, annotation: Expression | None = None
    ) -> FunctionArg:
        """Create a keyword-only argument."""
        return cls(name=name, kind=ArgKind.KEYWORD_ONLY, default=default, annotation=annotation)

    @classmethod
    def standard(
        cls, name: str, *, default: Expression | None = None, annotation: Expression | None = None
    ) -> FunctionArg:
        """Create a positional-or-keyword argument (the Python default)."""
        return cls(name=name, kind=ArgKind.POSITIONAL_OR_KEYWORD, default=default, annotation=annotation)


def _normalize_args(args: Sequence[str | FunctionArg]) -> list[FunctionArg]:
    """Normalize a mixed list of str and FunctionArg into a list of FunctionArg."""
    return [FunctionArg(name=a) if isinstance(a, str) else a for a in args]


def _validate_arg_order(args: list[FunctionArg]) -> None:
    """Validate that args are in the correct order:
    positional-only, then positional-or-keyword, then keyword-only.
    Within each group, defaults must come after non-defaults.
    """
    # Check kind ordering
    KIND_ORDER = {
        ArgKind.POSITIONAL_ONLY: 0,
        ArgKind.POSITIONAL_OR_KEYWORD: 1,
        ArgKind.KEYWORD_ONLY: 2,
    }
    prev_order = -1
    for arg in args:
        order = KIND_ORDER[arg.kind]
        if order < prev_order:
            raise ValueError(
                f"Argument '{arg.name}' of kind {arg.kind.value} "
                f"is out of order: positional-only args must come first, "
                f"then positional-or-keyword, then keyword-only"
            )
        prev_order = order

    # Check default ordering within positional groups
    # (positional-only and positional-or-keyword share defaults list,
    #  so non-default can't follow default across these groups)
    seen_default_in_positional = False
    for arg in args:
        if arg.kind in (ArgKind.POSITIONAL_ONLY, ArgKind.POSITIONAL_OR_KEYWORD):
            if arg.default is not None:
                seen_default_in_positional = True
            elif seen_default_in_positional:
                raise ValueError(f"Non-default argument '{arg.name}' follows default argument in positional arguments")

    # keyword-only args can have defaults in any order (Python allows it)


class Function(Scope, Statement):
    """A function definition statement that also acts as a :class:`Scope` for its body."""

    def __init__(
        self,
        name: str,
        args: Sequence[str | FunctionArg] | None = None,
        parent_scope: Scope | None = None,
        decorators: Sequence[Expression] | None = None,
        return_type: Expression | None = None,
    ):
        super().__init__(parent_scope=parent_scope)
        self.body = Block(self)
        self.func_name = name
        self.decorators: list[Expression] = list(decorators) if decorators else []
        self.return_type: Expression | None = return_type
        self._args: list[FunctionArg] = []
        if args is not None:
            self.add_args(args)

    @property
    def args(self) -> Sequence[FunctionArg]:
        """Return the function's arguments as a read-only sequence."""
        return tuple(self._args)

    def add_args(self, args: Sequence[str | FunctionArg]) -> None:
        """Add arguments to the function, with the same validation as in __init__."""
        normalized = _normalize_args(args)
        combined = self._args + normalized
        _validate_arg_order(combined)
        for arg in normalized:
            if self.is_name_in_use(arg.name):
                raise AssertionError(f"Can't use '{arg.name}' as function argument name because it shadows other names")
            self.reserve_name(arg.name, function_arg=True)
        self._args = combined

    def as_ast(self, *, include_comments: bool = False) -> py_ast.stmt:
        if not allowable_name(self.func_name):
            raise AssertionError(f"Expected '{self.func_name}' to be a valid Python identifier")
        for arg in self._args:
            if not allowable_name(arg.name):
                raise AssertionError(f"Expected '{arg.name}' to be a valid Python identifier")

        def _make_arg(a: FunctionArg) -> py_ast.arg:
            return py_ast.arg(
                arg=a.name,
                annotation=a.annotation.as_ast() if a.annotation is not None else None,
                **DEFAULT_AST_ARGS,
            )

        posonlyargs = [_make_arg(a) for a in self._args if a.kind == ArgKind.POSITIONAL_ONLY]
        regular_args = [_make_arg(a) for a in self._args if a.kind == ArgKind.POSITIONAL_OR_KEYWORD]
        kwonlyargs = [_make_arg(a) for a in self._args if a.kind == ArgKind.KEYWORD_ONLY]

        # defaults: right-aligned to posonlyargs + regular_args
        positional_all = [a for a in self._args if a.kind in (ArgKind.POSITIONAL_ONLY, ArgKind.POSITIONAL_OR_KEYWORD)]
        defaults = [a.default.as_ast() for a in positional_all if a.default is not None]

        # kw_defaults: one entry per kwonlyarg, None if no default
        kw_defaults: list[py_ast.expr | None] = [
            a.default.as_ast() if a.default is not None else None for a in self._args if a.kind == ArgKind.KEYWORD_ONLY
        ]

        return py_ast.FunctionDef(
            name=self.func_name,
            args=py_ast.arguments(
                posonlyargs=posonlyargs,
                args=regular_args,
                vararg=None,
                kwonlyargs=kwonlyargs,
                kw_defaults=kw_defaults,
                kwarg=None,
                defaults=defaults,
                **DEFAULT_AST_ARGS_ARGUMENTS,
            ),
            body=self.body.as_ast_list(allow_empty=False, include_comments=include_comments),
            decorator_list=[d.as_ast() for d in self.decorators],
            type_params=[],
            returns=self.return_type.as_ast() if self.return_type is not None else None,
            **DEFAULT_AST_ARGS,
        )

    def create_return(self, value: Expression):
        """Add a ``return`` statement to the function body."""
        self.body.create_return(value)


class Class(Scope, Statement):
    """A class definition statement that also acts as a :class:`Scope` for its body."""

    def __init__(
        self,
        name: str,
        parent_scope: Scope | None = None,
        bases: Sequence[Expression] | None = None,
        decorators: Sequence[Expression] | None = None,
    ):
        super().__init__(parent_scope=parent_scope)
        self.body = Block(self)
        self.class_name = name
        self.bases: list[Expression] = list(bases) if bases else []
        self.decorators: list[Expression] = list(decorators) if decorators else []

    def as_ast(self, *, include_comments: bool = False) -> py_ast.stmt:
        if not allowable_name(self.class_name):
            raise AssertionError(f"Expected '{self.class_name}' to be a valid Python identifier")
        return py_ast.ClassDef(
            name=self.class_name,
            bases=[b.as_ast() for b in self.bases],
            keywords=[],
            body=self.body.as_ast_list(allow_empty=False, include_comments=include_comments),
            decorator_list=[d.as_ast() for d in self.decorators],
            type_params=[],
            **DEFAULT_AST_ARGS,
        )


class Return(Statement):
    """A ``return`` statement."""

    def __init__(self, value: Expression):
        self.value = value

    def as_ast(self, *, include_comments: bool = False):
        return py_ast.Return(self.value.as_ast(), **DEFAULT_AST_ARGS)

    def __repr__(self):
        return f"Return({repr(self.value)}"


class Assert(Statement):
    """An ``assert`` statement with an optional message."""

    def __init__(self, test: Expression, msg: Expression | None = None):
        self.test = test
        self.msg = msg

    def as_ast(self, *, include_comments: bool = False):
        msg_ast = self.msg.as_ast() if self.msg is not None else None
        return py_ast.Assert(
            test=self.test.as_ast(),
            msg=msg_ast,
            **DEFAULT_AST_ARGS,
        )

    def __repr__(self):
        return f"Assert({repr(self.test)}, {repr(self.msg)})"


class If(Statement):
    """A compound ``if``/``elif``/``else`` statement."""

    def __init__(self, parent_scope: Scope, parent_block: Block | None = None):
        # We model a "compound if statement" as a list of if blocks
        # (if/elif/elif etc), each with their own condition, with a final else
        # block. Note this is quite different from Python's AST for the same
        # thing, so conversion to AST is more complex because of this.
        self.if_blocks: list[Block] = []
        self.conditions: list[Expression] = []
        self._parent_block = parent_block
        self.else_block = Block(parent_scope, parent_block=self._parent_block)
        self._parent_scope = parent_scope

    def create_if_branch(self, condition: Expression) -> Block:
        """
        Create new if branch with a condition.
        """
        new_if = Block(self._parent_scope, parent_block=self._parent_block)
        self.if_blocks.append(new_if)
        self.conditions.append(condition)
        return new_if

    def finalize(self) -> Block | Statement:
        """Return a simplified node: the else block if there are no conditions, otherwise *self*."""
        if not self.if_blocks:
            # Unusual case of no conditions, only default case, but it
            # simplifies other code to be able to handle this uniformly. We can
            # replace this if statement with a single unconditional block.
            return self.else_block
        return self

    def as_ast(self, *, include_comments: bool = False) -> py_ast.If:
        if len(self.if_blocks) == 0:
            raise AssertionError("Should have called `finalize` on If")
        if_ast = empty_If()
        current_if = if_ast
        previous_if = None
        for condition, if_block in zip(self.conditions, self.if_blocks):
            current_if.test = condition.as_ast()
            current_if.body = if_block.as_ast_list(include_comments=include_comments)
            if previous_if is not None:
                previous_if.orelse.append(current_if)

            previous_if = current_if
            current_if = empty_If()

        if self.else_block.statements:
            assert previous_if is not None
            previous_if.orelse = self.else_block.as_ast_list(include_comments=include_comments)

        return if_ast


class With(Statement):
    """A ``with`` statement."""

    def __init__(
        self,
        context_expr: Expression,
        target: Name | None = None,
        *,
        parent_scope: Scope,
        parent_block: Block | None = None,
    ):
        self.context_expr = context_expr
        self.target = target
        self._parent_scope = parent_scope
        self._parent_block = parent_block
        self.body = Block(parent_scope, parent_block=parent_block)

    def as_ast(self, *, include_comments: bool = False) -> py_ast.With:
        optional_vars = None
        if self.target is not None:
            optional_vars = py_ast.Name(id=self.target.name, ctx=py_ast.Store(), **DEFAULT_AST_ARGS)

        return py_ast.With(
            items=[
                py_ast.withitem(
                    context_expr=self.context_expr.as_ast(),
                    optional_vars=optional_vars,
                )
            ],
            body=self.body.as_ast_list(allow_empty=False, include_comments=include_comments),
            **DEFAULT_AST_ARGS,
        )


class Try(Statement):
    """A ``try``/``except``/``else`` statement."""

    def __init__(self, catch_exceptions: Sequence[Expression], parent_scope: Scope):
        self.catch_exceptions = catch_exceptions
        self.try_block = Block(parent_scope)
        self.except_block = Block(parent_scope)
        self.else_block = Block(parent_scope)

    def as_ast(self, *, include_comments: bool = False) -> py_ast.Try:
        return py_ast.Try(
            body=self.try_block.as_ast_list(allow_empty=False, include_comments=include_comments),
            handlers=[
                py_ast.ExceptHandler(
                    type=(
                        self.catch_exceptions[0].as_ast()
                        if len(self.catch_exceptions) == 1
                        else py_ast.Tuple(
                            elts=[e.as_ast() for e in self.catch_exceptions],
                            ctx=py_ast.Load(),
                            **DEFAULT_AST_ARGS,
                        )
                    ),
                    name=None,
                    body=self.except_block.as_ast_list(allow_empty=False, include_comments=include_comments),
                    **DEFAULT_AST_ARGS,
                )
            ],
            orelse=self.else_block.as_ast_list(allow_empty=True, include_comments=include_comments),
            finalbody=[],
            **DEFAULT_AST_ARGS,
        )

    def has_assignment_for_name(self, name: str) -> bool:
        if (
            self.try_block.has_assignment_for_name(name) or self.else_block.has_assignment_for_name(name)
        ) and self.except_block.has_assignment_for_name(name):
            return True
        return False


class Import(Statement):
    """
    Simple import statements, supporting:
    - import foo
    - import foo as bar
    - import foo.bar
    - import foo.bar as baz

    Use via `Block.create_import`

    We deliberately don't support multiple imports - these should
    be cleaned up later using a linter on the generated code, if desired.
    """

    def __init__(self, module: str, as_: Name | None) -> None:
        self.module = module
        self.as_ = as_

    def as_ast(self, *, include_comments: bool = False) -> py_ast.AST:
        if self.as_ is None:
            # No alias needed:
            return py_ast.Import(names=[py_ast.alias(name=self.module)], **DEFAULT_AST_ARGS)
        else:
            return py_ast.Import(names=[py_ast.alias(name=self.module, asname=self.as_.name)], **DEFAULT_AST_ARGS)


class ImportFrom(Statement):
    """
    ``from ... import`` statement, supporting:

    - ``from foo import bar``
    - ``from foo import bar as baz``

    Use via `Block.create_import_from`

    We deliberately don't support multiple imports - these should
    be cleaned up later using a linter on the generated code, if desired.
    """

    def __init__(self, from_module: str, import_: str, as_: Name | None) -> None:
        self.from_module = from_module
        self.import_ = import_
        self.as_ = as_

    def as_ast(self, *, include_comments: bool = False) -> py_ast.AST:
        if self.as_ is None:
            # No alias needed:
            return py_ast.ImportFrom(
                module=self.from_module,
                names=[py_ast.alias(name=self.import_)],
                level=0,
                **DEFAULT_AST_ARGS,
            )
        else:
            return py_ast.ImportFrom(
                module=self.from_module,
                names=[py_ast.alias(name=self.import_, asname=self.as_.name)],
                level=0,
                **DEFAULT_AST_ARGS,
            )


class Expression(CodeGenAst):
    """Base class for code-generation nodes that represent Python expressions."""

    @abstractmethod
    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr: ...

    # Some utilities for easy chaining:

    def attr(self, attribute: str, /) -> Attr:
        """Return an :class:`Attr` expression for accessing *attribute* on this expression."""
        return Attr(self, attribute)

    def call(
        self,
        args: Sequence[Expression] | None = None,
        kwargs: Mapping[str, Expression] | None = None,
    ) -> Call:
        """Return a :class:`Call` expression invoking this expression."""
        return Call(self, args or [], kwargs or {})

    def method_call(
        self,
        attribute: str,
        args: Sequence[Expression] | None = None,
        kwargs: Mapping[str, Expression] | None = None,
    ) -> Call:
        """Return a :class:`Call` expression for a method call on this expression."""
        return self.attr(attribute).call(args, kwargs)

    def subscript(self, slice: Expression, /) -> Subscript:
        """Return a :class:`Subscript` expression indexing this expression."""
        return Subscript(self, slice)

    # Arithmetic operators

    def add(self, other: Expression, /) -> Add:
        """Return an :class:`Add` (``+``) expression."""
        return Add(self, other)

    def sub(self, other: Expression, /) -> Sub:
        """Return a :class:`Sub` (``-``) expression."""
        return Sub(self, other)

    def mul(self, other: Expression, /) -> Mul:
        """Return a :class:`Mul` (``*``) expression."""
        return Mul(self, other)

    def div(self, other: Expression, /) -> Div:
        """Return a :class:`Div` (``/``) expression."""
        return Div(self, other)

    def floordiv(self, other: Expression, /) -> FloorDiv:
        """Return a :class:`FloorDiv` (``//``) expression."""
        return FloorDiv(self, other)

    def mod(self, other: Expression, /) -> Mod:
        """Return a :class:`Mod` (``%``) expression."""
        return Mod(self, other)

    def pow(self, other: Expression, /) -> Pow:
        """Return a :class:`Pow` (``**``) expression."""
        return Pow(self, other)

    def matmul(self, other: Expression, /) -> MatMul:
        """Return a :class:`MatMul` (``@``) expression."""
        return MatMul(self, other)

    # Comparison operators

    def eq(self, other: Expression, /) -> Equals:
        """Return an :class:`Equals` (``==``) expression."""
        return Equals(self, other)

    def ne(self, other: Expression, /) -> NotEquals:
        """Return a :class:`NotEquals` (``!=``) expression."""
        return NotEquals(self, other)

    def lt(self, other: Expression, /) -> Lt:
        """Return a :class:`Lt` (``<``) expression."""
        return Lt(self, other)

    def gt(self, other: Expression, /) -> Gt:
        """Return a :class:`Gt` (``>``) expression."""
        return Gt(self, other)

    def le(self, other: Expression, /) -> LtE:
        """Return a :class:`LtE` (``<=``) expression."""
        return LtE(self, other)

    def ge(self, other: Expression, /) -> GtE:
        """Return a :class:`GtE` (``>=``) expression."""
        return GtE(self, other)

    # Boolean operators

    def and_(self, other: Expression, /) -> And:
        """Return an :class:`And` (``and``) expression."""
        return And(self, other)

    def or_(self, other: Expression, /) -> Or:
        """Return an :class:`Or` (``or``) expression."""
        return Or(self, other)

    # Membership operators

    def in_(self, other: Expression, /) -> In:
        """Return an :class:`In` (``in``) expression."""
        return In(self, other)

    def not_in(self, other: Expression, /) -> NotIn:
        """Return a :class:`NotIn` (``not in``) expression."""
        return NotIn(self, other)

    # Unpacking

    def starred(self) -> Starred:
        """Return a :class:`Starred` (``*self``) unpacking expression."""
        return Starred(self)


class String(Expression):
    """A string literal expression."""

    def __init__(self, string_value: str):
        self.string_value = string_value

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Constant(
            self.string_value,
            kind=None,  # 3.8, indicates no prefix, needed only for tests
            **DEFAULT_AST_ARGS,
        )

    def __repr__(self):
        return f"String({repr(self.string_value)})"

    def __eq__(self, other: object):
        return isinstance(other, String) and other.string_value == self.string_value


class Bool(Expression):
    """A boolean literal expression (``True`` or ``False``)."""

    def __init__(self, value: bool):
        self.value = value

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Constant(self.value, **DEFAULT_AST_ARGS)

    def __repr__(self):
        return f"Bool({self.value!r})"


class Bytes(Expression):
    """A bytes literal expression."""

    def __init__(self, value: bytes):
        self.value = value

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Constant(self.value, **DEFAULT_AST_ARGS)

    def __repr__(self):
        return f"Bytes({self.value!r})"


class Number(Expression):
    """A numeric literal expression (``int`` or ``float``)."""

    def __init__(self, number: int | float):
        self.number = number

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Constant(self.number, **DEFAULT_AST_ARGS)

    def __repr__(self):
        return f"Number({repr(self.number)})"


class List(Expression):
    """A list literal expression."""

    def __init__(self, items: Sequence[Expression]):
        self.items = items

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.List(elts=[i.as_ast() for i in self.items], ctx=py_ast.Load(), **DEFAULT_AST_ARGS)


class Tuple(Expression):
    """A tuple literal expression."""

    def __init__(self, items: Sequence[Expression]):
        self.items = items

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Tuple(elts=[i.as_ast() for i in self.items], ctx=py_ast.Load(), **DEFAULT_AST_ARGS)


class Set(Expression):
    """A set literal expression, using ``set([])`` for the empty case."""

    def __init__(self, items: Sequence[Expression]):
        self.items = items

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        if len(self.items) == 0:
            # {} is a dict literal in Python, so empty sets must use set([])
            return py_ast.Call(
                func=py_ast.Name(id="set", ctx=py_ast.Load(), **DEFAULT_AST_ARGS),
                args=[py_ast.List(elts=[], ctx=py_ast.Load(), **DEFAULT_AST_ARGS)],
                keywords=[],
                **DEFAULT_AST_ARGS,
            )
        return py_ast.Set(elts=[i.as_ast() for i in self.items], **DEFAULT_AST_ARGS)


class Dict(Expression):
    """A dict literal expression."""

    def __init__(self, pairs: Sequence[tuple[Expression, Expression]]):
        self.pairs = pairs

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Dict(
            keys=[k.as_ast() for k, _ in self.pairs],
            values=[v.as_ast() for _, v in self.pairs],
            **DEFAULT_AST_ARGS,
        )


class StringJoinBase(Expression):
    """Base class for string concatenation expressions."""

    def __init__(self, parts: Sequence[Expression]):
        self.parts = parts

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(repr(p) for p in self.parts)}])"

    @classmethod
    def build(cls: type[StringJoinBase], parts: Sequence[Expression]) -> StringJoinBase | Expression:
        """
        Build a string join operation, but return a simpler expression if possible.
        """
        # Merge adjacent String objects.
        new_parts: list[Expression] = []
        for part in parts:
            if len(new_parts) > 0 and isinstance(new_parts[-1], String) and isinstance(part, String):
                new_parts[-1] = String(new_parts[-1].string_value + part.string_value)
            else:
                new_parts.append(part)
        parts = new_parts

        # See if we can eliminate the StringJoin altogether
        if len(parts) == 0:
            return String("")
        if len(parts) == 1:
            return parts[0]
        return cls(parts)


class FStringJoin(StringJoinBase):
    """Join string parts using an f-string expression."""

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        # f-strings
        values: list[py_ast.expr] = []
        for part in self.parts:
            if isinstance(part, String):
                values.append(part.as_ast())
            else:
                values.append(
                    py_ast.FormattedValue(
                        value=part.as_ast(),
                        conversion=-1,
                        format_spec=None,
                        **DEFAULT_AST_ARGS,
                    )
                )
        return py_ast.JoinedStr(values=values, **DEFAULT_AST_ARGS)


class ConcatJoin(StringJoinBase):
    """Join string parts using ``+`` concatenation."""

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        # Concatenate with +
        left = self.parts[0].as_ast()
        for part in self.parts[1:]:
            right = part.as_ast()
            left = py_ast.BinOp(
                left=left,
                op=py_ast.Add(**DEFAULT_AST_ARGS_ADD),
                right=right,
                **DEFAULT_AST_ARGS,
            )
        return left


# For CPython, f-strings give a measurable improvement over concatenation,
# so make that default

StringJoin = FStringJoin


class Name(Expression):
    """A reference to a named variable that must already be reserved in its :class:`Scope`."""

    def __init__(self, name: str, scope: Scope):
        if not scope.is_name_in_use(name):
            raise AssertionError(f"Cannot refer to undefined name '{name}'")
        self.name = name

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        if not allowable_name(self.name, allow_builtin=True):
            raise AssertionError(f"Expected {self.name} to be a valid Python identifier")
        return py_ast.Name(id=self.name, ctx=py_ast.Load(), **DEFAULT_AST_ARGS)

    def __eq__(self, other: object):
        return type(other) is type(self) and other.name == self.name

    def __repr__(self):
        return f"Name({repr(self.name)})"


class Attr(Expression):
    """An attribute access expression (e.g. ``obj.attr``)."""

    def __init__(self, value: Expression, attribute: str) -> None:
        self.value = value
        if not allowable_name(attribute, allow_builtin=True):
            raise AssertionError(f"Expected {attribute} to be a valid Python identifier")
        self.attribute = attribute

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Attribute(value=self.value.as_ast(), attr=self.attribute, **DEFAULT_AST_ARGS)


class Starred(Expression):
    """A starred (unpacking) expression (e.g. ``*args``)."""

    def __init__(self, value: Expression):
        self.value = value

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Starred(value=self.value.as_ast(), ctx=py_ast.Load(), **DEFAULT_AST_ARGS)

    def __repr__(self):
        return f"Starred({self.value!r})"


def function_call(
    function_name: str,
    args: Sequence[Expression],
    kwargs: Mapping[str, Expression],
    scope: Scope,
) -> Expression:
    """Create a function call expression, validating that the name exists in *scope*."""
    if not scope.is_name_in_use(function_name):
        raise AssertionError(f"Cannot call unknown function '{function_name}'")
    if function_name in SENSITIVE_FUNCTIONS:
        raise AssertionError(f"Disallowing call to '{function_name}'")

    return Name(name=function_name, scope=scope).call(args, kwargs)


class Call(Expression):
    """A function or method call expression."""

    def __init__(
        self,
        value: Expression,
        args: Sequence[Expression],
        kwargs: Mapping[str, Expression],
    ):
        self.value = value
        self.args = list(args)
        self.kwargs = dict(kwargs)

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:

        for name in self.kwargs.keys():
            if not allowable_keyword_arg_name(name):
                raise AssertionError(f"Expected {name} to be a valid Fluent NamedArgument name")

        if any(not allowable_name(name) for name in self.kwargs.keys()):
            # This branch covers function arg names like 'foo-bar', which are
            # allowable in languages like Fluent, but not normally in Python. We work around
            # this using `my_function(**{'foo-bar': baz})` syntax.

            # (If we only wanted to exec the resulting AST, this branch is technically not
            # necessary, since it is the Python parser that disallows `foo-bar` as an identifier,
            # and we are by-passing that by creating AST directly. However, to produce something
            # that can be decompiled to valid Python, we solve the general case).

            kwarg_pairs = list(sorted(self.kwargs.items()))
            kwarg_names, kwarg_values = [k for k, _ in kwarg_pairs], [v for _, v in kwarg_pairs]
            return py_ast.Call(
                func=self.value.as_ast(),
                args=[arg.as_ast() for arg in self.args],
                keywords=[
                    py_ast.keyword(
                        arg=None,
                        value=py_ast.Dict(
                            keys=[py_ast.Constant(k, kind=None, **DEFAULT_AST_ARGS) for k in kwarg_names],
                            values=[v.as_ast() for v in kwarg_values],
                            **DEFAULT_AST_ARGS,
                        ),
                        **DEFAULT_AST_ARGS,
                    )
                ],
                **DEFAULT_AST_ARGS,
            )

        # Normal `my_function(foo=bar)` syntax
        return py_ast.Call(
            func=self.value.as_ast(),
            args=[arg.as_ast() for arg in self.args],
            keywords=[
                py_ast.keyword(arg=name, value=value.as_ast(), **DEFAULT_AST_ARGS)
                for name, value in self.kwargs.items()
            ],
            **DEFAULT_AST_ARGS,
        )

    def __repr__(self):
        return f"Call({self.value!r}, {self.args}, {self.kwargs})"


def method_call(
    obj: Expression,
    method_name: str,
    args: Sequence[Expression],
    kwargs: Mapping[str, Expression],
):
    """Create a method call expression on *obj*."""
    return obj.attr(method_name).call(args=args, kwargs=kwargs)


class Subscript(Expression):
    """A subscript (indexing) expression (e.g. ``obj[key]``)."""

    def __init__(self, value: Expression, slice: Expression):
        self.value = value
        self.slice = slice

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Subscript(
            value=self.value.as_ast(),
            slice=py_ast.subscript_slice_object(self.slice.as_ast()),
            ctx=py_ast.Load(),
            **DEFAULT_AST_ARGS,
        )


#: Type alias for valid assignment target expressions.
#: A :class:`Name`, :class:`Attr`, or :class:`Subscript` expression.
Target = Name | Attr | Subscript


def is_target(value: object) -> TypeIs[Target]:
    """Return whether *value* is a valid assignment :data:`Target`."""
    return isinstance(value, (Name, Attr, Subscript))


def _target_as_store_ast(target: Target) -> py_ast.Name | py_ast.Attribute | py_ast.Subscript:
    """Convert a Target expression to an AST node with Store context."""
    node = target.as_ast()
    if isinstance(node, (py_ast.Name, py_ast.Attribute, py_ast.Subscript)):
        node.ctx = py_ast.Store()
        return node
    raise AssertionError(f"Unexpected AST node type for target: {type(node)}")  # pragma: no cover


def _target_name(target: Target) -> str | None:
    """Extract the top-level assigned name from a Target, if it is a plain Name."""
    if isinstance(target, Name):
        return target.name
    return None


create_class_instance = function_call


class NoneExpr(Expression):
    """A ``None`` literal expression."""

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Constant(value=None, **DEFAULT_AST_ARGS)


class BinaryOperator(Expression):
    """Base class for binary operator expressions with a left and right operand."""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right


class ArithOp(BinaryOperator, ABC):
    """Arithmetic binary operator (ast.BinOp)."""

    op: ClassVar[type[py_ast.operator]]

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.BinOp(
            left=self.left.as_ast(),
            op=self.op(**DEFAULT_AST_ARGS_ADD),
            right=self.right.as_ast(),
            **DEFAULT_AST_ARGS,
        )


class Add(ArithOp):
    """Addition (``+``) operator."""

    op = py_ast.Add


class Sub(ArithOp):
    """Subtraction (``-``) operator."""

    op = py_ast.Sub


class Mul(ArithOp):
    """Multiplication (``*``) operator."""

    op = py_ast.Mult


class Div(ArithOp):
    """True division (``/``) operator."""

    op = py_ast.Div


class FloorDiv(ArithOp):
    """Floor division (``//``) operator."""

    op = py_ast.FloorDiv


class Mod(ArithOp):
    """Modulo (``%``) operator."""

    op = py_ast.Mod


class Pow(ArithOp):
    """Exponentiation (``**``) operator."""

    op = py_ast.Pow


class MatMul(ArithOp):
    """Matrix multiplication (``@``) operator."""

    op = py_ast.MatMult


class CompareOp(BinaryOperator, ABC):
    """Comparison operator (ast.Compare)."""

    op: ClassVar[type[py_ast.cmpop]]

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.Compare(
            left=self.left.as_ast(),
            comparators=[self.right.as_ast()],
            ops=[self.op()],
            **DEFAULT_AST_ARGS,
        )


class Equals(CompareOp):
    """Equality (``==``) comparison."""

    op = py_ast.Eq


class NotEquals(CompareOp):
    """Inequality (``!=``) comparison."""

    op = py_ast.NotEq


class Lt(CompareOp):
    """Less-than (``<``) comparison."""

    op = py_ast.Lt


class Gt(CompareOp):
    """Greater-than (``>``) comparison."""

    op = py_ast.Gt


class LtE(CompareOp):
    """Less-than-or-equal (``<=``) comparison."""

    op = py_ast.LtE


class GtE(CompareOp):
    """Greater-than-or-equal (``>=``) comparison."""

    op = py_ast.GtE


class In(CompareOp):
    """Membership (``in``) comparison."""

    op = py_ast.In


class NotIn(CompareOp):
    """Non-membership (``not in``) comparison."""

    op = py_ast.NotIn


class BoolOp(BinaryOperator, ABC):
    """Boolean operator (``and`` / ``or``)."""

    op: ClassVar[type[py_ast.boolop]]

    def as_ast(self, *, include_comments: bool = False) -> py_ast.expr:
        return py_ast.BoolOp(
            op=self.op(),
            values=[self.left.as_ast(), self.right.as_ast()],
            **DEFAULT_AST_ARGS,
        )


class And(BoolOp):
    """Logical ``and`` operator."""

    op = py_ast.And


class Or(BoolOp):
    """Logical ``or`` operator."""

    op = py_ast.Or


def traverse(ast_node: py_ast.AST, func: Callable[[py_ast.AST], None]):
    """
    Apply 'func' to ast_node (which is `ast.*` object)
    """
    for node in py_ast.walk(ast_node):
        func(node)


def simplify(codegen_ast: CodeGenAstType, simplifier: Callable[[CodeGenAstType, list[bool]], CodeGenAst]):
    """Repeatedly apply *simplifier* to *codegen_ast* until no more changes are made."""
    changes = [True]

    # Wrap `simplifier` (which takes additional `changes` arg)
    # into function that take just `node`, as required by rewriting_traverse
    def rewriter(node: CodeGenAstType) -> CodeGenAstType:
        return simplifier(node, changes)

    while any(changes):
        changes[:] = []
        rewriting_traverse(codegen_ast, rewriter)
    return codegen_ast


def rewriting_traverse(
    node: CodeGenAstType | Sequence[CodeGenAstType],
    func: Callable[[CodeGenAstType], CodeGenAstType],
    _visited: set[int] | None = None,
):
    """
    Apply 'func' to node and all sub CodeGenAst nodes.

    Discovers child nodes by introspecting instance attributes rather than
    relying on a manually-maintained list.  A *visited* set (keyed by
    ``id()``) prevents infinite recursion through circular references
    (e.g. Block.scope → Function → body → Block).
    """
    if _visited is None:
        _visited = set()
    if isinstance(node, (CodeGenAst, CodeGenAstList)):
        node_id = id(node)
        if node_id in _visited:
            return
        _visited.add(node_id)
        new_node = func(node)
        if new_node is not node:
            morph_into(node, new_node)
        for value in node.__dict__.values():
            rewriting_traverse(value, func, _visited)
    elif isinstance(node, (list, tuple)):
        for i in node:
            rewriting_traverse(i, func, _visited)
    elif isinstance(node, dict):
        for v in node.values():  # type: ignore[reportUnknownVariableType]
            rewriting_traverse(v, func, _visited)  # type: ignore[reportUnknownVariableType]


def morph_into(item: object, new_item: object) -> None:
    """Mutate *item* in-place to behave identically to *new_item* while preserving identity."""
    item.__class__ = new_item.__class__
    item.__dict__ = new_item.__dict__


def empty_If() -> py_ast.If:
    """
    Create an empty If ast node. The `test` attribute
    must be added later.
    """
    return py_ast.If(test=None, orelse=[], **DEFAULT_AST_ARGS)  # type: ignore[reportArgumentType]


type PythonObj = (
    bool
    | str
    | bytes
    | int
    | float
    | None
    | list[PythonObj]
    | tuple[PythonObj, ...]
    | set[PythonObj]
    | frozenset[PythonObj]
    | dict[PythonObj, PythonObj]
)


@overload
def auto(value: bool) -> Bool: ...  # type: ignore[overload-overlap]  # bool before int/float is intentional
@overload
def auto(value: str) -> String: ...
@overload
def auto(value: bytes) -> Bytes: ...
@overload
def auto(value: int) -> Number: ...
@overload
def auto(value: float) -> Number: ...
@overload
def auto(value: None) -> NoneExpr: ...
@overload
def auto(value: list[PythonObj]) -> List: ...
@overload
def auto(value: tuple[PythonObj, ...]) -> Tuple: ...
@overload
def auto(value: set[PythonObj]) -> Set: ...
@overload
def auto(value: frozenset[PythonObj]) -> Set: ...
@overload
def auto(value: dict[PythonObj, PythonObj]) -> Dict: ...


def auto(value: PythonObj) -> Expression:
    """
    Create a codegen Expression from a plain Python object.

    Supports bool, str, bytes, int, float, None, and recursively
    list, tuple, set, frozenset, and dict.
    """
    if isinstance(value, bool):
        return Bool(value)
    elif isinstance(value, str):
        return String(value)
    elif isinstance(value, bytes):
        return Bytes(value)
    elif isinstance(value, (int, float)):
        return Number(value)
    elif value is None:
        return constants.None_
    elif isinstance(value, list):
        return List([auto(item) for item in value])
    elif isinstance(value, tuple):
        return Tuple([auto(item) for item in value])
    elif isinstance(value, (set, frozenset)):
        return Set([auto(item) for item in sorted(value, key=repr)])
    elif isinstance(value, dict):  # type: ignore[reportUnnecessaryIsInstance]
        return Dict([(auto(k), auto(v)) for k, v in value.items()])
    assert_never(value)


class constants:
    """
    Useful pre-made Expression constants
    """

    None_: NoneExpr = NoneExpr()
    True_: Bool = auto(True)
    False_: Bool = auto(False)
