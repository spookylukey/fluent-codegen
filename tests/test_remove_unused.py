import pytest

from fluent_codegen.codegen import (
    Function,
    Name,
    Number,
    Scope,
    String,
    constants,
)
from fluent_codegen.remove_unused_assignments import remove_unused_assignments


def make_func(parent_scope=None):
    """Helper: create a Function named 'f' with no args."""
    if parent_scope is None:
        parent_scope = Scope()
        parent_scope.reserve_name("f")
    func = Function("f", args=[], parent_scope=parent_scope)
    return func


def source(func):
    return func.as_python_source().strip()


# ── positive cases (something gets removed) ──────────────────────────


def test_remove_simple_unused():
    func = make_func()
    x = func.body.scope.create_name("x")
    func.body.create_assignment(x, Number(1))
    func.body.create_return(String("done"))

    remove_unused_assignments(func)

    assert source(func) == "def f():\n    return 'done'"


def test_cascading_removal():
    """x = y, y = 1; x unused ⇒ remove x = y ⇒ y now unused ⇒ remove y = 1."""
    func = make_func()
    y = func.body.scope.create_name("y")
    func.body.create_assignment(y, Number(1))
    x = func.body.scope.create_name("x")
    func.body.create_assignment(x, Name("y", func.body.scope))
    func.body.create_return(String("done"))

    remove_unused_assignments(func)

    assert source(func) == "def f():\n    return 'done'"


def test_remove_unused_in_if_block():
    """Unused assignment inside an if-branch is removed."""
    func = make_func()
    x = func.body.scope.create_name("x")
    if_stmt = func.body.create_if()
    branch = if_stmt.create_if_branch(constants.True_)
    branch.create_assignment(x, Number(42))
    func.body.create_return(String("done"))

    remove_unused_assignments(func)

    # The assignment in the if-branch should be gone; branch body is now empty
    src = source(func)
    assert "x = 42" not in src


def test_remove_unused_in_try_block():
    """Unused assignment inside a try block is removed."""
    from fluent_codegen.codegen import Try

    func = make_func()
    x = func.body.scope.create_name("x")
    scope = func.body.scope
    scope.reserve_name("Exception", is_builtin=True)
    try_stmt = Try(
        catch_exceptions=[Name("Exception", scope)],
        parent_scope=scope,
    )
    try_stmt.try_block.create_assignment(x, Number(1))
    func.body.add_statement(try_stmt)
    func.body.create_return(String("ok"))

    remove_unused_assignments(func)

    src = source(func)
    assert "x = 1" not in src


# ── negative cases (nothing should be removed) ───────────────────────


def test_keep_used_assignment():
    func = make_func()
    x = func.body.scope.create_name("x")
    func.body.create_assignment(x, Number(1))
    func.body.create_return(Name("x", func.body.scope))

    remove_unused_assignments(func)

    assert source(func) == "def f():\n    x = 1\n    return x"


def test_keep_used_in_nested_block():
    """x assigned at top level, used inside an if-branch — keep it."""
    func = make_func()
    x = func.body.scope.create_name("x")
    func.body.create_assignment(x, Number(1))
    if_stmt = func.body.create_if()
    branch = if_stmt.create_if_branch(constants.True_)
    branch.create_return(Name("x", func.body.scope))

    remove_unused_assignments(func)

    src = source(func)
    assert "x = 1" in src


def test_keep_function_args():
    """Function arguments should never be removed even if unused in body."""
    scope = Scope()
    scope.reserve_name("g")
    func = Function("g", args=["a"], parent_scope=scope)
    func.body.create_return(String("hi"))

    remove_unused_assignments(func)

    assert source(func) == "def g(a):\n    return 'hi'"


def test_keep_when_used_in_another_assignment_value():
    """y = 1; x = y; return x  ⇒  y is used (in x's value), so keep both."""
    func = make_func()
    y = func.body.scope.create_name("y")
    func.body.create_assignment(y, Number(1))
    x = func.body.scope.create_name("x")
    func.body.create_assignment(x, Name("y", func.body.scope))
    func.body.create_return(Name("x", func.body.scope))

    remove_unused_assignments(func)

    src = source(func)
    assert "y = 1" in src
    assert "x = y" in src


# ── error case: nested scope ─────────────────────────────────────────


def test_raises_on_nested_function():
    func = make_func()
    func.body.create_function("inner", args=[])

    with pytest.raises(AssertionError, match="nested.*[Ss]cope"):
        remove_unused_assignments(func)


def test_raises_on_nested_class():
    func = make_func()
    func.body.create_class("MyClass")

    with pytest.raises(AssertionError, match="nested.*[Ss]cope"):
        remove_unused_assignments(func)
