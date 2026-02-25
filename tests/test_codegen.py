import ast
import builtins
import keyword
import textwrap

import pytest
from hypothesis import example, given
from hypothesis.strategies import text

from fluent_codegen import codegen
from fluent_codegen.utils import allowable_name


def normalize_python(txt: str):
    return textwrap.dedent(txt.rstrip()).strip()


def as_source_code(codegen_ast: codegen.CodeGenAstType) -> str:
    return codegen_ast.as_python_source()


def assert_code_equal(code1: str | codegen.CodeGenAstType, code2: str | codegen.CodeGenAstType):
    if not isinstance(code1, str):
        code1 = code1.as_python_source()
    if not isinstance(code2, str):
        code2 = code2.as_python_source()
    assert normalize_python(code1) == normalize_python(code2)


# Hypothesis strategies
non_keyword_text = text().filter(lambda t: t not in keyword.kwlist)
non_builtin_text = non_keyword_text.filter(lambda t: t not in dir(builtins))


# --- Scope tests ---


def test_reserve_name():
    scope = codegen.Scope()
    name1 = scope.reserve_name("name")
    name2 = scope.reserve_name("name")
    assert name1 == "name"
    assert name1 != name2
    assert name2 == "name_2"


def test_reserve_name_function_arg_disallowed():
    scope = codegen.Scope()
    scope.reserve_name("name")
    with pytest.raises(AssertionError):
        scope.reserve_name("name", function_arg=True)


def test_reserve_name_function_arg():
    scope = codegen.Scope()
    scope.reserve_function_arg_name("arg_name")
    scope.reserve_name("myfunc")
    func = codegen.Function("myfunc", args=["arg_name"], parent_scope=scope)
    assert not func.is_name_reserved("arg_name2")


def test_reserve_name_nested():
    parent = codegen.Scope()
    parent_name = parent.reserve_name("name")
    assert parent_name == "name"

    child1 = codegen.Scope(parent_scope=parent)
    child2 = codegen.Scope(parent_scope=parent)

    child1_name = child1.reserve_name("name")
    assert child1_name != parent_name

    child2_name = child2.reserve_name("name")
    assert child2_name != parent_name

    # Children can have same names, they don't shadow each other.
    assert child1_name == child2_name


def test_reserve_name_after_reserve_function_arg():
    scope = codegen.Scope()
    scope.reserve_function_arg_name("my_arg")
    name = scope.reserve_name("my_arg")
    assert name == "my_arg_2"


def test_reserve_function_arg_after_reserve_name():
    scope = codegen.Scope()
    scope.reserve_name("my_arg")
    with pytest.raises(AssertionError):
        scope.reserve_function_arg_name("my_arg")


def test_scope_name_helper():
    scope = codegen.Scope()
    name = scope.reserve_name("name")
    ref1 = codegen.Name(name, scope)
    ref2 = scope.name(name)
    assert ref1 == ref2


# --- Function tests ---


def test_function():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["myarg1", "myarg2"], parent_scope=module.scope)
    assert_code_equal(
        func,
        """
        def myfunc(myarg1, myarg2):
            pass
        """,
    )


def test_function_return():
    module = codegen.Module()
    func = codegen.Function("myfunc", parent_scope=module.scope)
    func.create_return(codegen.String("Hello"))
    assert_code_equal(
        func,
        """
        def myfunc():
            return 'Hello'
        """,
    )


def test_create_assert():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["x"], parent_scope=module.scope)
    func.body.create_assert(func.name("x"))
    assert_code_equal(
        func,
        """
        def myfunc(x):
            assert x
        """,
    )


def test_create_assert_with_message():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["x"], parent_scope=module.scope)
    func.body.create_assert(func.name("x"), codegen.String("x must be truthy"))
    assert_code_equal(
        func,
        """
        def myfunc(x):
            assert x, 'x must be truthy'
        """,
    )


def test_assert_class_directly():
    assert_stmt = codegen.Assert(codegen.Number(1))
    assert_code_equal(assert_stmt, "assert 1")


def test_assert_class_with_message_directly():
    assert_stmt = codegen.Assert(codegen.Number(0), codegen.String("fail"))
    assert_code_equal(assert_stmt, "assert 0, 'fail'")


def test_assert_repr():
    assert_stmt = codegen.Assert(codegen.Number(1))
    assert "Assert(" in repr(assert_stmt)


def test_function_bad_name():
    module = codegen.Module()
    func = codegen.Function("my func", args=[], parent_scope=module.scope)
    with pytest.raises(AssertionError):
        as_source_code(func)


def test_function_bad_arg():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["my arg"], parent_scope=module.scope)
    with pytest.raises(AssertionError):
        as_source_code(func)


def test_create_function():
    module = codegen.Module()
    func, func_name = module.create_function("my_func", args=["my_arg"])
    assert_code_equal(
        module,
        """
        def my_func(my_arg):
            pass
        """,
    )
    assert_code_equal(func_name, "my_func")


def test_function_args_name_check():
    module = codegen.Module()
    module.scope.reserve_name("my_arg")
    func_name = module.scope.reserve_name("myfunc")
    with pytest.raises(AssertionError):
        codegen.Function(func_name, args=["my_arg"], parent_scope=module.scope)


def test_function_add_args():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["a"], parent_scope=module.scope)
    func.add_args(["b", "c"])
    func.create_return(func.name("a"))
    assert_code_equal(
        func,
        """
        def myfunc(a, b, c):
            return a
        """,
    )


def test_function_add_args_with_function_arg():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, parent_scope=module.scope)
    func.add_args([codegen.FunctionArg.keyword("x", default=codegen.Number(1))])
    func.create_return(func.name("x"))
    assert_code_equal(
        func,
        """
        def myfunc(*, x=1):
            return x
        """,
    )


def test_function_add_args_name_shadow_check():
    module = codegen.Module()
    module.scope.reserve_name("my_arg")
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, parent_scope=module.scope)
    with pytest.raises(AssertionError):
        func.add_args(["my_arg"])


def test_function_add_args_duplicate_check():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["a"], parent_scope=module.scope)
    with pytest.raises(AssertionError):
        func.add_args(["a"])


def test_function_add_args_order_check():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=[codegen.FunctionArg.keyword("x")], parent_scope=module.scope)
    with pytest.raises(ValueError):
        func.add_args([codegen.FunctionArg.positional("y")])


def test_function_add_args_reserves_name():
    """Added args should be usable as Name references within the function."""
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, parent_scope=module.scope)
    func.add_args(["my_arg"])
    ref = codegen.Name("my_arg", func)
    assert_code_equal(ref, "my_arg")


def test_function_args_property_empty():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, parent_scope=module.scope)
    assert func.args == ()


def test_function_args_property():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["a", "b"], parent_scope=module.scope)
    assert len(func.args) == 2
    assert func.args[0].name == "a"
    assert func.args[1].name == "b"


def test_function_args_property_with_function_arg():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(
        func_name,
        args=[
            codegen.FunctionArg.positional("x"),
            "y",
            codegen.FunctionArg.keyword("z", default=codegen.Number(1)),
        ],
        parent_scope=module.scope,
    )
    assert len(func.args) == 3
    assert func.args[0].name == "x"
    assert func.args[0].kind == codegen.ArgKind.POSITIONAL_ONLY
    assert func.args[1].name == "y"
    assert func.args[1].kind == codegen.ArgKind.POSITIONAL_OR_KEYWORD
    assert func.args[2].name == "z"
    assert func.args[2].kind == codegen.ArgKind.KEYWORD_ONLY


def test_function_args_property_after_add_args():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["a"], parent_scope=module.scope)
    assert len(func.args) == 1
    func.add_args(["b", "c"])
    assert len(func.args) == 3
    assert func.args[2].name == "c"


def test_function_args_property_is_readonly():
    """Mutating the returned tuple should not affect the function's internal args."""
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["a", "b"], parent_scope=module.scope)
    args = func.args
    assert len(args) == 2
    # tuple is immutable, so this should be a fresh copy each time
    assert func.args is not func.args or isinstance(func.args, tuple)
    assert len(func.args) == 2


def test_function_args_name_reserved_check():
    module = codegen.Module()
    module.scope.reserve_function_arg_name("my_arg")
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["my_arg"], parent_scope=module.scope)
    func.create_return(func.name("my_arg"))
    assert_code_equal(
        func,
        """
        def myfunc(my_arg):
            return my_arg
        """,
    )


# --- Name reference tests ---


def test_name():
    scope = codegen.Scope()
    ref = scope.create_name("name")
    assert as_source_code(ref) == "name"


def test_name_check():
    scope = codegen.Scope()
    with pytest.raises(AssertionError):
        codegen.Name("name", scope)


def test_name_function_arg_check():
    scope = codegen.Scope()
    func_name = scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["my_arg"], parent_scope=scope)
    # Can't use undefined 'some_name'
    with pytest.raises(AssertionError):
        codegen.Name("some_name", func)
    # But can use function argument 'my_arg'
    ref = codegen.Name("my_arg", func)
    assert_code_equal(ref, "my_arg")


def test_name_bad():
    module = codegen.Module()
    name = module.scope.reserve_name("name")
    ref = codegen.Name(name, module.scope)
    ref.name = "bad name"
    with pytest.raises(AssertionError):
        as_source_code(ref)


def test_create_name():
    module = codegen.Module()
    var = module.scope.create_name("myname")
    assert isinstance(var, codegen.Name)
    assert module.scope.is_name_in_use("myname")


# --- Assignment tests ---


def test_create_assignment_unreserved():
    scope = codegen.Module()
    with pytest.raises(AssertionError):
        scope.create_assignment("x", codegen.String("a string"))


def test_create_assignment_reserved():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_assignment(name, codegen.String("a string"))
    assert_code_equal(
        module,
        """
        x = 'a string'
        """,
    )


def test_create_assignment_with_name_object():
    module = codegen.Module()
    name_obj = module.scope.create_name("x")
    module.create_assignment(name_obj, codegen.String("a string"))
    assert_code_equal(
        module,
        """
        x = 'a string'
        """,
    )


def test_create_assignment_bad():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_assignment(name, codegen.String("a string"))
    stmt = module.statements[0]
    assert isinstance(stmt, codegen.Assignment)
    stmt.names = ["something with a space"]
    with pytest.raises(AssertionError):
        as_source_code(module)


def test_create_assignment_type_hint():
    module = codegen.Module()
    module.create_assignment(
        module.scope.create_name("x"), codegen.String("a string"), type_hint=module.scope.name("str")
    )
    assert_code_equal(
        module,
        """
        x: str = 'a string'
        """,
    )


def test_multiple_create_assignment():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_assignment(name, codegen.String("a string"))
    with pytest.raises(AssertionError):
        module.create_assignment(name, codegen.String("another string"))


def test_multiple_create_assignment_in_inherited_scope():
    scope = codegen.Scope()
    scope.reserve_name("myfunc")
    func = codegen.Function("myfunc", args=[], parent_scope=scope)
    try_ = codegen.Try([], func)
    name = func.reserve_name("name")

    try_.try_block.create_assignment(name, codegen.Number(1))
    with pytest.raises(AssertionError):
        try_.try_block.create_assignment(name, codegen.Number(2))
    with pytest.raises(AssertionError):
        try_.except_block.create_assignment(name, codegen.Number(2))
    try_.except_block.create_assignment(name, codegen.Number(2), allow_multiple=True)


# --- Target assignment tests ---


def test_assign_to_attr():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).attr("bar")
    module.add_statement(codegen.Assignment(target, codegen.String("baz")))
    assert_code_equal(
        module,
        """
        foo.bar = 'baz'
        """,
    )


def test_assign_to_subscript():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).subscript(codegen.String("bar"))
    module.add_statement(codegen.Assignment(target, codegen.String("baz")))
    assert_code_equal(
        module,
        """
        foo['bar'] = 'baz'
        """,
    )


def test_assign_to_nested_attr():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).attr("bar").attr("baz")
    module.add_statement(codegen.Assignment(target, codegen.Number(42)))
    assert_code_equal(
        module,
        """
        foo.bar.baz = 42
        """,
    )


def test_assign_to_subscript_of_attr():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).attr("bar").subscript(codegen.String("key"))
    module.add_statement(codegen.Assignment(target, codegen.Number(1)))
    assert_code_equal(
        module,
        """
        foo.bar['key'] = 1
        """,
    )


def test_assign_to_attr_of_subscript():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).subscript(codegen.Number(0)).attr("bar")
    module.add_statement(codegen.Assignment(target, codegen.Number(1)))
    assert_code_equal(
        module,
        """
        foo[0].bar = 1
        """,
    )


def test_assign_to_name_target_with_type_hint():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope)
    module.add_statement(codegen.Assignment(target, codegen.String("baz"), type_hint=module.scope.name("str")))
    assert_code_equal(
        module,
        """
        foo: str = 'baz'
        """,
    )


def test_assign_to_attr_with_type_hint_rejected():
    module = codegen.Module()
    name = module.scope.reserve_name("foo")
    target = codegen.Name(name, module.scope).attr("bar")
    with pytest.raises(AssertionError, match="Type hints are only supported"):
        codegen.Assignment(target, codegen.String("baz"), type_hint=module.scope.name("str"))


def test_assign_tuple_unpack():
    module = codegen.Module()
    x = module.scope.reserve_name("x")
    y = module.scope.reserve_name("y")
    target = (codegen.Name(x, module.scope), codegen.Name(y, module.scope))
    module.add_statement(codegen.Assignment(target, codegen.Name(module.scope.reserve_name("vals"), module.scope)))
    assert_code_equal(
        module,
        """
        x, y = vals
        """,
    )


def test_assign_tuple_unpack_with_attr_and_subscript():
    module = codegen.Module()
    foo = module.scope.reserve_name("foo")
    bar = module.scope.reserve_name("bar")
    target = (
        codegen.Name(foo, module.scope).attr("x"),
        codegen.Name(bar, module.scope).subscript(codegen.Number(0)),
    )
    module.add_statement(codegen.Assignment(target, codegen.Name(module.scope.reserve_name("vals"), module.scope)))
    assert_code_equal(
        module,
        """
        foo.x, bar[0] = vals
        """,
    )


def test_assign_nested_tuple_unpack():
    module = codegen.Module()
    a = module.scope.reserve_name("a")
    b = module.scope.reserve_name("b")
    c = module.scope.reserve_name("c")
    s = module.scope
    target = (
        codegen.Name(a, s),
        (codegen.Name(b, s), codegen.Name(c, s)),
    )
    module.add_statement(codegen.Assignment(target, codegen.Name(module.scope.reserve_name("vals"), s)))
    assert_code_equal(
        module,
        """
        a, (b, c) = vals
        """,
    )


def test_assign_tuple_unpack_rejects_type_hint():
    module = codegen.Module()
    x = module.scope.reserve_name("x")
    y = module.scope.reserve_name("y")
    target = (codegen.Name(x, module.scope), codegen.Name(y, module.scope))
    with pytest.raises(AssertionError, match="Type hints are only supported"):
        codegen.Assignment(target, codegen.Number(1), type_hint=module.scope.name("str"))


def test_assign_tuple_unpack_has_assignment_for_name():
    module = codegen.Module()
    x = module.scope.reserve_name("x")
    y = module.scope.reserve_name("y")
    target = (codegen.Name(x, module.scope), codegen.Name(y, module.scope))
    stmt = codegen.Assignment(target, codegen.Number(1))
    assert stmt.has_assignment_for_name("x")
    assert stmt.has_assignment_for_name("y")
    assert not stmt.has_assignment_for_name("z")


def test_assign_target_rejects_arbitrary_expression():
    target = codegen.String("not a target")
    with pytest.raises((AssertionError, TypeError)):
        codegen.Assignment(target, codegen.Number(1))  # type: ignore[arg-type]


def test_assign_tuple_unpack_rejects_non_target_element():
    with pytest.raises(AssertionError, match="Invalid assignment target"):
        codegen.Assignment((codegen.String("bad"),), codegen.Number(1))  # type: ignore[arg-type]


# --- Annotation tests ---


def test_create_annotation():
    module = codegen.Module()
    name = module.create_annotation("x", module.scope.name("int"))
    assert_code_equal(
        module,
        """
        x: int
        """,
    )
    assert_code_equal(name, "x")


def test_create_annotation_reserves_name():
    module = codegen.Module()
    module.create_annotation("x", module.scope.name("int"))
    assert module.scope.is_name_in_use("x")


def test_create_annotation_duplicate_name_gets_renamed():
    module = codegen.Module()
    module.create_annotation("x", module.scope.name("int"))
    name2 = module.create_annotation("x", module.scope.name("str"))
    assert_code_equal(name2, "x_2")


def test_create_annotation_bad_name():
    module = codegen.Module()
    # Force a bad name through by directly constructing _Annotation
    from fluent_codegen.codegen import Annotation

    stmt = Annotation("bad name", module.scope.name("int"))
    with pytest.raises(AssertionError):
        as_source_code(stmt)


def test_annotation_has_assignment_for_name():
    from fluent_codegen.codegen import Annotation

    scope = codegen.Scope()
    scope.reserve_name("int")
    stmt = Annotation("x", scope.name("int"))
    assert stmt.has_assignment_for_name("x") is True
    assert stmt.has_assignment_for_name("y") is False


# --- Field tests ---


def test_create_field_no_default():
    module = codegen.Module()
    name = module.create_field("x", module.scope.name("int"))
    assert_code_equal(
        module,
        """
        x: int
        """,
    )
    assert_code_equal(name, "x")


def test_create_field_with_default():
    module = codegen.Module()
    name = module.create_field("x", module.scope.name("int"), default=codegen.Number(42))
    assert_code_equal(
        module,
        """
        x: int = 42
        """,
    )
    assert_code_equal(name, "x")


def test_create_field_dataclass():
    """Test a full dataclass with mixed fields."""
    module = codegen.Module()
    module.scope.reserve_name("dataclass")
    cls, _ = module.create_class("Point", decorators=[module.scope.name("dataclass")])
    cls.body.create_field("x", module.scope.name("float"))
    cls.body.create_field("y", module.scope.name("float"))
    cls.body.create_field("label", module.scope.name("str"), default=codegen.String(""))
    assert_code_equal(
        module,
        """
        @dataclass
        class Point:
            x: float
            y: float
            label: str = ''
        """,
    )


def test_create_field_reserves_name():
    module = codegen.Module()
    module.create_field("x", module.scope.name("int"))
    assert module.scope.is_name_in_use("x")


def test_create_field_duplicate_name_gets_renamed():
    module = codegen.Module()
    module.create_field("x", module.scope.name("int"))
    name2 = module.create_field("x", module.scope.name("str"))
    assert_code_equal(name2, "x_2")


# --- Function call tests ---


def test_function_call_unknown():
    scope = codegen.Scope()
    with pytest.raises(AssertionError):
        codegen.function_call("a_function", [], {}, scope)


def test_function_call_known():
    module = codegen.Module()
    module.scope.create_name("a_function")
    func_call = codegen.function_call("a_function", [], {}, module.scope)
    assert_code_equal(func_call, "a_function()")


def test_function_call_args_and_kwargs():
    module = codegen.Module()
    module.scope.create_name("a_function")
    func_call = codegen.function_call(
        "a_function",
        [codegen.Number(123)],
        {"x": codegen.String("hello")},
        module.scope,
    )
    assert_code_equal(func_call, "a_function(123, x='hello')")


def test_function_call_bad_kwarg_names():
    module = codegen.Module()
    module.scope.create_name("a_function")
    allowed_args = [
        ("hyphen-ated", True),
        ("class", True),
        ("True", True),
        (" pre_space", False),
        ("post_space ", False),
        ("mid space", False),
        ("valid_arg", True),
    ]
    for arg_name, allowed in allowed_args:
        func_call = codegen.function_call("a_function", [], {arg_name: codegen.String("a")}, module.scope)
        if allowed:
            output = as_source_code(func_call)
            assert output != ""
            if not allowable_name(arg_name):
                assert "**{" in output
        else:
            with pytest.raises(AssertionError):
                as_source_code(func_call)


def test_function_call_kwarg_star_syntax():
    module = codegen.Module()
    module.scope.create_name("a_function")
    func_call = codegen.function_call("a_function", [], {"hyphen-ated": codegen.Number(1)}, module.scope)
    assert_code_equal(
        func_call,
        """
        a_function(**{'hyphen-ated': 1})
        """,
    )


def test_function_call_sensitive():
    module = codegen.Module()
    module.scope.create_name("exec")
    with pytest.raises(AssertionError):
        codegen.function_call("exec", [], {}, module.scope)


def test_method_call_bad_name():
    s = codegen.String("x")
    with pytest.raises(AssertionError):
        codegen.method_call(s, "bad method name", [], {})


def test_method_call_chained_name():
    s = codegen.String("x")
    call = s.method_call("startswith", [codegen.String("y")], {})
    assert_code_equal(call, "'x'.startswith('y')")


# --- Try/Catch tests ---


def test_try_catch():
    scope = codegen.Scope()
    my_error = scope.create_name("MyError")
    try_ = codegen.Try([my_error], scope)
    assert_code_equal(
        try_,
        """
        try:
            pass
        except MyError:
            pass
        """,
    )
    scope.reserve_name("x")
    scope.reserve_name("y")
    scope.reserve_name("z")
    try_.try_block.create_assignment("x", codegen.String("x"))
    try_.except_block.create_assignment("y", codegen.String("y"))
    try_.else_block.create_assignment("z", codegen.String("z"))
    assert_code_equal(
        try_,
        """
        try:
            x = 'x'
        except MyError:
            y = 'y'
        else:
            z = 'z'
        """,
    )


def test_try_catch_multiple_exceptions():
    scope = codegen.Scope()
    my_error = scope.create_name("MyError")
    other_error = scope.create_name("OtherError")
    try_ = codegen.Try([my_error, other_error], scope)
    assert_code_equal(
        try_,
        """
        try:
            pass
        except (MyError, OtherError):
            pass
        """,
    )


def test_try_catch_has_assignment_for_name_1():
    scope = codegen.Scope()
    try_ = codegen.Try([], scope)
    name = scope.reserve_name("foo")
    assert not try_.has_assignment_for_name(name)

    try_.try_block.create_assignment(name, codegen.String("x"))
    assert not try_.has_assignment_for_name(name)

    try_.except_block.create_assignment(name, codegen.String("x"), allow_multiple=True)
    assert try_.has_assignment_for_name(name)


def test_try_catch_has_assignment_for_name_2():
    scope = codegen.Scope()
    try_ = codegen.Try([], scope)
    name = scope.reserve_name("foo")

    try_.except_block.create_assignment(name, codegen.String("x"))
    assert not try_.has_assignment_for_name(name)

    try_.else_block.create_assignment(name, codegen.String("x"), allow_multiple=True)
    assert try_.has_assignment_for_name(name)


# --- If tests ---


def test_if_empty():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    if_statement = if_statement.finalize()
    assert_code_equal(if_statement, "")


def test_if_one_if():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    first_block = if_statement.create_if_branch(codegen.Number(1))
    first_block.create_return(codegen.Number(2))
    assert_code_equal(
        if_statement,
        """
        if 1:
            return 2
        """,
    )


def test_if_two_ifs():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    first_block = if_statement.create_if_branch(codegen.Number(1))
    first_block.create_return(codegen.Number(2))
    second_block = if_statement.create_if_branch(codegen.Number(3))
    second_block.create_return(codegen.Number(4))
    assert_code_equal(
        if_statement,
        """
        if 1:
            return 2
        elif 3:
            return 4
        """,
    )


def test_if_with_else():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    first_block = if_statement.create_if_branch(codegen.Number(1))
    first_block.create_return(codegen.Number(2))
    if_statement.else_block.create_return(codegen.Number(3))
    assert_code_equal(
        if_statement,
        """
        if 1:
            return 2
        else:
            return 3
        """,
    )


def test_if_no_ifs():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    if_statement.else_block.create_return(codegen.Number(3))
    if_statement = if_statement.finalize()
    assert_code_equal(
        if_statement,
        """
        return 3
        """,
    )


def test_if_scope():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    func.reserve_name("myvalue")
    if_statement = codegen.If(parent_scope=func, parent_block=func.body)
    if_block = if_statement.create_if_branch(codegen.constants.True_)
    assert if_block.scope.is_name_in_use("myvalue")

    name_in_if_block = if_block.scope.reserve_name("myvalue")
    assert name_in_if_block == "myvalue_2"


def test_block_create_if():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    if_stmt = func.body.create_if()
    if_block = if_stmt.create_if_branch(codegen.constants.True_)
    if_block.create_return(codegen.Number(1))
    if_stmt.else_block.create_return(codegen.Number(2))
    assert_code_equal(
        module,
        """
        def myfunc():
            if True:
                return 1
            else:
                return 2
        """,
    )


def test_block_create_if_scope():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    func.reserve_name("myvalue")
    if_stmt = func.body.create_if()
    if_block = if_stmt.create_if_branch(codegen.constants.True_)
    assert if_block.scope.is_name_in_use("myvalue")


def test_block_create_if_parent_block():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    if_stmt = func.body.create_if()
    if_block = if_stmt.create_if_branch(codegen.constants.True_)
    assert if_block.parent_block is func.body


# --- With tests ---


def test_with_no_target():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    with_stmt = func.body.create_with(codegen.String("ctx"))
    with_stmt.body.create_return(codegen.Number(1))
    assert_code_equal(
        module,
        """
        def myfunc():
            with 'ctx':
                return 1
        """,
    )


def test_with_target():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    name = func.create_name("f")
    with_stmt = func.body.create_with(codegen.String("ctx"), name)
    with_stmt.body.create_return(name)
    assert_code_equal(
        module,
        """
        def myfunc():
            with 'ctx' as f:
                return f
        """,
    )


def test_with_scope():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    func.reserve_name("myvalue")
    with_stmt = func.body.create_with(codegen.String("ctx"))
    assert with_stmt.body.scope.is_name_in_use("myvalue")


def test_with_parent_block():
    module = codegen.Module()
    func, _ = module.create_function("myfunc", [])
    with_stmt = func.body.create_with(codegen.String("ctx"))
    assert with_stmt.body.parent_block is func.body


def test_with_standalone():
    module = codegen.Module()
    name = module.scope.reserve_name("ctx_mgr")
    with_stmt = codegen.With(
        codegen.Name(name, module.scope),
        target=None,
        parent_scope=module.scope,
    )
    with_stmt.body.create_return(codegen.Number(42))
    assert_code_equal(
        with_stmt,
        """
        with ctx_mgr:
            return 42
        """,
    )


def test_with_empty_body():
    """An empty with body should produce a pass statement."""
    module = codegen.Module()
    with_stmt = codegen.With(
        codegen.String("ctx"),
        parent_scope=module.scope,
    )
    assert_code_equal(
        with_stmt,
        """
        with 'ctx':
            pass
        """,
    )


# --- Import tests ---


def test_import():
    module = codegen.Module()
    import_stmt, name = module.create_import(module="foo")
    assert name.name == "foo"
    assert module.scope.is_name_in_use("foo")
    assert_code_equal(
        import_stmt,
        """
        import foo
        """,
    )


def test_import_as():
    module = codegen.Module()
    import_stmt, name = module.create_import(module="foo", as_="bar")
    assert name.name == "bar"
    assert not module.scope.is_name_in_use("foo")
    assert module.scope.is_name_in_use("bar")
    assert_code_equal(
        import_stmt,
        """
        import foo as bar
        """,
    )


def test_import_as_clashing():
    module = codegen.Module()
    _, name1 = module.create_import(module="foo", as_="bar")
    assert name1.name == "bar"
    with pytest.raises(AssertionError):
        module.create_import(module="baz", as_="bar")


def test_import_illegal_module_name():
    module = codegen.Module()
    with pytest.raises(AssertionError):
        module.create_import(module="foo asdasd")
    with pytest.raises(AssertionError):
        module.create_import(module="foo. asdasd")

    # Builtin
    with pytest.raises(AssertionError):
        module.create_import(module="str")


def test_import_illegal_as_name():
    module = codegen.Module()
    with pytest.raises(AssertionError):
        module.create_import(module="foo", as_="bar.baz")


def test_import_dotted():
    module = codegen.Module()
    import_stmt, _ = module.create_import(module="foo.bar")
    assert module.scope.is_name_in_use("foo")
    assert not module.scope.is_name_in_use("foo.bar")
    assert not module.scope.is_name_in_use("bar")
    assert_code_equal(
        import_stmt,
        """
        import foo.bar
        """,
    )


def test_import_dotted_multiple():
    # You can do multiple imports from the same top level module.
    module = codegen.Module()
    assert not module.scope.is_name_in_use("foo")
    module.create_import(module="foo.bar")
    assert module.scope.is_name_in_use("foo")
    module.create_import(module="foo.baz")
    assert module.scope.is_name_in_use("foo")
    assert_code_equal(
        module,
        """
        import foo.bar
        import foo.baz
        """,
    )


def test_import_dotted_as():
    module = codegen.Module()
    import_stmt, _ = module.create_import(module="foo.bar", as_="baz")
    assert not module.scope.is_name_in_use("foo")
    assert not module.scope.is_name_in_use("foo.bar")
    assert not module.scope.is_name_in_use("bar")
    assert module.scope.is_name_in_use("baz")
    assert_code_equal(
        import_stmt,
        """
        import foo.bar as baz
        """,
    )


def test_import_from():
    module = codegen.Module()
    import_stmt, name = module.create_import_from(from_="foo", import_="bar")
    assert name.name == "bar"
    assert not module.scope.is_name_in_use("foo")
    assert module.scope.is_name_in_use("bar")
    assert_code_equal(
        import_stmt,
        """
        from foo import bar
        """,
    )


def test_import_from_as():
    module = codegen.Module()
    import_stmt, name = module.create_import_from(from_="foo", import_="bar", as_="baz")
    assert name.name == "baz"
    assert not module.scope.is_name_in_use("foo")
    assert module.scope.is_name_in_use("baz")
    assert_code_equal(
        import_stmt,
        """
        from foo import bar as baz
        """,
    )


def test_import_from_name_clash():
    module = codegen.Module()
    _, name = module.create_import_from(from_="foo", import_="bar")
    assert name.name == "bar"
    with pytest.raises(AssertionError):
        module.create_import_from(from_="quz", import_="bar")


def test_import_from_as_name_clash():
    module = codegen.Module()
    _, name = module.create_import_from(from_="foo", import_="bar", as_="baz")
    assert name.name == "baz"
    with pytest.raises(AssertionError):
        module.create_import_from(from_="quz", import_="bar", as_="baz")


def test_import_from_bad_as_name():
    module = codegen.Module()
    with pytest.raises(AssertionError):
        module.create_import_from(from_="foo", import_="bar", as_="bad name")


def test_import_from_bad_from_name():
    module = codegen.Module()
    with pytest.raises(AssertionError):
        module.create_import_from(from_="bad name", import_="bar")


# --- Expression tests ---


@given(text())
def test_string(t):
    assert t == eval(as_source_code(codegen.String(t))), f" for t = {t!r}"


def test_string_join_empty():
    join = codegen.StringJoin.build([])
    assert_code_equal(join, "''")


def test_string_join_one():
    join = codegen.StringJoin.build([codegen.String("hello")])
    assert_code_equal(join, "'hello'")


def test_concat_string_join_two():
    module = codegen.Module()
    module.scope.reserve_name("tmp")
    var = module.scope.name("tmp")
    join = codegen.ConcatJoin([codegen.String("hello "), var])
    assert_code_equal(join, "'hello ' + tmp")


def test_f_string_join_two():
    module = codegen.Module()
    module.scope.reserve_name("tmp")
    var = module.scope.name("tmp")
    join = codegen.FStringJoin([codegen.String("hello "), var])
    assert_code_equal(join, "f'hello {tmp}'")


def test_string_join_collapse_strings():
    scope = codegen.Scope()
    scope.reserve_name("tmp")
    var = scope.name("tmp")
    join1 = codegen.ConcatJoin.build(
        [
            codegen.String("hello "),
            codegen.String("there "),
            var,
            codegen.String(" how"),
            codegen.String(" are you?"),
        ]
    )
    assert_code_equal(join1, "'hello there ' + tmp + ' how are you?'")


def test_subscript():
    scope = codegen.Scope()
    var = scope.create_name("tmp")
    lookup = var.subscript(codegen.String("x"))
    assert_code_equal(lookup, "tmp['x']")


def test_equals():
    eq = codegen.Equals(codegen.String("x"), codegen.String("y"))
    assert_code_equal(eq, "'x' == 'y'")


def test_or():
    or_ = codegen.Or(codegen.String("x"), codegen.String("y"))
    assert_code_equal(or_, "'x' or 'y'")


def test_attr():
    scope = codegen.Scope()
    name = scope.create_name("foo")
    attr = name.attr("bar")
    assert_code_equal(attr, "foo.bar")


def test_attr_bad_name():
    scope = codegen.Scope()
    name = scope.create_name("foo")
    with pytest.raises(AssertionError):
        name.attr("a bar")


# --- cleanup_name tests ---


def test_cleanup_name():
    cases = [
        ("abc-def()[]ghi,.<>\u00a1!?\u00bf", "abcdefghi"),  # illegal chars
        ("1abc", "n1abc"),  # leading digit
        ("_allowed", "_allowed"),  # leading _ (which is allowed)
        ("-", "n"),  # empty after removing illegals
    ]
    for name, expected in cases:
        assert codegen.cleanup_name(name) == expected


@given(text())
def test_cleanup_name_not_empty(t):
    assert len(codegen.cleanup_name(t)) > 0, f" for t = {t!r}"


@given(non_builtin_text)
@example("!$abc<>")
@example(":id")
def test_cleanup_name_allowed_identifier(t):
    cleaned = codegen.cleanup_name(t)
    assert allowable_name(cleaned) or (cleaned in dir(builtins)) or keyword.iskeyword(cleaned), f" for t = {t!r}"


# --- Additional coverage tests ---


def test_reserve_name_keyword_avoidance():
    scope = codegen.Scope()
    # 'class' is a keyword, so reserve_name should avoid it
    name = scope.reserve_name("class")
    assert name == "class_2"  # skips 'class' because it's a keyword


def test_block_add_statement_bare_expression():
    """Test that bare expressions (e.g. method calls) get wrapped in Expr."""
    module = codegen.Module()
    module.scope.create_name("a_function")
    func_call = codegen.function_call("a_function", [], {}, module.scope)
    module.add_statement(func_call)
    assert_code_equal(module, "a_function()")


def test_block_add_statement_reassign_parent():
    """Test that adding a block with a different parent raises."""
    module1 = codegen.Module()
    module2 = codegen.Module()
    scope = codegen.Scope()
    block = codegen.Block(scope, parent_block=module1)
    with pytest.raises(AssertionError):
        module2.add_statement(block)


def test_block_has_assignment_from_parent():
    """Test has_assignment_for_name delegation to parent block."""
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_assignment(name, codegen.String("hello"))
    child = codegen.Block(module.scope, parent_block=module)
    assert child.has_assignment_for_name(name)


def test_return_repr():
    ret = codegen.Return(codegen.String("hello"))
    assert "Return" in repr(ret)


def test_string_repr():
    s = codegen.String("hello")
    assert repr(s) == "String('hello')"


def test_string_equality():
    assert codegen.String("a") == codegen.String("a")
    assert codegen.String("a") != codegen.String("b")
    assert codegen.String("a") != "a"


def test_number_repr():
    n = codegen.Number(42)
    assert repr(n) == "Number(42)"


def test_list_expression():
    lst = codegen.List([codegen.Number(1), codegen.String("two")])
    assert_code_equal(lst, "[1, 'two']")


def test_dict_expression():
    d = codegen.Dict([(codegen.String("a"), codegen.Number(1))])
    assert_code_equal(d, "{'a': 1}")


def test_none_expr():
    n = codegen.NoneExpr()
    assert_code_equal(n, "None")


def test_function_call_repr():
    module = codegen.Module()
    module.scope.create_name("f")
    fc = codegen.function_call("f", [], {}, module.scope)
    assert "Call" in repr(fc)


def test_name_repr():
    scope = codegen.Scope()
    ref = scope.create_name("x")
    assert repr(ref) == "Name('x')"


def test_name_equality():
    scope = codegen.Scope()
    ref1 = scope.create_name("x")
    ref2 = scope.name("x")
    assert ref1 == ref2
    assert ref1 != "x"


def test_string_join_repr():
    scope = codegen.Scope()
    scope.reserve_name("tmp")
    var = scope.name("tmp")
    join = codegen.ConcatJoin([codegen.String("hello "), var])
    assert "ConcatJoin" in repr(join)


def test_if_finalize_returns_self():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    if_statement.create_if_branch(codegen.Number(1))
    result = if_statement.finalize()
    assert result is if_statement


def test_if_unfinalized_as_ast_raises():
    """If with no if blocks should raise when calling as_ast without finalize."""
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    with pytest.raises(AssertionError, match="finalize"):
        if_statement.as_ast()


def test_traverse():
    """Test traverse function on Python AST nodes."""
    import ast

    module = ast.parse("x = 1")
    nodes = []
    codegen.traverse(module, lambda n: nodes.append(type(n).__name__))
    assert "Module" in nodes
    assert "Assign" in nodes


def test_simplify():
    """Test simplify replaces nodes using a simplifier function."""
    scope = codegen.Scope()
    scope.reserve_name("x")
    var = scope.name("x")
    join = codegen.ConcatJoin([codegen.String("a"), codegen.String("b"), var])

    def simplifier(node, changes):
        # Replace ConcatJoin with a simplified version collapsing strings
        if isinstance(node, codegen.ConcatJoin):
            new_parts = []
            for part in node.parts:
                if new_parts and isinstance(new_parts[-1], codegen.String) and isinstance(part, codegen.String):
                    new_parts[-1] = codegen.String(new_parts[-1].string_value + part.string_value)
                    changes.append(True)
                else:
                    new_parts.append(part)
            if changes:
                return codegen.ConcatJoin(new_parts)
        return node

    result = codegen.simplify(join, simplifier)
    assert_code_equal(result, "'ab' + x")


def test_morph_into():
    """Test morph_into changes class and dict."""
    s1 = codegen.String("hello")
    s2 = codegen.Number(42)
    codegen.morph_into(s1, s2)
    assert isinstance(s1, codegen.Number)
    assert s1.number == 42


def test_block_as_ast_list_with_codegen_ast_list():
    """Test that Block handles CodeGenAstList sub-statements."""
    module = codegen.Module()
    scope = codegen.Scope()
    inner_block = codegen.Block(scope)
    name = scope.reserve_name("x")
    inner_block.create_assignment(name, codegen.String("hello"))
    module.add_statement(inner_block)
    source = as_source_code(module)
    assert "x = 'hello'" in source


def test_block_add_statement_sets_parent():
    """Test that add_statement sets parent_block when None."""
    module = codegen.Module()
    scope = codegen.Scope()
    block = codegen.Block(scope)  # parent_block is None
    module.add_statement(block)
    assert block.parent_block is module


def test_rewriting_traverse_replaces_node():
    """Test that rewriting_traverse actually replaces nodes via morph_into."""
    scope = codegen.Scope()
    var = scope.create_name("x")
    join = codegen.ConcatJoin([codegen.String("hello"), var])

    def replace_hello(node):
        if isinstance(node, codegen.String) and node.string_value == "hello":
            return codegen.String("world")
        return node

    codegen.rewriting_traverse(join, replace_hello)
    assert_code_equal(join, "'world' + x")


def test_rewriting_traverse_tuple():
    """Test rewriting_traverse with tuple input."""
    s = codegen.String("hello")
    t = (s,)
    codegen.rewriting_traverse(t, lambda n: n)


def _collect_traversed_types(root):
    """Helper: traverse a tree and return list of (class_name) for every node visited."""
    visited = []

    def visitor(node):
        visited.append(type(node).__name__)
        return node

    codegen.rewriting_traverse(root, visitor)
    return visited


def test_rewriting_traverse_complex_module():
    """Test traversal visits all nodes in a complex module with nested structures."""
    module = codegen.Module()
    scope = module.scope

    # Function with decorators, return type, body containing if/else, assignments, returns
    scope.reserve_name("my_decorator")
    decorator = codegen.Name("my_decorator", scope)
    func, _ = module.create_function(
        "process",
        args=["items", "flag"],
        decorators=[decorator],
        return_type=codegen.Name("str", scope),
    )
    n_items = codegen.Name("items", func)
    n_flag = codegen.Name("flag", func)

    # Assignment with type hint
    result_name = func.body.scope.reserve_name("result")
    func.body.create_assignment(result_name, codegen.String(""), type_hint=codegen.Name("str", func))

    # If/elif/else
    if_stmt = func.body.create_if()
    branch1 = if_stmt.create_if_branch(n_flag.eq(codegen.Bool(True)))
    branch1.create_assignment(
        result_name,
        codegen.function_call("str", [n_items], {}, func),
        allow_multiple=True,
    )
    branch2 = if_stmt.create_if_branch(n_flag.eq(codegen.Bool(False)))
    branch2.create_return(codegen.String("none"))
    if_stmt.else_block.create_return(codegen.String("default"))

    # Return
    func.body.create_return(codegen.Name("result", func))

    visited = _collect_traversed_types(module)

    # Verify key node types are all visited
    assert "Module" in visited
    assert "Function" in visited
    assert "Block" in visited
    assert "If" in visited
    assert "Equals" in visited
    assert "String" in visited
    assert "Bool" in visited
    assert "Return" in visited
    assert "Call" in visited
    assert "Name" in visited


def test_rewriting_traverse_call_with_kwargs():
    """Test that kwargs dict values in Call nodes are traversed."""
    scope = codegen.Scope()
    func_name = scope.create_name("my_func")
    call = codegen.Call(
        func_name,
        [codegen.String("positional")],
        {"key1": codegen.String("val1"), "key2": codegen.Number(42)},
    )

    visited = _collect_traversed_types(call)
    assert "Call" in visited
    assert "Name" in visited
    # The positional arg
    assert visited.count("String") >= 2  # "positional" and "val1"
    assert "Number" in visited  # 42 from kwargs


def test_rewriting_traverse_replaces_in_kwargs():
    """Test that rewriting_traverse can replace nodes inside Call kwargs."""
    scope = codegen.Scope()
    func_name = scope.create_name("f")
    call = codegen.Call(
        func_name,
        [],
        {"k": codegen.String("old")},
    )

    def replace_old(node):
        if isinstance(node, codegen.String) and node.string_value == "old":
            return codegen.String("new")
        return node

    codegen.rewriting_traverse(call, replace_old)
    assert_code_equal(call, "f(k='new')")


def test_rewriting_traverse_function_decorators_and_return_type():
    """Test that decorators and return_type on Function are traversed."""
    scope = codegen.Scope()
    decorator = codegen.String("deco_marker")
    ret_type = codegen.String("ret_marker")
    func = codegen.Function("myfunc", parent_scope=scope, decorators=[decorator], return_type=ret_type)

    visited = _collect_traversed_types(func)
    assert visited.count("String") == 2  # decorator + return_type


def test_rewriting_traverse_replaces_in_decorator():
    """Test replacing a decorator expression via traverse."""
    scope = codegen.Scope()
    scope.reserve_name("old_deco")
    scope.reserve_name("new_deco")
    func = codegen.Function(
        "myfunc",
        parent_scope=scope,
        decorators=[codegen.Name("old_deco", scope)],
    )

    def replace_deco(node):
        if isinstance(node, codegen.Name) and node.name == "old_deco":
            return codegen.Name("new_deco", scope)
        return node

    codegen.rewriting_traverse(func, replace_deco)
    decorator = func.decorators[0]
    assert isinstance(decorator, codegen.Name)
    assert decorator.name == "new_deco"


def test_rewriting_traverse_assignment_type_hint():
    """Test that type_hint on _Assignment is traversed."""
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_assignment(name, codegen.String("hello"), type_hint=codegen.Name("str", module.scope))

    visited = _collect_traversed_types(module)
    # The type hint Name("str") should be visited
    assert visited.count("Name") >= 1


def test_rewriting_traverse_annotation():
    """Test that annotation Expression on _Annotation is traversed."""
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.create_annotation(name, codegen.Name("int", module.scope))

    visited = _collect_traversed_types(module)
    # Module -> Block(statements) -> _Annotation -> annotation(Name)
    assert "Name" in visited


def test_rewriting_traverse_with_statement():
    """Test traversal into With statement's context_expr, target, and body."""
    module = codegen.Module()
    module.scope.reserve_name("ctx_manager")
    module.scope.reserve_name("f")
    ctx = codegen.function_call("ctx_manager", [codegen.String("file.txt")], {}, module.scope)
    with_stmt = module.create_with(ctx, target=codegen.Name("f", module.scope))
    with_stmt.body.create_return(codegen.String("done"))

    visited = _collect_traversed_types(module)
    assert "With" in visited
    assert "Call" in visited
    assert "Return" in visited


def test_rewriting_traverse_try_except():
    """Test traversal into Try blocks."""
    module = codegen.Module()
    exc = codegen.Name("ValueError", module.scope)
    try_stmt = codegen.Try([exc], module.scope)
    try_stmt.try_block.create_return(codegen.String("tried"))
    try_stmt.except_block.create_return(codegen.String("caught"))
    try_stmt.else_block.create_return(codegen.String("else"))
    module.add_statement(try_stmt)

    visited = _collect_traversed_types(module)
    assert "Try" in visited
    assert visited.count("Block") >= 3  # try, except, else blocks
    assert visited.count("Return") >= 3
    assert visited.count("String") >= 3


def test_rewriting_traverse_dict_expression():
    """Test traversal into Dict pairs (tuples of expressions)."""
    d = codegen.Dict(
        [
            (codegen.String("k1"), codegen.Number(1)),
            (codegen.String("k2"), codegen.Number(2)),
        ]
    )

    visited = _collect_traversed_types(d)
    assert visited.count("String") == 2
    assert visited.count("Number") == 2


def test_rewriting_traverse_nested_binary_ops():
    """Test deep nesting of binary operators."""
    # (1 + 2) * (3 - 4)
    expr = codegen.Number(1).add(codegen.Number(2)).mul(codegen.Number(3).sub(codegen.Number(4)))

    visited = _collect_traversed_types(expr)
    assert visited.count("Number") == 4
    assert "Mul" in visited
    assert "Add" in visited
    assert "Sub" in visited


def test_rewriting_traverse_subscript():
    """Test traversal into Subscript value and slice."""
    scope = codegen.Scope()
    scope.reserve_name("items")
    expr = codegen.Name("items", scope).subscript(codegen.Number(0))

    visited = _collect_traversed_types(expr)
    assert "Subscript" in visited
    assert "Name" in visited
    assert "Number" in visited


def test_rewriting_traverse_attr():
    """Test traversal into Attr value."""
    scope = codegen.Scope()
    scope.reserve_name("obj")
    expr = codegen.Name("obj", scope).attr("prop")

    visited = _collect_traversed_types(expr)
    assert "Attr" in visited
    assert "Name" in visited


def test_rewriting_traverse_starred():
    """Test traversal into Starred value."""
    scope = codegen.Scope()
    scope.reserve_name("args")
    expr = codegen.Name("args", scope).starred()

    visited = _collect_traversed_types(expr)
    assert "Starred" in visited
    assert "Name" in visited


def test_rewriting_traverse_list_tuple_set():
    """Test traversal into List, Tuple, Set items."""
    for cls in [codegen.List, codegen.Tuple, codegen.Set]:
        coll = cls([codegen.String("a"), codegen.Number(1)])
        visited = _collect_traversed_types(coll)
        assert "String" in visited
        assert "Number" in visited


def test_rewriting_traverse_class_with_bases_and_decorators():
    """Test traversal into Class body, bases, and decorators."""
    module = codegen.Module()
    module.scope.reserve_name("BaseClass")
    module.scope.reserve_name("my_class_deco")
    base = codegen.Name("BaseClass", module.scope)
    deco = codegen.Name("my_class_deco", module.scope)
    cls, _ = module.create_class("MyClass", bases=[base], decorators=[deco])

    visited = _collect_traversed_types(module)
    assert "Class" in visited
    assert "Block" in visited
    # bases and decorators should be visited
    assert visited.count("Name") >= 2


def test_rewriting_traverse_assert():
    """Test traversal into Assert test and msg."""
    module = codegen.Module()
    module.create_assert(codegen.Bool(True), codegen.String("oops"))

    visited = _collect_traversed_types(module)
    assert "Assert" in visited
    assert "Bool" in visited
    assert "String" in visited


def test_rewriting_traverse_fstring():
    """Test traversal into FStringJoin parts."""
    scope = codegen.Scope()
    scope.reserve_name("world")
    fstr = codegen.FStringJoin([codegen.String("hello "), codegen.Name("world", scope)])

    visited = _collect_traversed_types(fstr)
    assert "FStringJoin" in visited
    assert "String" in visited
    assert "Name" in visited


def test_rewriting_traverse_deeply_nested_codegen_tree():
    """
    Build a complex, deeply-nested codegen tree and verify that rewriting_traverse
    visits every single CodeGenAst/CodeGenAstList node by replacing all String
    values with 'REPLACED'.
    """
    module = codegen.Module()
    scope = module.scope

    # Import
    module.create_import("os")
    module.create_import_from(from_="sys", import_="argv")

    # Class with decorator and base
    scope.reserve_name("dataclass")
    class_deco = codegen.Name("dataclass", scope)
    base = codegen.Name("object", scope)
    cls, _ = module.create_class("Config", bases=[base], decorators=[class_deco])
    field_name = cls.body.scope.reserve_name("name")
    cls.body.create_assignment(field_name, codegen.String("default_name"))

    # Function with kwargs call, nested if, try/except, with, assert
    scope.reserve_name("process")
    func, _ = module.create_function(
        "run",
        args=["config"],
        return_type=codegen.Name("bool", scope),
    )
    config = codegen.Name("config", func)

    # Assignment with function call using kwargs
    result_name = func.body.scope.reserve_name("result")
    func.body.create_assignment(
        result_name,
        codegen.function_call(
            "process",
            [config.attr("name")],
            {"timeout": codegen.Number(30)},
            func,
        ),
    )

    # Nested if
    if_stmt = func.body.create_if()
    branch = if_stmt.create_if_branch(codegen.Name("result", func).eq(codegen.String("ok")))
    branch.create_return(codegen.Bool(True))

    # Try/except in else
    try_stmt = codegen.Try([codegen.Name("Exception", func)], func)
    try_stmt.try_block.create_return(codegen.Bool(False))
    try_stmt.except_block.create_return(codegen.Bool(False))
    if_stmt.else_block.add_statement(try_stmt)

    # With statement
    func.reserve_name("open_log")
    ctx_expr = codegen.function_call("open_log", [codegen.String("log.txt")], {}, func)
    with_block = func.body.create_with(ctx_expr)
    with_block.body.create_return(codegen.String("logged"))

    # Assert
    func.body.create_assert(
        codegen.Name("result", func).ne(codegen.NoneExpr()),
        codegen.String("must not be None"),
    )

    # Now do a rewriting_traverse that replaces all Strings
    string_count = [0]

    def count_and_replace_strings(node):
        if isinstance(node, codegen.String):
            string_count[0] += 1
            return codegen.String("REPLACED")
        return node

    codegen.rewriting_traverse(module, count_and_replace_strings)

    # We should have found all string literals in the tree
    assert string_count[0] >= 5  # default_name, ok, log.txt, logged, must not be None

    # Verify ALL strings are now REPLACED in the generated source
    source = module.as_python_source()
    for original in ["default_name", "'ok'", "log.txt", "logged", "must not be None"]:
        assert original not in source, f"String '{original}' should have been replaced but found in: {source}"
    assert "REPLACED" in source


def test_rewriting_traverse_replace_number_in_nested_kwargs():
    """
    Specifically test that numbers deep inside kwargs of a Call inside a
    function body can be found and replaced.
    """
    module = codegen.Module()
    module.scope.reserve_name("setup")
    func, _ = module.create_function("f", args=[])
    func.body.create_assignment(
        func.body.scope.reserve_name("x"),
        codegen.function_call(
            "setup",
            [],
            {"timeout": codegen.Number(99), "retries": codegen.Number(3)},
            func,
        ),
    )

    def double_numbers(node):
        if isinstance(node, codegen.Number):
            return codegen.Number(node.number * 2)
        return node

    codegen.rewriting_traverse(module, double_numbers)
    source = module.as_python_source()
    assert "198" in source  # 99 * 2
    assert "6" in source  # 3 * 2
    assert "99" not in source


def test_rewriting_traverse_none_expr():
    """Test that NoneExpr nodes are visited (they previously lacked child_elements)."""
    scope = codegen.Scope()
    scope.reserve_name("x")
    expr = codegen.Name("x", scope).eq(codegen.NoneExpr())

    visited = _collect_traversed_types(expr)
    assert "NoneExpr" in visited
    assert "Equals" in visited
    assert "Name" in visited


def test_rewriting_traverse_import_nodes():
    """Test that Import and ImportFrom nodes are visited (they previously lacked child_elements)."""
    module = codegen.Module()
    module.create_import("os")
    module.create_import_from(from_="sys", import_="argv")

    visited = _collect_traversed_types(module)
    assert "Import" in visited
    assert "ImportFrom" in visited


# --- auto() tests ---


def test_auto_string():
    result = codegen.auto("hello")
    assert isinstance(result, codegen.String)
    assert_code_equal(result, "'hello'")


def test_auto_int():
    result = codegen.auto(42)
    assert isinstance(result, codegen.Number)
    assert_code_equal(result, "42")


def test_auto_float():
    result = codegen.auto(3.14)
    assert isinstance(result, codegen.Number)
    assert_code_equal(result, "3.14")


def test_auto_none():
    result = codegen.auto(None)
    assert isinstance(result, codegen.NoneExpr)
    assert_code_equal(result, "None")


def test_auto_list():
    result = codegen.auto([1, "two", 3.0])
    assert isinstance(result, codegen.List)
    assert_code_equal(result, "[1, 'two', 3.0]")


def test_auto_list_empty():
    result = codegen.auto([])
    assert isinstance(result, codegen.List)
    assert_code_equal(result, "[]")


def test_auto_dict():
    result = codegen.auto({"a": 1})
    assert isinstance(result, codegen.Dict)
    assert_code_equal(result, "{'a': 1}")


def test_auto_dict_empty():
    result = codegen.auto({})
    assert isinstance(result, codegen.Dict)
    assert_code_equal(result, "{}")


def test_auto_nested():
    result = codegen.auto({"items": [1, 2], "name": "test"})
    assert isinstance(result, codegen.Dict)
    source = as_source_code(result)
    assert "'items'" in source
    assert "[1, 2]" in source
    assert "'name'" in source
    assert "'test'" in source


def test_auto_unsupported_type():
    with pytest.raises(AssertionError):
        codegen.auto(object())  # type: ignore[arg-type]


# --- Bool tests ---


def test_bool_true():
    b = codegen.Bool(True)
    assert_code_equal(b, "True")


def test_bool_false():
    b = codegen.Bool(False)
    assert_code_equal(b, "False")


def test_bool_repr():
    assert repr(codegen.Bool(True)) == "Bool(True)"
    assert repr(codegen.Bool(False)) == "Bool(False)"


def test_auto_bool_true():
    result = codegen.auto(True)
    assert isinstance(result, codegen.Bool)
    assert_code_equal(result, "True")


def test_auto_bool_false():
    result = codegen.auto(False)
    assert isinstance(result, codegen.Bool)
    assert_code_equal(result, "False")


# --- Bytes tests ---


def test_bytes():
    b = codegen.Bytes(b"hello")
    assert_code_equal(b, "b'hello'")


def test_bytes_empty():
    b = codegen.Bytes(b"")
    assert_code_equal(b, "b''")


def test_bytes_repr():
    assert repr(codegen.Bytes(b"hi")) == "Bytes(b'hi')"


def test_auto_bytes():
    result = codegen.auto(b"hello")
    assert isinstance(result, codegen.Bytes)
    assert_code_equal(result, "b'hello'")


# --- Tuple tests ---


def test_tuple():
    t = codegen.Tuple([codegen.Number(1), codegen.String("two")])
    assert_code_equal(t, "(1, 'two')")


def test_tuple_empty():
    t = codegen.Tuple([])
    assert_code_equal(t, "()")


def test_tuple_single():
    t = codegen.Tuple([codegen.Number(1)])
    assert_code_equal(t, "(1,)")


def test_auto_tuple():
    result = codegen.auto((1, "two", None))
    assert isinstance(result, codegen.Tuple)
    assert_code_equal(result, "(1, 'two', None)")


def test_auto_tuple_empty():
    result = codegen.auto(())
    assert isinstance(result, codegen.Tuple)
    assert_code_equal(result, "()")


# --- Set tests ---


def test_set():
    s = codegen.Set([codegen.Number(1), codegen.Number(2)])
    assert_code_equal(s, "{1, 2}")


def test_set_single():
    s = codegen.Set([codegen.String("a")])
    assert_code_equal(s, "{'a'}")


def test_set_empty():
    """Empty set literal is not valid Python syntax ({} is a dict), so we emit set([])."""
    s = codegen.Set([])
    assert_code_equal(s, "set([])")


def test_auto_set():
    # Sets are unordered, so we test membership rather than exact output
    result = codegen.auto({1, 2})
    assert isinstance(result, codegen.Set)


def test_auto_frozenset():
    result = codegen.auto(frozenset({1}))
    assert isinstance(result, codegen.Set)


# --- Arithmetic operator tests, Comparison operator tests, Boolean operator tests

# The classes are tested indirectly by the utilty methods on `Expression`,
# we don't really need separate tests.


# --- Expression utility method tests ---


def test_expression_add_method():
    result = codegen.Number(1).add(codegen.Number(2))
    assert isinstance(result, codegen.Add)
    assert_code_equal(result, "1 + 2")


def test_expression_sub_method():
    result = codegen.Number(5).sub(codegen.Number(3))
    assert isinstance(result, codegen.Sub)
    assert_code_equal(result, "5 - 3")


def test_expression_mul_method():
    result = codegen.Number(4).mul(codegen.Number(3))
    assert isinstance(result, codegen.Mul)
    assert_code_equal(result, "4 * 3")


def test_expression_div_method():
    result = codegen.Number(10).div(codegen.Number(3))
    assert isinstance(result, codegen.Div)
    assert_code_equal(result, "10 / 3")


def test_expression_floordiv_method():
    result = codegen.Number(10).floordiv(codegen.Number(3))
    assert isinstance(result, codegen.FloorDiv)
    assert_code_equal(result, "10 // 3")


def test_expression_mod_method():
    result = codegen.Number(10).mod(codegen.Number(3))
    assert isinstance(result, codegen.Mod)
    assert_code_equal(result, "10 % 3")


def test_expression_pow_method():
    result = codegen.Number(2).pow(codegen.Number(8))
    assert isinstance(result, codegen.Pow)
    assert_code_equal(result, "2 ** 8")


def test_expression_eq_method():
    result = codegen.String("x").eq(codegen.String("y"))
    assert isinstance(result, codegen.Equals)
    assert_code_equal(result, "'x' == 'y'")


def test_expression_ne_method():
    result = codegen.String("x").ne(codegen.String("y"))
    assert isinstance(result, codegen.NotEquals)
    assert_code_equal(result, "'x' != 'y'")


def test_expression_lt_method():
    result = codegen.Number(1).lt(codegen.Number(2))
    assert isinstance(result, codegen.Lt)
    assert_code_equal(result, "1 < 2")


def test_expression_gt_method():
    result = codegen.Number(2).gt(codegen.Number(1))
    assert isinstance(result, codegen.Gt)
    assert_code_equal(result, "2 > 1")


def test_expression_le_method():
    result = codegen.Number(1).le(codegen.Number(2))
    assert isinstance(result, codegen.LtE)
    assert_code_equal(result, "1 <= 2")


def test_expression_ge_method():
    result = codegen.Number(2).ge(codegen.Number(1))
    assert isinstance(result, codegen.GtE)
    assert_code_equal(result, "2 >= 1")


def test_expression_and_method():
    result = codegen.Bool(True).and_(codegen.Bool(False))
    assert isinstance(result, codegen.And)
    assert_code_equal(result, "True and False")


def test_expression_or_method():
    result = codegen.Bool(True).or_(codegen.Bool(False))
    assert isinstance(result, codegen.Or)
    assert_code_equal(result, "True or False")


def test_expression_in_method():
    result = codegen.String("a").in_(codegen.List([codegen.String("a"), codegen.String("b")]))
    assert isinstance(result, codegen.In)
    assert_code_equal(result, "'a' in ['a', 'b']")


def test_expression_not_in_method():
    result = codegen.String("c").not_in(codegen.List([codegen.String("a"), codegen.String("b")]))
    assert isinstance(result, codegen.NotIn)
    assert_code_equal(result, "'c' not in ['a', 'b']")


def test_expression_matmul_method():
    result = codegen.Scope().create_name("a").matmul(codegen.auto(1))
    assert isinstance(result, codegen.MatMul)
    assert_code_equal(result, "a @ 1")


# --- Starred tests ---


def test_starred_expression():
    scope = codegen.Scope()
    name = scope.create_name("args")
    starred = name.starred()
    assert isinstance(starred, codegen.Starred)
    assert_code_equal(starred, "*args")


def test_starred_in_function_call():
    module = codegen.Module()
    module.scope.create_name("my_func")
    args_name = module.scope.create_name("args")
    call = codegen.function_call("my_func", [args_name.starred()], {}, module.scope)
    assert_code_equal(call, "my_func(*args)")


def test_starred_in_list():
    scope = codegen.Scope()
    name = scope.create_name("items")
    lst = codegen.List([name.starred(), codegen.Number(1)])
    assert_code_equal(lst, "[*items, 1]")


def test_starred_in_tuple():
    scope = codegen.Scope()
    name = scope.create_name("items")
    tup = codegen.Tuple([codegen.Number(0), name.starred()])
    assert_code_equal(tup, "(0, *items)")


def test_starred_repr():
    scope = codegen.Scope()
    name = scope.create_name("x")
    starred = codegen.Starred(name)
    assert repr(starred) == "Starred(Name('x'))"


def test_starred_in_assignment():
    """Starred on the value side, e.g. `a = (*b,)`."""
    module = codegen.Module()
    b = module.scope.create_name("b")
    a = module.scope.create_name("a")
    module.create_assignment(a, codegen.Tuple([b.starred()]))
    assert_code_equal(
        module,
        """
        a = (*b,)
        """,
    )


# --- Chaining tests ---


def test_chained_arithmetic():
    """Test that arithmetic methods chain correctly: (1 + 2) * 3"""
    result = codegen.Number(1).add(codegen.Number(2)).mul(codegen.Number(3))
    assert_code_equal(result, "(1 + 2) * 3")


def test_chained_comparison_with_arithmetic():
    """Test chaining: (x + 1) > 0"""
    scope = codegen.Scope()
    x = scope.create_name("x")
    result = x.add(codegen.Number(1)).gt(codegen.Number(0))
    assert_code_equal(result, "x + 1 > 0")


def test_chained_boolean():
    """Test chaining: (a > 0) and (b < 10)"""
    scope = codegen.Scope()
    a = scope.create_name("a")
    b = scope.create_name("b")
    result = a.gt(codegen.Number(0)).and_(b.lt(codegen.Number(10)))
    assert_code_equal(result, "a > 0 and b < 10")


def test_chained_with_method_call():
    """Test chaining operator methods with .attr() and .call()"""
    scope = codegen.Scope()
    items = scope.create_name("items")
    # len(items) > 0  -- modeled as items.attr('__len__').call([],{}) > 0
    # But more practically: items.method_call("count", ...).gt(Number(0))
    result = items.method_call("count", [codegen.String("x")], {}).gt(codegen.Number(0))
    assert_code_equal(result, "items.count('x') > 0")


def test_call_no_args():
    scope = codegen.Scope()
    name = scope.create_name("foo")
    result = name.call()
    assert_code_equal(result, "foo()")


def test_call_args_only():
    scope = codegen.Scope()
    name = scope.create_name("foo")
    result = name.call([codegen.Number(1)])
    assert_code_equal(result, "foo(1)")


def test_call_kwargs_only():
    scope = codegen.Scope()
    name = scope.create_name("foo")
    result = name.call(kwargs={"x": codegen.Number(1)})
    assert_code_equal(result, "foo(x=1)")


def test_method_call_no_args():
    scope = codegen.Scope()
    name = scope.create_name("obj")
    result = name.method_call("do_thing")
    assert_code_equal(result, "obj.do_thing()")


def test_method_call_args_only():
    scope = codegen.Scope()
    name = scope.create_name("obj")
    result = name.method_call("do_thing", [codegen.String("a")])
    assert_code_equal(result, "obj.do_thing('a')")


# --- FunctionArg tests ---


def test_function_arg_positional_only():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[codegen.FunctionArg.positional("a"), codegen.FunctionArg.positional("b")],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a, b, /):
            pass
        """,
    )


def test_function_arg_keyword_only():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[codegen.FunctionArg.keyword("x"), codegen.FunctionArg.keyword("y")],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(*, x, y):
            pass
        """,
    )


def test_function_arg_all_kinds():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.positional("a"),
            codegen.FunctionArg.standard("b"),
            codegen.FunctionArg.keyword("c"),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a, /, b, *, c):
            pass
        """,
    )


def test_function_arg_with_defaults():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.positional("a"),
            codegen.FunctionArg.positional("b", default=codegen.Number(1)),
            codegen.FunctionArg.standard("c", default=codegen.Number(2)),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a, b=1, /, c=2):
            pass
        """,
    )


def test_function_arg_keyword_defaults():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.keyword("x"),
            codegen.FunctionArg.keyword("y", default=codegen.String("hello")),
            codegen.FunctionArg.keyword("z"),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(*, x, y='hello', z):
            pass
        """,
    )


def test_function_arg_mixed_str_and_functionarg():
    """str args are treated as positional-or-keyword."""
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=["a", codegen.FunctionArg.keyword("b")],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a, *, b):
            pass
        """,
    )


def test_function_arg_bad_order_kw_before_positional():
    module = codegen.Module()
    with pytest.raises(ValueError, match="out of order"):
        codegen.Function(
            "myfunc",
            args=[
                codegen.FunctionArg.keyword("x"),
                codegen.FunctionArg.standard("y"),
            ],
            parent_scope=module.scope,
        )


def test_function_arg_bad_order_standard_before_posonly():
    module = codegen.Module()
    with pytest.raises(ValueError, match="out of order"):
        codegen.Function(
            "myfunc",
            args=[
                codegen.FunctionArg.standard("x"),
                codegen.FunctionArg.positional("y"),
            ],
            parent_scope=module.scope,
        )


def test_function_arg_non_default_after_default():
    module = codegen.Module()
    with pytest.raises(ValueError, match="Non-default argument"):
        codegen.Function(
            "myfunc",
            args=[
                codegen.FunctionArg.standard("a", default=codegen.Number(1)),
                codegen.FunctionArg.standard("b"),
            ],
            parent_scope=module.scope,
        )


def test_function_arg_non_default_posonly_after_default_posonly():
    module = codegen.Module()
    with pytest.raises(ValueError, match="Non-default argument"):
        codegen.Function(
            "myfunc",
            args=[
                codegen.FunctionArg.positional("a", default=codegen.Number(1)),
                codegen.FunctionArg.positional("b"),
            ],
            parent_scope=module.scope,
        )


def test_function_arg_non_default_standard_after_default_posonly():
    """Positional default followed by standard non-default is invalid."""
    module = codegen.Module()
    with pytest.raises(ValueError, match="Non-default argument"):
        codegen.Function(
            "myfunc",
            args=[
                codegen.FunctionArg.positional("a", default=codegen.Number(1)),
                codegen.FunctionArg.standard("b"),
            ],
            parent_scope=module.scope,
        )


def test_function_arg_bad_name():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[codegen.FunctionArg.positional("bad arg")],
        parent_scope=module.scope,
    )
    with pytest.raises(AssertionError):
        as_source_code(func)


def test_function_arg_shadows():
    module = codegen.Module()
    module.scope.reserve_name("x")
    with pytest.raises(AssertionError):
        codegen.Function(
            "myfunc",
            args=[codegen.FunctionArg.keyword("x")],
            parent_scope=module.scope,
        )


def test_function_arg_enum_values():
    assert codegen.ArgKind.POSITIONAL_ONLY.value == "positional_only"
    assert codegen.ArgKind.POSITIONAL_OR_KEYWORD.value == "positional_or_keyword"
    assert codegen.ArgKind.KEYWORD_ONLY.value == "keyword_only"


def test_function_arg_dataclass_frozen():
    arg = codegen.FunctionArg("x")
    with pytest.raises(AttributeError):
        arg.name = "y"  # type: ignore[misc]


def test_function_arg_default_kind():
    arg = codegen.FunctionArg("x")
    assert arg.kind == codegen.ArgKind.POSITIONAL_OR_KEYWORD
    assert arg.default is None


def test_function_with_decorator():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=["x"],
        parent_scope=module.scope,
        decorators=[module.scope.name("staticmethod")],
    )
    assert_code_equal(
        func,
        """
        @staticmethod
        def myfunc(x):
            pass
        """,
    )


def test_function_with_multiple_decorators():
    module = codegen.Module()
    module.scope.reserve_name("my_decorator")
    func = codegen.Function(
        "myfunc",
        args=[],
        parent_scope=module.scope,
        decorators=[
            module.scope.name("my_decorator"),
            module.scope.name("staticmethod"),
        ],
    )
    assert_code_equal(
        func,
        """
        @my_decorator
        @staticmethod
        def myfunc():
            pass
        """,
    )


def test_function_with_decorator_call():
    module = codegen.Module()
    module.scope.reserve_name("my_decorator")
    decorator = codegen.function_call("my_decorator", [codegen.String("arg")], {}, module.scope)
    func = codegen.Function(
        "myfunc",
        args=[],
        parent_scope=module.scope,
        decorators=[decorator],
    )
    assert_code_equal(
        func,
        """
        @my_decorator('arg')
        def myfunc():
            pass
        """,
    )


def test_create_function_with_decorators():
    module = codegen.Module()
    func, func_name = module.create_function(
        "my_func",
        args=["x"],
        decorators=[module.scope.name("staticmethod")],
    )
    assert_code_equal(
        module,
        """
        @staticmethod
        def my_func(x):
            pass
        """,
    )


def test_function_no_decorators_default():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=[], parent_scope=module.scope)
    assert func.decorators == []


def test_function_return_type():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=["x"],
        parent_scope=module.scope,
        return_type=module.scope.name("int"),
    )
    assert_code_equal(
        func,
        """
        def myfunc(x) -> int:
            pass
        """,
    )


def test_function_return_type_none():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[],
        parent_scope=module.scope,
        return_type=codegen.constants.None_,
    )
    assert_code_equal(
        func,
        """
        def myfunc() -> None:
            pass
        """,
    )


def test_function_return_type_string_expr():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[],
        parent_scope=module.scope,
        return_type=codegen.String("MyClass"),
    )
    assert_code_equal(
        func,
        """
        def myfunc() -> 'MyClass':
            pass
        """,
    )


def test_function_no_return_type_default():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=[], parent_scope=module.scope)
    assert func.return_type is None


def test_create_function_with_return_type():
    module = codegen.Module()
    func, func_name = module.create_function(
        "my_func",
        args=["x"],
        return_type=module.scope.name("str"),
    )
    assert_code_equal(
        module,
        """
        def my_func(x) -> str:
            pass
        """,
    )


def test_create_function_with_function_args():
    module = codegen.Module()
    func, func_name = module.create_function(
        "my_func",
        args=[
            codegen.FunctionArg.positional("a"),
            codegen.FunctionArg.keyword("b", default=codegen.Number(42)),
        ],
    )
    assert_code_equal(
        module,
        """
        def my_func(a, /, *, b=42):
            pass
        """,
    )
    assert_code_equal(func_name, "my_func")


def test_function_arg_only_positional_with_default():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[codegen.FunctionArg.positional("a", default=codegen.String("hi"))],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a='hi', /):
            pass
        """,
    )


def test_function_arg_standard_with_default():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("a"),
            codegen.FunctionArg.standard("b", default=codegen.NoneExpr()),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a, b=None):
            pass
        """,
    )


def test_normalize_args_all_strings():
    result = codegen._normalize_args(["a", "b", "c"])
    assert all(isinstance(a, codegen.FunctionArg) for a in result)
    assert all(a.kind == codegen.ArgKind.POSITIONAL_OR_KEYWORD for a in result)
    assert [a.name for a in result] == ["a", "b", "c"]


def test_normalize_args_all_function_args():
    args = [codegen.FunctionArg.positional("a"), codegen.FunctionArg.keyword("b")]
    result = codegen._normalize_args(args)
    assert result == args


def test_function_arg_complex_defaults():
    """Test default values that are complex expressions."""
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("a", default=codegen.List([codegen.Number(1), codegen.Number(2)])),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a=[1, 2]):
            pass
        """,
    )


def test_function_arg_annotation_standard():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("x", annotation=module.scope.name("int")),
            codegen.FunctionArg.standard("y", annotation=module.scope.name("str")),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(x: int, y: str):
            pass
        """,
    )


def test_function_arg_annotation_positional():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.positional("x", annotation=module.scope.name("int")),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(x: int, /):
            pass
        """,
    )


def test_function_arg_annotation_keyword():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.keyword("x", annotation=module.scope.name("int")),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(*, x: int):
            pass
        """,
    )


def test_function_arg_annotation_with_default():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("x", annotation=module.scope.name("int"), default=codegen.Number(0)),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(x: int=0):
            pass
        """,
    )


def test_function_arg_annotation_mixed():
    """Some args annotated, some not."""
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("x", annotation=module.scope.name("int")),
            codegen.FunctionArg.standard("y"),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(x: int, y):
            pass
        """,
    )


def test_function_arg_annotation_none_default():
    arg = codegen.FunctionArg.standard("x")
    assert arg.annotation is None


def test_function_arg_annotation_all_kinds():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.positional("a", annotation=module.scope.name("int")),
            codegen.FunctionArg.standard("b", annotation=module.scope.name("str")),
            codegen.FunctionArg.keyword("c", annotation=module.scope.name("float")),
        ],
        parent_scope=module.scope,
    )
    assert_code_equal(
        func,
        """
        def myfunc(a: int, /, b: str, *, c: float):
            pass
        """,
    )


def test_function_arg_annotation_with_return_type():
    module = codegen.Module()
    func = codegen.Function(
        "myfunc",
        args=[
            codegen.FunctionArg.standard("x", annotation=module.scope.name("int")),
        ],
        parent_scope=module.scope,
        return_type=module.scope.name("str"),
    )
    assert_code_equal(
        func,
        """
        def myfunc(x: int) -> str:
            pass
        """,
    )


# --- Class tests ---


def test_class_empty():
    module = codegen.Module()
    cls = codegen.Class("MyClass", parent_scope=module.scope)
    assert_code_equal(
        cls,
        """
        class MyClass:
            pass
        """,
    )


def test_class_with_base():
    module = codegen.Module()
    module.scope.reserve_name("BaseClass")
    cls = codegen.Class(
        "MyClass",
        parent_scope=module.scope,
        bases=[module.scope.name("BaseClass")],
    )
    assert_code_equal(
        cls,
        """
        class MyClass(BaseClass):
            pass
        """,
    )


def test_class_with_multiple_bases():
    module = codegen.Module()
    module.scope.reserve_name("Base1")
    module.scope.reserve_name("Base2")
    cls = codegen.Class(
        "MyClass",
        parent_scope=module.scope,
        bases=[module.scope.name("Base1"), module.scope.name("Base2")],
    )
    assert_code_equal(
        cls,
        """
        class MyClass(Base1, Base2):
            pass
        """,
    )


def test_class_with_decorator():
    module = codegen.Module()
    module.scope.reserve_name("my_decorator")
    cls = codegen.Class(
        "MyClass",
        parent_scope=module.scope,
        decorators=[module.scope.name("my_decorator")],
    )
    assert_code_equal(
        cls,
        """
        @my_decorator
        class MyClass:
            pass
        """,
    )


def test_class_with_multiple_decorators():
    module = codegen.Module()
    module.scope.reserve_name("deco1")
    module.scope.reserve_name("deco2")
    cls = codegen.Class(
        "MyClass",
        parent_scope=module.scope,
        decorators=[module.scope.name("deco1"), module.scope.name("deco2")],
    )
    assert_code_equal(
        cls,
        """
        @deco1
        @deco2
        class MyClass:
            pass
        """,
    )


def test_class_with_body():
    module = codegen.Module()
    cls = codegen.Class("MyClass", parent_scope=module.scope)
    name = cls.reserve_name("x")
    cls.body.create_assignment(name, codegen.Number(42))
    assert_code_equal(
        cls,
        """
        class MyClass:
            x = 42
        """,
    )


def test_class_with_method():
    module = codegen.Module()
    cls = codegen.Class("MyClass", parent_scope=module.scope)
    func, _ = cls.body.create_function("my_method", args=["self"])
    func.create_return(codegen.Number(1))
    source = as_source_code(cls)
    assert "class MyClass:" in source
    assert "def my_method(self):" in source
    assert "return 1" in source


def test_class_with_decorator_and_base_and_body():
    module = codegen.Module()
    module.scope.reserve_name("dataclass")
    module.scope.reserve_name("Base")
    cls = codegen.Class(
        "MyClass",
        parent_scope=module.scope,
        bases=[module.scope.name("Base")],
        decorators=[module.scope.name("dataclass")],
    )
    name = cls.reserve_name("value")
    cls.body.create_assignment(name, codegen.Number(0))
    source = as_source_code(cls)
    assert "@dataclass" in source
    assert "class MyClass(Base):" in source
    assert "value = 0" in source


def test_class_bad_name():
    module = codegen.Module()
    cls = codegen.Class("bad name", parent_scope=module.scope)
    with pytest.raises(AssertionError):
        as_source_code(cls)


def test_class_defaults():
    module = codegen.Module()
    cls = codegen.Class("MyClass", parent_scope=module.scope)
    assert cls.bases == []
    assert cls.decorators == []


def test_create_class():
    module = codegen.Module()
    cls, cls_name = module.create_class("MyClass")
    assert_code_equal(
        module,
        """
        class MyClass:
            pass
        """,
    )
    assert_code_equal(cls_name, "MyClass")


def test_create_class_with_base():
    module = codegen.Module()
    module.scope.reserve_name("Base")
    cls, cls_name = module.create_class(
        "MyClass",
        bases=[module.scope.name("Base")],
    )
    assert_code_equal(
        module,
        """
        class MyClass(Base):
            pass
        """,
    )


def test_create_class_with_decorators():
    module = codegen.Module()
    module.scope.reserve_name("deco")
    cls, cls_name = module.create_class(
        "MyClass",
        decorators=[module.scope.name("deco")],
    )
    assert_code_equal(
        module,
        """
        @deco
        class MyClass:
            pass
        """,
    )


def test_create_class_full():
    module = codegen.Module()
    module.scope.reserve_name("Base")
    module.scope.reserve_name("deco")
    cls, cls_name = module.create_class(
        "MyClass",
        bases=[module.scope.name("Base")],
        decorators=[module.scope.name("deco")],
    )
    func, _ = cls.body.create_function("__init__", args=["self"])
    source = as_source_code(module)
    assert "@deco" in source
    assert "class MyClass(Base):" in source
    assert "def __init__(self):" in source


def test_class_scope_isolation():
    """Class body scope doesn't leak into module scope."""
    module = codegen.Module()
    cls = codegen.Class("MyClass", parent_scope=module.scope)
    cls.reserve_name("internal_var")
    assert cls.is_name_in_use("internal_var")
    assert not module.scope.is_name_in_use("internal_var")


# --- as_python_source() tests ---


def test_as_python_source_expression():
    s = codegen.String("hello")
    assert s.as_python_source() == "'hello'"


def test_as_python_source_statement():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["x"], parent_scope=module.scope)
    func.create_return(func.name("x"))
    source = func.as_python_source()
    assert "def myfunc(x):" in source
    assert "return x" in source


def test_as_python_source_module():
    module = codegen.Module()
    module.scope.reserve_name("x")
    module.create_assignment("x", codegen.Number(42))
    source = module.as_python_source()
    assert "x = 42" in source


def test_as_python_source_block():
    scope = codegen.Scope()
    block = codegen.Block(scope)
    name = scope.reserve_name("y")
    block.create_assignment(name, codegen.String("test"))
    source = block.as_python_source()
    assert "y = 'test'" in source


def test_as_python_source_matches_as_source_code():
    """Verify as_python_source() matches the test helper as_source_code()."""
    module = codegen.Module()
    func, _ = module.create_function("my_func", args=["a", "b"])
    func.create_return(func.name("a").add(func.name("b")))
    assert module.as_python_source() == as_source_code(module)


# ---- Misc module methods


def test_module_as_ast():
    mod = codegen.Module()
    mod.scope.reserve_name("foo")
    mod.create_assignment("foo", codegen.Number(1))
    assert isinstance(mod.as_ast(), ast.Module)


def test_module_reserves_builtins_by_default():
    mod = codegen.Module()
    assert mod.scope.is_name_reserved("str")


def test_block_add_comment():
    mod = codegen.Module()
    mod.add_comment("Hello")
    mod.create_import("foo")
    assert_code_equal(
        mod,
        """
        # Hello
        import foo
        """,
    )


def test_comment_in_function_body():
    mod = codegen.Module()
    func, _ = mod.create_function("my_func", args=["x"])
    func.body.add_comment("Process the value")
    x = codegen.Name("x", func)
    func.body.create_return(x)
    assert_code_equal(
        mod,
        """
        def my_func(x):
            # Process the value
            return x
        """,
    )


def test_comment_in_class_body():
    mod = codegen.Module()
    cls, _ = mod.create_class("MyClass")
    cls.body.add_comment("Class fields")
    cls.body.create_field("name", codegen.Name("str", mod.scope))
    assert_code_equal(
        mod,
        """
        class MyClass:
            # Class fields
            name: str
        """,
    )


def test_multiple_comments():
    mod = codegen.Module()
    mod.add_comment("First comment")
    mod.add_comment("Second comment")
    mod.create_import("foo")
    assert_code_equal(
        mod,
        """
        # First comment
        # Second comment
        import foo
        """,
    )


def test_comment_compile_constraint():
    """Comments must not affect compile() — as_ast() must return a compilable AST."""
    mod = codegen.Module()
    mod.add_comment("This is a comment")
    func, _ = mod.create_function("greet", args=["name"])
    func.body.add_comment("Return greeting")
    func.body.create_return(codegen.String("hello"))

    # as_ast() must be compilable
    ast_tree = mod.as_ast()
    code = compile(ast_tree, "<test>", "exec")
    ns: dict[str, object] = {}
    exec(code, ns)
    assert ns["greet"]("world") == "hello"  # type: ignore


def test_comment_source_constraint():
    """as_python_source() must include the comments."""
    mod = codegen.Module()
    mod.add_comment("File header")
    mod.create_import("os")
    source = mod.as_python_source()
    assert "# File header" in source
    assert "import os" in source


def test_comment_interleaved_with_statements():
    mod = codegen.Module()
    mod.create_import("os")
    mod.add_comment("Now import sys")
    mod.create_import("sys")
    assert_code_equal(
        mod,
        """
        import os
        # Now import sys
        import sys
        """,
    )


def test_comment_only_block():
    """A block with only comments should compile (comments stripped = empty body)."""
    mod = codegen.Module()
    mod.add_comment("Just a comment")
    ast_tree = mod.as_ast()
    code = compile(ast_tree, "<test>", "exec")
    ns: dict[str, object] = {}
    exec(code, ns)
    # Should execute without error


def test_add_comment_wrap_basic():
    mod = codegen.Module()
    mod.add_comment("This is a long comment that should be wrapped into multiple lines", wrap=40)
    mod.create_import("foo")
    assert_code_equal(
        mod,
        """
        # This is a long comment that should be
        # wrapped into multiple lines
        import foo
        """,
    )


def test_add_comment_wrap_short_text_unchanged():
    """Text shorter than wrap width should remain a single comment."""
    mod = codegen.Module()
    mod.add_comment("Short", wrap=80)
    assert_code_equal(
        mod,
        """
        # Short
        """,
    )


def test_add_comment_wrap_none_default():
    """Without wrap, long text stays on one line."""
    mod = codegen.Module()
    long_text = "word " * 30
    mod.add_comment(long_text.strip())
    source = mod.as_python_source()
    comment_lines = [line for line in source.splitlines() if line.startswith("#")]
    assert len(comment_lines) == 1


def test_add_comment_wrap_in_function_body():
    mod = codegen.Module()
    func, _ = mod.create_function("f", args=[])
    func.body.add_comment("This is a very detailed explanation of what the function does", wrap=40)
    func.body.create_return(codegen.Number(1))
    assert_code_equal(
        mod,
        """
        def f():
            # This is a very detailed explanation of
            # what the function does
            return 1
        """,
    )


def test_add_comment_wrap_empty_string():
    mod = codegen.Module()
    mod.add_comment("", wrap=40)
    source = mod.as_python_source()
    assert "#" in source
