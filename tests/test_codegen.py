import ast
import builtins
import keyword
import textwrap

import pytest
from ast_decompiler.decompiler import Decompiler
from hypothesis import example, given
from hypothesis.strategies import text

from fluent_compiler import codegen
from fluent_compiler.utils import allowable_name


def normalize_python(txt):
    return textwrap.dedent(txt.rstrip()).strip()


def decompile(ast_node, indentation=4, line_length=100, starting_indentation=0):
    """Decompiles an AST into Python code."""
    decompiler = Decompiler(
        indentation=indentation,
        line_length=line_length,
        starting_indentation=starting_indentation,
    )
    return decompiler.run(ast_node)


def decompile_ast_list(ast_list):
    return decompile(ast.Module(body=ast_list, **codegen.DEFAULT_AST_ARGS_MODULE))


def as_source_code(codegen_ast):
    if not hasattr(codegen_ast, "as_ast"):
        ast_list = codegen_ast.as_ast_list()
    else:
        ast_list = [codegen_ast.as_ast()]
    return decompile_ast_list(ast_list)


def assert_code_equal(code1, code2):
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
    assert name2 == "name2"


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
    assert name == "my_arg2"


def test_reserve_function_arg_after_reserve_name():
    scope = codegen.Scope()
    scope.reserve_name("my_arg")
    with pytest.raises(AssertionError):
        scope.reserve_function_arg_name("my_arg")


def test_name_properties():
    scope = codegen.Scope()
    scope.reserve_name("name", properties={"FOO": True})
    assert scope.get_name_properties("name") == {"FOO": True}


def test_scope_variable_helper():
    scope = codegen.Scope()
    name = scope.reserve_name("name")
    ref1 = codegen.VariableReference(name, scope)
    ref2 = scope.variable(name)
    assert ref1 == ref2


# --- Function tests ---


def test_function():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["myarg1", "myarg2"], parent_scope=module.scope)
    assert_code_equal(
        as_source_code(func),
        """
        def myfunc(myarg1, myarg2):
            pass
        """,
    )


def test_function_return():
    module = codegen.Module()
    func = codegen.Function("myfunc", parent_scope=module)
    func.add_return(codegen.String("Hello"))
    assert_code_equal(
        as_source_code(func),
        """
        def myfunc():
            return 'Hello'
        """,
    )


def test_function_bad_name():
    module = codegen.Module()
    func = codegen.Function("my func", args=[], parent_scope=module)
    with pytest.raises(AssertionError):
        as_source_code(func)


def test_function_bad_arg():
    module = codegen.Module()
    func = codegen.Function("myfunc", args=["my arg"], parent_scope=module.scope)
    with pytest.raises(AssertionError):
        as_source_code(func)


def test_add_function():
    module = codegen.Module()
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, parent_scope=module)
    module.add_function(func_name, func)
    assert_code_equal(
        as_source_code(module),
        """
        def myfunc():
            pass
        """,
    )


def test_function_args_name_check():
    module = codegen.Module()
    module.scope.reserve_name("my_arg")
    func_name = module.scope.reserve_name("myfunc")
    with pytest.raises(AssertionError):
        codegen.Function(func_name, args=["my_arg"], parent_scope=module.scope)


def test_function_args_name_reserved_check():
    module = codegen.Module()
    module.scope.reserve_function_arg_name("my_arg")
    func_name = module.scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["my_arg"], parent_scope=module.scope)
    func.add_return(func.variable("my_arg"))
    assert_code_equal(
        as_source_code(func),
        """
        def myfunc(my_arg):
            return my_arg
        """,
    )


# --- Variable reference tests ---


def test_variable_reference():
    scope = codegen.Scope()
    name = scope.reserve_name("name")
    ref = codegen.VariableReference(name, scope)
    assert as_source_code(ref) == "name"


def test_variable_reference_check():
    scope = codegen.Scope()
    with pytest.raises(AssertionError):
        codegen.VariableReference("name", scope)


def test_variable_reference_function_arg_check():
    scope = codegen.Scope()
    func_name = scope.reserve_name("myfunc")
    func = codegen.Function(func_name, args=["my_arg"], parent_scope=scope)
    # Can't use undefined 'some_name'
    with pytest.raises(AssertionError):
        codegen.VariableReference("some_name", func)
    # But can use function argument 'my_arg'
    ref = codegen.VariableReference("my_arg", func)
    assert_code_equal(as_source_code(ref), "my_arg")


def test_variable_reference_bad():
    module = codegen.Module()
    name = module.scope.reserve_name("name")
    ref = codegen.VariableReference(name, module.scope)
    ref.name = "bad name"
    with pytest.raises(AssertionError):
        as_source_code(ref)


# --- Assignment tests ---


def test_add_assignment_unreserved():
    scope = codegen.Module()
    with pytest.raises(AssertionError):
        scope.add_assignment("x", codegen.String("a string"))


def test_add_assignment_reserved():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.add_assignment(name, codegen.String("a string"))
    assert_code_equal(
        as_source_code(module),
        """
        x = 'a string'
        """,
    )


def test_add_assignment_bad():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.add_assignment(name, codegen.String("a string"))
    module.statements[0].name = "something with a space"
    with pytest.raises(AssertionError):
        as_source_code(module)


def test_multiple_add_assignment():
    module = codegen.Module()
    name = module.scope.reserve_name("x")
    module.add_assignment(name, codegen.String("a string"))
    with pytest.raises(AssertionError):
        module.add_assignment(name, codegen.String("another string"))


def test_multiple_add_assignment_in_inherited_scope():
    scope = codegen.Scope()
    scope.reserve_name("myfunc")
    func = codegen.Function("myfunc", args=[], parent_scope=scope)
    try_ = codegen.Try([], func)
    name = func.reserve_name("name")

    try_.try_block.add_assignment(name, codegen.Number(1))
    with pytest.raises(AssertionError):
        try_.try_block.add_assignment(name, codegen.Number(2))
    with pytest.raises(AssertionError):
        try_.except_block.add_assignment(name, codegen.Number(2))
    try_.except_block.add_assignment(name, codegen.Number(2), allow_multiple=True)


# --- Function call tests ---


def test_function_call_unknown():
    scope = codegen.Scope()
    with pytest.raises(AssertionError):
        codegen.FunctionCall("a_function", [], {}, scope)


def test_function_call_known():
    module = codegen.Module()
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall("a_function", [], {}, module.scope)
    assert_code_equal(as_source_code(func_call), "a_function()")


def test_function_call_args_and_kwargs():
    module = codegen.Module()
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall(
        "a_function",
        [codegen.Number(123)],
        {"x": codegen.String("hello")},
        module.scope,
    )
    assert_code_equal(as_source_code(func_call), "a_function(123, x='hello')")


def test_function_call_bad_name():
    module = codegen.Module()
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall("a_function", [], {}, module.scope)
    func_call.function_name = "bad function name"
    with pytest.raises(AssertionError):
        as_source_code(func_call)


def test_function_call_bad_kwarg_names():
    module = codegen.Module()
    module.scope.reserve_name("a_function")
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
        func_call = codegen.FunctionCall("a_function", [], {arg_name: codegen.String("a")}, module.scope)
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
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall("a_function", [], {"hyphen-ated": codegen.Number(1)}, module.scope)
    assert_code_equal(
        as_source_code(func_call),
        """
        a_function(**{'hyphen-ated': 1})
        """,
    )


def test_function_call_sensitive():
    module = codegen.Module()
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall("a_function", [], {}, module.scope)
    func_call.function_name = "exec"
    with pytest.raises(AssertionError):
        as_source_code(func_call)


def test_method_call_bad_name():
    scope = codegen.Module()
    s = codegen.String("x")
    method_call = codegen.MethodCall(s, "bad method name", [], scope)
    with pytest.raises(AssertionError):
        as_source_code(method_call)


# --- Try/Catch tests ---


def test_try_catch():
    scope = codegen.Scope()
    scope.reserve_name("MyError")
    try_ = codegen.Try([scope.variable("MyError")], scope)
    assert_code_equal(
        as_source_code(try_),
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
    try_.try_block.add_assignment("x", codegen.String("x"))
    try_.except_block.add_assignment("y", codegen.String("y"))
    try_.else_block.add_assignment("z", codegen.String("z"))
    assert_code_equal(
        as_source_code(try_),
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
    scope.reserve_name("MyError")
    scope.reserve_name("OtherError")
    try_ = codegen.Try([scope.variable("MyError"), scope.variable("OtherError")], scope)
    assert_code_equal(
        as_source_code(try_),
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

    try_.try_block.add_assignment(name, codegen.String("x"))
    assert not try_.has_assignment_for_name(name)

    try_.except_block.add_assignment(name, codegen.String("x"), allow_multiple=True)
    assert try_.has_assignment_for_name(name)


def test_try_catch_has_assignment_for_name_2():
    scope = codegen.Scope()
    try_ = codegen.Try([], scope)
    name = scope.reserve_name("foo")

    try_.except_block.add_assignment(name, codegen.String("x"))
    assert not try_.has_assignment_for_name(name)

    try_.else_block.add_assignment(name, codegen.String("x"), allow_multiple=True)
    assert try_.has_assignment_for_name(name)


# --- If tests ---


def test_if_empty():
    scope = codegen.Module()
    if_statement = codegen.If(scope)
    if_statement = if_statement.finalize()
    assert_code_equal(as_source_code(if_statement), "")


def test_if_one_if():
    scope = codegen.Module()
    if_statement = codegen.If(scope)
    first_block = if_statement.add_if(codegen.Number(1))
    first_block.add_return(codegen.Number(2))
    assert_code_equal(
        as_source_code(if_statement),
        """
        if 1:
            return 2
        """,
    )


def test_if_two_ifs():
    scope = codegen.Module()
    if_statement = codegen.If(scope)
    first_block = if_statement.add_if(codegen.Number(1))
    first_block.add_return(codegen.Number(2))
    second_block = if_statement.add_if(codegen.Number(3))
    second_block.add_return(codegen.Number(4))
    assert_code_equal(
        as_source_code(if_statement),
        """
        if 1:
            return 2
        elif 3:
            return 4
        """,
    )


def test_if_with_else():
    scope = codegen.Module()
    if_statement = codegen.If(scope)
    first_block = if_statement.add_if(codegen.Number(1))
    first_block.add_return(codegen.Number(2))
    if_statement.else_block.add_return(codegen.Number(3))
    assert_code_equal(
        as_source_code(if_statement),
        """
        if 1:
            return 2
        else:
            return 3
        """,
    )


def test_if_no_ifs():
    scope = codegen.Module()
    if_statement = codegen.If(scope)
    if_statement.else_block.add_return(codegen.Number(3))
    if_statement = if_statement.finalize()
    assert_code_equal(
        as_source_code(if_statement),
        """
        return 3
        """,
    )


# --- Expression tests ---


@given(text())
def test_string(t):
    assert t == eval(as_source_code(codegen.String(t))), f" for t = {t!r}"


def test_string_join_empty():
    join = codegen.StringJoin.build([])
    assert_code_equal(as_source_code(join), "''")


def test_string_join_one():
    join = codegen.StringJoin.build([codegen.String("hello")])
    assert_code_equal(as_source_code(join), "'hello'")


def test_concat_string_join_two():
    module = codegen.Module()
    module.scope.reserve_name("tmp", properties={codegen.PROPERTY_TYPE: str})
    var = module.scope.variable("tmp")
    join = codegen.ConcatJoin([codegen.String("hello "), var])
    assert_code_equal(as_source_code(join), "'hello ' + tmp")


def test_f_string_join_two():
    module = codegen.Module()
    module.scope.reserve_name("tmp", properties={codegen.PROPERTY_TYPE: str})
    var = module.scope.variable("tmp")
    join = codegen.FStringJoin([codegen.String("hello "), var])
    assert_code_equal(as_source_code(join), "f'hello {tmp}'")


def test_string_join_collapse_strings():
    scope = codegen.Scope()
    scope.reserve_name("tmp", properties={codegen.PROPERTY_TYPE: str})
    var = scope.variable("tmp")
    join1 = codegen.ConcatJoin.build(
        [
            codegen.String("hello "),
            codegen.String("there "),
            var,
            codegen.String(" how"),
            codegen.String(" are you?"),
        ]
    )
    assert_code_equal(as_source_code(join1), "'hello there ' + tmp + ' how are you?'")


def test_dict_lookup():
    scope = codegen.Scope()
    scope.reserve_name("tmp")
    var = scope.variable("tmp")
    lookup = codegen.DictLookup(var, codegen.String("x"))
    assert_code_equal(as_source_code(lookup), "tmp['x']")


def test_equals():
    eq = codegen.Equals(codegen.String("x"), codegen.String("y"))
    assert_code_equal(as_source_code(eq), "'x' == 'y'")


def test_or():
    or_ = codegen.Or(codegen.String("x"), codegen.String("y"))
    assert_code_equal(as_source_code(or_), "'x' or 'y'")


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
    assert (
        allowable_name(cleaned) or (cleaned in dir(builtins)) or keyword.iskeyword(cleaned)
    ), f" for t = {t!r}"
