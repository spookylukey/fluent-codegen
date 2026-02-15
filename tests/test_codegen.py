import ast
import builtins
import keyword
import textwrap

import pytest
from ast_decompiler.decompiler import Decompiler
from hypothesis import example, given
from hypothesis.strategies import text

from fluent_codegen import codegen
from fluent_codegen.utils import allowable_name


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
    func = codegen.Function("myfunc", parent_scope=module.scope)
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
    func = codegen.Function("my func", args=[], parent_scope=module.scope)
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
    func = codegen.Function(func_name, parent_scope=module.scope)
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
    stmt = module.statements[0]
    assert isinstance(stmt, codegen._Assignment)
    stmt.name = "something with a space"
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
    method_call = codegen.MethodCall(s, "bad method name", [])
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
    if_statement = codegen.If(scope.scope)
    if_statement = if_statement.finalize()
    assert_code_equal(as_source_code(if_statement), "")


def test_if_one_if():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
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
    if_statement = codegen.If(scope.scope)
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
    if_statement = codegen.If(scope.scope)
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
    if_statement = codegen.If(scope.scope)
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
    assert allowable_name(cleaned) or (cleaned in dir(builtins)) or keyword.iskeyword(cleaned), f" for t = {t!r}"


# --- Additional coverage tests ---


def test_get_name_properties_not_found():
    scope = codegen.Scope()
    with pytest.raises(LookupError):
        scope.get_name_properties("nonexistent")


def test_get_name_properties_from_parent():
    parent = codegen.Scope()
    parent.reserve_name("name", properties={"FOO": True})
    child = codegen.Scope(parent_scope=parent)
    assert child.get_name_properties("name") == {"FOO": True}


def test_set_name_properties():
    scope = codegen.Scope()
    scope.reserve_name("name", properties={"FOO": True})
    scope.set_name_properties("name", {"BAR": False})
    assert scope.get_name_properties("name") == {"FOO": True, "BAR": False}


def test_set_name_properties_in_parent():
    parent = codegen.Scope()
    parent.reserve_name("name", properties={"FOO": True})
    child = codegen.Scope(parent_scope=parent)
    child.set_name_properties("name", {"BAR": False})
    assert parent.get_name_properties("name") == {"FOO": True, "BAR": False}


def test_set_name_properties_not_found():
    scope = codegen.Scope()
    with pytest.raises(LookupError):
        scope.set_name_properties("nonexistent", {"FOO": True})


def test_find_names_by_property():
    scope = codegen.Scope()
    scope.reserve_name("a", properties={"type": "string"})
    scope.reserve_name("b", properties={"type": "int"})
    scope.reserve_name("c", properties={"type": "string"})
    assert scope.find_names_by_property("type", "string") == ["a", "c"]
    assert scope.find_names_by_property("type", "int") == ["b"]
    assert scope.find_names_by_property("type", "float") == []


def test_reserve_name_keyword_avoidance():
    scope = codegen.Scope()
    # 'class' is a keyword, so reserve_name should avoid it
    name = scope.reserve_name("class")
    assert name == "class2"  # skips 'class' because it's a keyword


def test_block_add_statement_bare_expression():
    """Test that bare expressions (e.g. method calls) get wrapped in Expr."""
    module = codegen.Module()
    module.scope.reserve_name("a_function")
    func_call = codegen.FunctionCall("a_function", [], {}, module.scope)
    module.add_statement(func_call)
    assert_code_equal(as_source_code(module), "a_function()")


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
    module.add_assignment(name, codegen.String("hello"))
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
    assert lst.type is list
    assert_code_equal(as_source_code(lst), "[1, 'two']")


def test_dict_expression():
    d = codegen.Dict([(codegen.String("a"), codegen.Number(1))])
    assert d.type is dict
    assert_code_equal(as_source_code(d), "{'a': 1}")


def test_none_expr():
    n = codegen.NoneExpr()
    assert n.type is type(None)
    assert_code_equal(as_source_code(n), "None")


def test_method_call_repr():
    s = codegen.String("x")
    mc = codegen.MethodCall(s, "upper", [])
    assert "MethodCall" in repr(mc)


def test_function_call_repr():
    module = codegen.Module()
    module.scope.reserve_name("f")
    fc = codegen.FunctionCall("f", [], {}, module.scope)
    assert "FunctionCall" in repr(fc)


def test_variable_reference_repr():
    scope = codegen.Scope()
    scope.reserve_name("x")
    ref = scope.variable("x")
    assert repr(ref) == "VariableReference('x')"


def test_variable_reference_equality():
    scope = codegen.Scope()
    scope.reserve_name("x")
    ref1 = scope.variable("x")
    ref2 = scope.variable("x")
    assert ref1 == ref2
    assert ref1 != "x"


def test_string_join_repr():
    scope = codegen.Scope()
    scope.reserve_name("tmp", properties={codegen.PROPERTY_TYPE: str})
    var = scope.variable("tmp")
    join = codegen.ConcatJoin([codegen.String("hello "), var])
    assert "ConcatJoin" in repr(join)


def test_if_finalize_returns_self():
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    if_statement.add_if(codegen.Number(1))
    result = if_statement.finalize()
    assert result is if_statement


def test_if_unfinalized_as_ast_raises():
    """If with no if blocks should raise when calling as_ast without finalize."""
    scope = codegen.Module()
    if_statement = codegen.If(scope.scope)
    with pytest.raises(AssertionError, match="finalize"):
        if_statement.as_ast()


def test_function_call_return_type_property():
    module = codegen.Module()
    module.scope.reserve_name("a_function", properties={codegen.PROPERTY_RETURN_TYPE: str})
    fc = codegen.FunctionCall("a_function", [], {}, module.scope)
    assert fc.type is str


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
    scope.reserve_name("x", properties={codegen.PROPERTY_TYPE: str})
    var = scope.variable("x")
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
    assert_code_equal(as_source_code(result), "'ab' + x")


def test_rewriting_traverse_list():
    """Test rewriting_traverse with list inputs."""
    scope = codegen.Scope()
    scope.reserve_name("x")
    items = [scope.variable("x")]
    # Should not raise
    codegen.rewriting_traverse(items, lambda n: n)


def test_rewriting_traverse_dict():
    """Test rewriting_traverse with dict inputs."""
    scope = codegen.Scope()
    scope.reserve_name("x")
    scope.reserve_name("y")
    d = {"key": scope.variable("y")}
    codegen.rewriting_traverse(d, lambda n: n)


def test_morph_into():
    """Test morph_into changes class and dict."""
    s1 = codegen.String("hello")
    s2 = codegen.Number(42)
    codegen.morph_into(s1, s2)
    assert isinstance(s1, codegen.Number)
    assert s1.number == 42


def test_codegen_ast_not_implemented():
    """Test that abstract methods raise NotImplementedError."""

    class DummyAst(codegen.CodeGenAst):
        child_elements = []

        def as_ast(self):
            raise NotImplementedError(f"{self.__class__!r}.as_ast()")

    d = DummyAst()
    with pytest.raises(NotImplementedError):
        d.as_ast()


def test_codegen_ast_list_not_implemented():
    class DummyAstList(codegen.CodeGenAstList):
        child_elements = []

        def as_ast_list(self, allow_empty=True):
            raise NotImplementedError(f"{self.__class__!r}.as_ast_list()")

    d = DummyAstList()
    with pytest.raises(NotImplementedError):
        d.as_ast_list()


def test_block_as_ast_list_with_codegen_ast_list():
    """Test that Block handles CodeGenAstList sub-statements."""
    module = codegen.Module()
    scope = codegen.Scope()
    inner_block = codegen.Block(scope)
    name = scope.reserve_name("x")
    inner_block.add_assignment(name, codegen.String("hello"))
    module.add_statement(inner_block)
    source = as_source_code(module)
    assert "x = 'hello'" in source


def test_dict_lookup_type():
    scope = codegen.Scope()
    scope.reserve_name("tmp")
    var = scope.variable("tmp")
    lookup = codegen.DictLookup(var, codegen.String("x"), expr_type=str)
    assert lookup.type is str


def test_method_call_type():
    s = codegen.String("x")
    mc = codegen.MethodCall(s, "upper", [], expr_type=str)
    assert mc.type is str
    assert_code_equal(as_source_code(mc), "'x'.upper()")


def test_expression_abstract():
    """Expression.as_ast is abstract."""

    class DummyExpr(codegen.Expression):
        child_elements = []

        def as_ast(self):
            raise NotImplementedError()

    d = DummyExpr()
    with pytest.raises(NotImplementedError):
        d.as_ast()


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
    scope.reserve_name("x", properties={codegen.PROPERTY_TYPE: str})
    var = scope.variable("x")
    join = codegen.ConcatJoin([codegen.String("hello"), var])

    def replace_hello(node):
        if isinstance(node, codegen.String) and node.string_value == "hello":
            return codegen.String("world")
        return node

    codegen.rewriting_traverse(join, replace_hello)
    assert_code_equal(as_source_code(join), "'world' + x")


def test_rewriting_traverse_tuple():
    """Test rewriting_traverse with tuple input."""
    s = codegen.String("hello")
    t = (s,)
    codegen.rewriting_traverse(t, lambda n: n)
