"""Tests for the FizzBuzz README example."""

import textwrap

import pytest

from fluent_codegen import codegen


def _build_fizzbuzz_module() -> codegen.Module:
    """Build the fizzbuzz module using the codegen API — matches the README example."""
    module = codegen.Module()
    func, _ = module.create_function("fizzbuzz", args=["n"])

    n = codegen.Name("n", func)

    if_stmt = func.body.create_if()

    # if n % 15 == 0: return "FizzBuzz"   — using fluent chaining
    branch_15 = if_stmt.create_if_branch(n.mod(codegen.Number(15)).eq(codegen.Number(0)))
    branch_15.create_return(codegen.String("FizzBuzz"))

    # elif n % 3 == 0: return "Fizz"
    branch_3 = if_stmt.create_if_branch(n.mod(codegen.Number(3)).eq(codegen.Number(0)))
    branch_3.create_return(codegen.String("Fizz"))

    # elif n % 5 == 0: return "Buzz"
    branch_5 = if_stmt.create_if_branch(n.mod(codegen.Number(5)).eq(codegen.Number(0)))
    branch_5.create_return(codegen.String("Buzz"))

    # else: return str(n)
    if_stmt.else_block.create_return(codegen.function_call("str", [n], {}, func))

    return module


def test_generated_source():
    assert _build_fizzbuzz_module().as_python_source() == textwrap.dedent("""\
        def fizzbuzz(n):
            if n % 15 == 0:
                return 'FizzBuzz'
            elif n % 3 == 0:
                return 'Fizz'
            elif n % 5 == 0:
                return 'Buzz'
            else:
                return str(n)""")


@pytest.fixture()
def fizzbuzz_func():
    code = compile(_build_fizzbuzz_module().as_ast(), "<fizzbuzz>", "exec")
    ns: dict[str, object] = {}
    exec(code, ns)
    return ns["fizzbuzz"]


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1"),
        (3, "Fizz"),
        (5, "Buzz"),
        (7, "7"),
        (15, "FizzBuzz"),
        (30, "FizzBuzz"),
    ],
)
def test_fizzbuzz_values(fizzbuzz_func, n, expected):
    assert fizzbuzz_func(n) == expected
