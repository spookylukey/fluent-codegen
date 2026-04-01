"""Tests for AugAssign (augmented assignment) support."""

import textwrap

import pytest

from fluent_codegen import codegen
from fluent_codegen.codegen import auto


def normalize_python(txt: str):
    return textwrap.dedent(txt.rstrip()).strip()


def assert_code_equal(code1, code2):
    if isinstance(code1, codegen.E):
        code1 = codegen.E_to_Expression(code1).as_python_source()
    elif not isinstance(code1, str):
        code1 = code1.as_python_source()
    if isinstance(code2, codegen.E):
        code2 = codegen.E_to_Expression(code2).as_python_source()
    elif not isinstance(code2, str):
        code2 = code2.as_python_source()
    assert normalize_python(code1) == normalize_python(code2)


class TestAugAssign:
    # --- Basic: one representative operator via the convenience method ---

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, "+=", auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    # --- Target types ---

    def test_subscript_target(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([0, 0, 0]))
        mod.aug_assign(x.subscript(auto(0)), "+=", auto(5))
        assert_code_equal(
            mod,
            """\
            x = [0, 0, 0]
            x[0] += 5
            """,
        )

    def test_attr_target(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x.attr("count"), "+=", auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x.count += 1
            """,
        )

    # --- Value types ---

    def test_iadd_list(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([1]))
        mod.aug_assign(x, "+=", auto([2, 3, 4]))
        assert_code_equal(
            mod,
            """\
            x = [1]
            x += [2, 3, 4]
            """,
        )

    def test_e_expression_as_value(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        y = mod.assign("y", auto(5))
        mod.aug_assign(x, "+=", y.e * 2)
        assert_code_equal(
            mod,
            """\
            x = 0
            y = 5
            x += y * 2
            """,
        )

    # --- Low-level Statement object ---

    def test_aug_assignment_statement_directly(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        stmt = codegen.AugAssignment(x, "+=", auto(1))
        mod.add_statement(stmt)
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    # --- Error cases ---

    def test_invalid_operator_string(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        with pytest.raises(AssertionError, match="Invalid augmented assignment operator"):
            mod.aug_assign(x, "!=", auto(1))  # type: ignore[arg-type]

    def test_tuple_target_rejected(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        y = mod.assign("y", auto(0))
        with pytest.raises(AssertionError, match="Invalid augmented assignment target"):
            mod.aug_assign((x, y), "+=", auto(1))  # type: ignore[arg-type]

    # --- All operators produce valid code (parametrized) ---

    @pytest.mark.parametrize(
        "op, expected_op",
        [
            ("+=", "+="),
            ("-=", "-="),
            ("*=", "*="),
            ("/=", "/="),
            ("//=", "//="),
            ("%=", "%="),
            ("**=", "**="),
            ("@=", "@="),
            ("<<=", "<<="),
            (">>=", ">>="),
            ("|=", "|="),
            ("&=", "&="),
            ("^=", "^="),
        ],
    )
    def test_all_operators(self, op, expected_op):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, op, auto(1))
        assert_code_equal(
            mod,
            f"""\
            x = 0
            x {expected_op} 1
            """,
        )

    # --- Realistic example ---

    def test_accumulator_loop(self):
        mod = codegen.Module()
        func, _ = mod.create_function("accumulate", args=["items"])
        items = func.name("items")
        total = func.body.assign("total", auto(0))
        loop, item = func.body.create_for("item", items)
        loop.body.aug_assign(total, "+=", item)
        func.body.add_statement(codegen.Return(total))
        assert_code_equal(
            mod,
            """\
            def accumulate(items):
                total = 0
                for item in items:
                    total += item
                return total
            """,
        )
