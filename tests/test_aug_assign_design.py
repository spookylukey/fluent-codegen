"""Design exploration for AugAssign (+=, -=, etc.) API.

This file contains several alternative API designs as test functions.
All tests are expected to FAIL — they test APIs that don't exist yet.
The goal is to evaluate which API feels best from the user's perspective.

Python AugAssign targets can be: Name, Attribute, Subscript (but NOT tuples).
Operators: +=, -=, *=, /=, //=, %=, **=, @=, <<=, >>=, |=, &=, ^=
"""

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


# ============================================================================
# DESIGN A: Named methods on Block — aug_assign(target, op, value)
#
# Mirrors the existing block.assign() but takes an operator string.
# Pros: Explicit, discoverable, consistent with block.assign().
# Cons: The op string is easy to get wrong; not very fluent.
# ============================================================================


class TestDesignA_named_method:
    """block.aug_assign(target, op_string, value)"""

    def test_simple_iadd(self):
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

    def test_isub(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, "-=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x -= 3
            """,
        )

    def test_imul_with_list(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([1, 2]))
        mod.aug_assign(x, "*=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = [1, 2]
            x *= 3
            """,
        )

    def test_iadd_list_extend(self):
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
        x = mod.assign("x", auto(0))  # pretend x has .value
        mod.aug_assign(x.attr("value"), "+=", auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x.value += 1
            """,
        )

    def test_ibitand(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0xFF))
        mod.aug_assign(x, "&=", auto(0x0F))
        assert_code_equal(
            mod,
            """\
            x = 255
            x &= 15
            """,
        )

    def test_all_operators(self):
        """Verify all operators are accepted."""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        for op in ["+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=", "<<=", ">>=", "|=", "&=", "^="]:
            mod.aug_assign(x, op, auto(1))


# ============================================================================
# DESIGN B: Operator-specific methods on Block
#
# block.iadd(target, value), block.isub(target, value), etc.
# Pros: No string matching, each method is clear.
# Cons: Many methods on Block; pollutes the namespace.
# ============================================================================


class TestDesignB_operator_methods:
    """block.iadd(target, value), block.isub(target, value), etc."""

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.iadd(x, auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    def test_isub(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.isub(x, auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x -= 3
            """,
        )

    def test_iadd_list(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([1]))
        mod.iadd(x, auto([2, 3, 4]))
        assert_code_equal(
            mod,
            """\
            x = [1]
            x += [2, 3, 4]
            """,
        )

    def test_subscript_target(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([0, 0, 0]))
        mod.iadd(x.subscript(auto(0)), auto(5))
        assert_code_equal(
            mod,
            """\
            x = [0, 0, 0]
            x[0] += 5
            """,
        )

    def test_ifloordiv(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(100))
        mod.ifloordiv(x, auto(3))
        assert_code_equal(
            mod,
            """\
            x = 100
            x //= 3
            """,
        )


# ============================================================================
# DESIGN C: Enum-based — block.aug_assign(target, Op.ADD, value)
#
# Uses an enum to select the operator. Similar to A but type-safe.
# Pros: Type-safe, IDE completion, no string matching.
# Cons: Slightly more verbose; user has to import the enum.
# ============================================================================


class TestDesignC_enum:
    """block.aug_assign(target, AugOp.ADD, value)"""

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, codegen.AugOp.ADD, auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    def test_isub(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, codegen.AugOp.SUB, auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x -= 3
            """,
        )

    def test_iadd_list(self):
        mod = codegen.Module()
        x = mod.assign("x", auto([1]))
        mod.aug_assign(x, codegen.AugOp.ADD, auto([2, 3, 4]))
        assert_code_equal(
            mod,
            """\
            x = [1]
            x += [2, 3, 4]
            """,
        )


# ============================================================================
# DESIGN D: Reuse existing ArithOp classes as operator tokens
#
# The codegen module already has Add, Sub, Mul etc. We could reuse these
# classes (not instances) as the operator selector:
#   block.aug_assign(target, codegen.Add, value)
#
# Pros: No new types needed; reuses what exists.
# Cons: Passing a class as an operator token is unusual; Add is a BinaryOperator
#       (expression), not an assignment op — conceptual mismatch.
# ============================================================================


class TestDesignD_reuse_arith_classes:
    """block.aug_assign(target, codegen.Add, value)"""

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, codegen.Add, auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    def test_isub(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, codegen.Sub, auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x -= 3
            """,
        )


# ============================================================================
# DESIGN E: Generalised assign() with optional `op` parameter
#
# Extends the existing block.assign() to accept an `op` keyword:
#   block.assign("x", value)           → x = value     (as today)
#   block.assign("x", value, op="+=")  → x += value
#
# This doesn't work cleanly: assign() creates new names and returns Name
# objects. AugAssign targets must already exist. Mixing the two would be
# confusing. Marking as a bad idea but including for completeness.
# ============================================================================


class TestDesignE_extended_assign:
    """block.assign(target, value, op='+=')"""

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        # This is awkward — assign() normally creates new names.
        # For aug_assign the name must already exist.
        # We'd pass the Name object, not a string:
        mod.assign(x, auto(1), op="+=")
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )


# ============================================================================
# DESIGN F: Statement-object API — AugAssign as a first-class Statement
#
# Creates an AugAssign statement object and adds it to the block explicitly.
# This parallels how Assignment is a Statement class today.
#   block.add_statement(codegen.AugAssign(x, "+=", auto(1)))
# or with a convenience method:
#   block.aug_assign(x, "+=", auto(1))
#
# This is really the same as Design A but showing the lower-level API.
# ============================================================================


class TestDesignF_statement_object:
    """codegen.AugAssign(target, op, value) as a Statement"""

    def test_iadd_via_add_statement(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.add_statement(codegen.AugAssign(x, "+=", auto(1)))
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    def test_iadd_via_convenience(self):
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


# ============================================================================
# DESIGN G: Unified assign() that accepts Name/Attr/Subscript as target
#
# Today, block.assign("x", value) creates a new name.
# We add: block.assign(existing_name, value, op="+=")
# where passing a Name (not str) signals "this name already exists".
#
# Regular re-assignment: block.assign(x, new_value)  → x = new_value
# Aug-assignment:        block.assign(x, delta, op="+=") → x += delta
#
# Pros: Unifies all assignment forms under one method.
# Cons: The method is getting overloaded; `op` parameter on assign feels odd.
#       Regular re-assignment of existing names also needs to work.
# ============================================================================


class TestDesignG_unified_assign:
    """block.assign(existing_name, value, op='+=')"""

    def test_regular_assign(self):
        mod = codegen.Module()
        mod.assign("x", auto(0))
        assert_code_equal(
            mod,
            """\
            x = 0
            """,
        )

    def test_iadd(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.assign(x, auto(1), op="+=")
        assert_code_equal(
            mod,
            """\
            x = 0
            x += 1
            """,
        )

    def test_reassign(self):
        """Plain re-assignment to an existing name."""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.assign(x, auto(99))
        assert_code_equal(
            mod,
            """\
            x = 0
            x = 99
            """,
        )


# ============================================================================
# EVALUATION
# ============================================================================
#
# Design A (string op):     Simple, readable, but strings are fragile.
# Design B (many methods):  Clean per-call, but adds ~13 methods to Block.
# Design C (enum op):       Type-safe, IDE-friendly, slightly verbose.
# Design D (reuse classes):  Clever but confusing — classes as tokens.
# Design E (extend assign):  Muddies assign()'s semantics.
# Design F (statement obj):  Good low-level API; pairs well with A.
# Design G (unified assign): Overloads assign() too much.
#
# RECOMMENDATION: Design A + F (combined)
#
# - An AugAssignment Statement class (Design F) for the low level.
# - A block.aug_assign(target, op, value) convenience method (Design A)
#   that accepts operator strings like "+=", "-=", etc.
# - The string-based operator is easy to read and write; the set of valid
#   strings is small and can be validated with a clear error message.
# - This follows the existing pattern: Assignment is a Statement class,
#   and block.assign() / block.create_assignment() are conveniences.
#
# The operator could alternatively be an enum (Design C) for extra
# type safety. The string approach is more ergonomic for a codegen
# library where the user is thinking in terms of Python syntax.
# ============================================================================


# ============================================================================
# RECOMMENDED API — comprehensive tests for Design A + F
# ============================================================================


class TestAugAssignRecommended:
    """Full test suite for the recommended design."""

    # --- Basic operators ---

    def test_iadd_number(self):
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

    def test_isub(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, "-=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x -= 3
            """,
        )

    def test_imul(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(5))
        mod.aug_assign(x, "*=", auto(2))
        assert_code_equal(
            mod,
            """\
            x = 5
            x *= 2
            """,
        )

    def test_idiv(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, "/=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x /= 3
            """,
        )

    def test_ifloordiv(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, "//=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x //= 3
            """,
        )

    def test_imod(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(10))
        mod.aug_assign(x, "%=", auto(3))
        assert_code_equal(
            mod,
            """\
            x = 10
            x %= 3
            """,
        )

    def test_ipow(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(2))
        mod.aug_assign(x, "**=", auto(10))
        assert_code_equal(
            mod,
            """\
            x = 2
            x **= 10
            """,
        )

    def test_imatmul(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, "@=", auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x @= 1
            """,
        )

    def test_ilshift(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(1))
        mod.aug_assign(x, "<<=", auto(4))
        assert_code_equal(
            mod,
            """\
            x = 1
            x <<= 4
            """,
        )

    def test_irshift(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(16))
        mod.aug_assign(x, ">>=", auto(2))
        assert_code_equal(
            mod,
            """\
            x = 16
            x >>= 2
            """,
        )

    def test_ibitor(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x, "|=", auto(0xFF))
        assert_code_equal(
            mod,
            """\
            x = 0
            x |= 255
            """,
        )

    def test_ibitand(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0xFF))
        mod.aug_assign(x, "&=", auto(0x0F))
        assert_code_equal(
            mod,
            """\
            x = 255
            x &= 15
            """,
        )

    def test_ixor(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0xFF))
        mod.aug_assign(x, "^=", auto(0x0F))
        assert_code_equal(
            mod,
            """\
            x = 255
            x ^= 15
            """,
        )

    # --- Target types ---

    def test_name_target(self):
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

    def test_nested_subscript_attr_target(self):
        """x.items[0] += 1"""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        mod.aug_assign(x.attr("items").subscript(auto(0)), "+=", auto(1))
        assert_code_equal(
            mod,
            """\
            x = 0
            x.items[0] += 1
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

    def test_iadd_string(self):
        mod = codegen.Module()
        x = mod.assign("x", auto("hello"))
        mod.aug_assign(x, "+=", auto(" world"))
        assert_code_equal(
            mod,
            """\
            x = 'hello'
            x += ' world'
            """,
        )

    def test_iadd_name_as_value(self):
        """x += y where y is another variable."""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        y = mod.assign("y", auto(10))
        mod.aug_assign(x, "+=", y)
        assert_code_equal(
            mod,
            """\
            x = 0
            y = 10
            x += y
            """,
        )

    def test_iadd_expression_as_value(self):
        """x += y * 2"""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        y = mod.assign("y", auto(5))
        mod.aug_assign(x, "+=", y.mul(auto(2)))
        assert_code_equal(
            mod,
            """\
            x = 0
            y = 5
            x += y * 2
            """,
        )

    def test_iadd_e_expression_as_value(self):
        """x += y * 2 using E-objects."""
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

    # --- Error cases ---

    def test_invalid_operator_string(self):
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        with pytest.raises((ValueError, AssertionError)):
            mod.aug_assign(x, "!=", auto(1))

    def test_tuple_target_rejected(self):
        """AugAssign doesn't support tuple targets in Python."""
        mod = codegen.Module()
        x = mod.assign("x", auto(0))
        y = mod.assign("y", auto(0))
        with pytest.raises((TypeError, AssertionError)):
            mod.aug_assign((x, y), "+=", auto(1))

    # --- Low-level Statement object (Design F) ---

    def test_aug_assignment_statement_directly(self):
        """AugAssignment can be created and added manually."""
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

    # --- In context: inside a function body ---

    def test_inside_function(self):
        mod = codegen.Module()
        func = mod.create_function("accumulate", args=["items"])
        items = func.scope.name("items")
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

    # --- Realistic example: building up a dictionary ---

    def test_dict_update_with_bitor(self):
        """d |= {'new_key': value} — Python 3.9+ dict merge."""
        mod = codegen.Module()
        d = mod.assign("d", auto({"a": 1}))
        mod.aug_assign(d, "|=", auto({"b": 2}))
        assert_code_equal(
            mod,
            """\
            d = {'a': 1}
            d |= {'b': 2}
            """,
        )
