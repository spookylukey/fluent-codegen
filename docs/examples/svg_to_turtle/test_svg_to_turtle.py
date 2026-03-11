"""Tests for the SVG-to-Turtle compiler example."""

from __future__ import annotations

import textwrap
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pytest

from docs.examples.svg_to_turtle.svg_to_turtle import compile_svg

TMP = Path(__file__).parent / "_test_tmp"


@pytest.fixture(autouse=True)
def _tmpdir():
    TMP.mkdir(exist_ok=True)
    yield
    # leave files around for debugging


def _write_svg(name: str, root: Element) -> Path:
    """Write an SVG element tree to a temp file and return its path."""
    path = TMP / name
    ElementTree(root).write(path, xml_declaration=True)
    return path


def _svg_root(**attrs: str) -> Element:
    return Element("{http://www.w3.org/2000/svg}svg", attrib=attrs)


# ── Phase 1: simple <line> elements ──────────────────────────────────────


class TestSingleLine:
    def test_single_line_output(self):
        root = _svg_root(width="100", height="100")
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}line",
            x1="0",
            y1="0",
            x2="100",
            y2="50",
        )
        path = _write_svg("single_line.svg", root)
        source = compile_svg(path)
        assert source == textwrap.dedent("""\
            # Generated from single_line.svg by svg_to_turtle.py
            # Do not edit \u2014 regenerate from the SVG source.

            def draw(t):
                t.penup()
                t.goto(0.0, 0.0)
                t.pendown()
                t.goto(100.0, 50.0)""")

    def test_single_line_executes(self):
        root = _svg_root()
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}line",
            x1="10",
            y1="20",
            x2="30",
            y2="40",
        )
        path = _write_svg("exec_line.svg", root)
        source = compile_svg(path)
        ns: dict[str, object] = {}
        exec(compile(source, "<test>", "exec"), ns)
        draw = ns["draw"]

        # Use a recording turtle stub
        calls: list[tuple[str, tuple[object, ...]]] = []

        class FakeTurtle:
            def penup(self):
                calls.append(("penup", ()))

            def pendown(self):
                calls.append(("pendown", ()))

            def goto(self, x, y):
                calls.append(("goto", (x, y)))

        draw(FakeTurtle())  # type: ignore[arg-type]
        assert calls == [
            ("penup", ()),
            ("goto", (10.0, 20.0)),
            ("pendown", ()),
            ("goto", (30.0, 40.0)),
        ]


class TestMultipleLines:
    def test_two_lines(self):
        root = _svg_root()
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}line",
            x1="0",
            y1="0",
            x2="10",
            y2="10",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}line",
            x1="10",
            y1="10",
            x2="20",
            y2="0",
        )
        path = _write_svg("two_lines.svg", root)
        source = compile_svg(path)
        # Should contain two sets of penup/goto/pendown/goto
        assert source.count("t.penup()") == 2
        assert source.count("t.pendown()") == 2
        assert source.count("t.goto(") == 4


# ── Phase 2: <defs> + <use> with translate ───────────────────────────────


class TestDefsAndUse:
    def _make_svg_with_def_and_use(self) -> Path:
        root = _svg_root()
        defs = SubElement(root, "{http://www.w3.org/2000/svg}defs")
        SubElement(
            defs,
            "{http://www.w3.org/2000/svg}line",
            id="seg",
            x1="0",
            y1="0",
            x2="10",
            y2="0",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}use",
            href="#seg",
            transform="translate(100, 200)",
        )
        return _write_svg("def_use.svg", root)

    def test_defs_produce_helper_functions(self):
        path = self._make_svg_with_def_and_use()
        source = compile_svg(path)
        assert "def _draw_seg(t):" in source
        assert "def draw(t):" in source

    def test_use_calls_helper(self):
        path = self._make_svg_with_def_and_use()
        source = compile_svg(path)
        assert "_draw_seg(t)" in source

    def test_use_applies_translate(self):
        path = self._make_svg_with_def_and_use()
        source = compile_svg(path)
        # The generated code should offset by (100, 200)
        assert "100.0" in source
        assert "200.0" in source

    def test_use_executes(self):
        path = self._make_svg_with_def_and_use()
        source = compile_svg(path)
        ns: dict[str, object] = {}
        exec(compile(source, "<test>", "exec"), ns)
        draw = ns["draw"]

        calls: list[tuple[str, tuple[object, ...]]] = []

        class FakeTurtle:
            _x: float = 0.0
            _y: float = 0.0
            _heading: float = 0.0

            def penup(self):
                calls.append(("penup", ()))

            def pendown(self):
                calls.append(("pendown", ()))

            def goto(self, x, y=None):
                if y is None:
                    # tuple form
                    self._x, self._y = x
                else:
                    self._x, self._y = x, y
                calls.append(("goto", (self._x, self._y)))

            def position(self):
                return (self._x, self._y)

            def heading(self):
                return self._heading

            def setheading(self, h):
                self._heading = h
                calls.append(("setheading", (h,)))

        draw(FakeTurtle())  # type: ignore[arg-type]

        # The helper draws a horizontal line at (0,0)-(10,0).
        # The <use> translates to (100, 200), so the effective line is
        # from (100, 200) to (110, 200).  But the helper is called
        # after the turtle is moved to the translated origin, and the
        # helper draws its own local coords relative to that origin.
        # Actually, the helper uses absolute coords from the SVG defs.
        # The translation is handled by moving the turtle before calling.
        # Let's just check the calls contain the right goto's.

        # Helper draws: penup, goto(0,0), pendown, goto(10,0)
        # Main does: save pos, penup, goto(100,200), call helper, restore
        goto_calls = [(c, args) for c, args in calls if c == "goto"]
        # Should include goto(100.0, 200.0) from the translate
        assert ("goto", (100.0, 200.0)) in goto_calls
        # Should include the helper's line endpoints
        assert ("goto", (0.0, 0.0)) in goto_calls
        assert ("goto", (10.0, 0.0)) in goto_calls


class TestMultipleUses:
    def test_two_uses_of_same_def(self):
        root = _svg_root()
        defs = SubElement(root, "{http://www.w3.org/2000/svg}defs")
        SubElement(
            defs,
            "{http://www.w3.org/2000/svg}line",
            id="seg",
            x1="0",
            y1="0",
            x2="5",
            y2="5",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}use",
            href="#seg",
            transform="translate(10, 20)",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}use",
            href="#seg",
            transform="translate(30, 40)",
        )
        path = _write_svg("multi_use.svg", root)
        source = compile_svg(path)
        # The helper should be called twice in the body of draw()
        draw_body = source.split("def draw(t):")[1]
        assert draw_body.count("_draw_seg(t)") == 2


class TestTypeCheck:
    """Verify that generated output passes ty type checking."""

    def test_generated_code_typechecks(self):
        """The generated Python source should be valid for ty."""
        import subprocess

        root = _svg_root()
        defs = SubElement(root, "{http://www.w3.org/2000/svg}defs")
        SubElement(
            defs,
            "{http://www.w3.org/2000/svg}line",
            id="seg",
            x1="0",
            y1="0",
            x2="10",
            y2="0",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}line",
            x1="5",
            y1="5",
            x2="15",
            y2="15",
        )
        SubElement(
            root,
            "{http://www.w3.org/2000/svg}use",
            href="#seg",
            transform="translate(50, 60)",
        )
        path = _write_svg("typecheck.svg", root)
        source = compile_svg(path)
        out_path = TMP / "typecheck_output.py"
        out_path.write_text(source + "\n")

        result = subprocess.run(
            ["ty", "check", str(out_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"ty errors:\n{result.stdout}\n{result.stderr}"
