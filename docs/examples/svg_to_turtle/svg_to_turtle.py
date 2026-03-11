#!/usr/bin/env python3
"""SVG-to-Turtle compiler.

Reads a subset of SVG (straight-line segments) and emits a Python module
that reproduces the drawing using the standard-library ``turtle`` module.

This is an example of using **fluent-codegen** to compile one language
into Python.

Supported SVG elements
----------------------
* ``<line x1=… y1=… x2=… y2=…>``  – a single line segment
* ``<defs>`` containing ``<line>`` elements with ``id`` attributes
* ``<use href="#id" transform="translate(tx, ty)">``

Usage::

    python svg_to_turtle.py drawing.svg

Produces ``drawing.py`` with a ``draw(t)`` function that accepts a
``turtle.Turtle``.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fluent_codegen import codegen

# ---------------------------------------------------------------------------
# SVG parsing helpers
# ---------------------------------------------------------------------------

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _float(element: ET.Element, attr: str) -> float:
    """Extract a float attribute from an SVG element."""
    return float(element.attrib[attr])


def _parse_translate(transform: str) -> tuple[float, float]:
    """Parse ``translate(tx, ty)`` or ``translate(tx ty)`` into floats."""
    m = re.match(r"translate\(\s*([^,\s]+)[,\s]+([^)]+)\)", transform)
    if not m:
        raise ValueError(f"Unsupported transform: {transform!r}")
    return float(m.group(1)), float(m.group(2))


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _emit_line(
    block: codegen.Block,
    t: codegen.E,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> None:
    """Emit turtle commands to draw a single line from (x1,y1) to (x2,y2)."""
    block.add_statement(t.penup())
    block.add_statement(t.goto(x1, y1))
    block.add_statement(t.pendown())
    block.add_statement(t.goto(x2, y2))


def compile_svg(svg_path: str | Path) -> str:
    """Compile an SVG file into Python source code."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    module = codegen.Module()
    module.add_comment(f"Generated from {Path(svg_path).name} by svg_to_turtle.py")
    module.add_comment("Do not edit — regenerate from the SVG source.")

    # We'll use the E-object for the turtle parameter throughout.
    # First, collect <defs> definitions (Phase 2).
    defs: dict[str, ET.Element] = {}
    for defs_elem in root.iter(f"{{{SVG_NS}}}defs"):
        for child in defs_elem:
            tag = child.tag.removeprefix(f"{{{SVG_NS}}}")
            elem_id = child.get("id")
            if elem_id and tag == "line":
                defs[elem_id] = child

    # Create a helper function for each definition.
    def_func_names: dict[str, codegen.Name] = {}
    for def_id, elem in defs.items():
        func, func_name = module.create_function(f"_draw_{def_id}", args=["t"])
        t_e = func.enames.t
        _emit_line(
            func.body,
            t_e,
            _float(elem, "x1"),
            _float(elem, "y1"),
            _float(elem, "x2"),
            _float(elem, "y2"),
        )
        def_func_names[def_id] = func_name

    # Main draw function.
    draw_func, _ = module.create_function("draw", args=["t"])
    t_e = draw_func.enames.t

    for child in root:
        tag = child.tag.removeprefix(f"{{{SVG_NS}}}")

        if tag == "defs":
            continue  # already handled above

        if tag == "line":
            _emit_line(
                draw_func.body,
                t_e,
                _float(child, "x1"),
                _float(child, "y1"),
                _float(child, "x2"),
                _float(child, "y2"),
            )

        elif tag == "use":
            href = child.get("href") or child.get(f"{{{XLINK_NS}}}href") or ""
            ref_id = href.lstrip("#")
            if ref_id not in def_func_names:
                raise ValueError(f"<use> references unknown id {ref_id!r}")

            transform = child.get("transform", "")
            tx, ty = _parse_translate(transform) if transform else (0.0, 0.0)

            # Save turtle state, translate, call helper, restore.
            pos = draw_func.body.assign("pos", t_e.position())
            heading = draw_func.body.assign("heading", t_e.heading())

            # Move to translated origin
            draw_func.body.add_statement(t_e.penup())
            draw_func.body.add_statement(t_e.goto(pos.e[0] + tx, pos.e[1] + ty))

            # Call the helper
            draw_func.body.add_statement(def_func_names[ref_id].e(draw_func.enames.t))

            # Restore
            draw_func.body.add_statement(t_e.penup())
            draw_func.body.add_statement(t_e.goto(pos.e))
            draw_func.body.add_statement(t_e.setheading(heading.e))

    return module.as_python_source()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.svg>", file=sys.stderr)
        sys.exit(1)

    svg_path = Path(sys.argv[1])
    if not svg_path.exists():
        print(f"File not found: {svg_path}", file=sys.stderr)
        sys.exit(1)

    output = svg_path.with_suffix(".py")
    source = compile_svg(svg_path)
    output.write_text(source + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
