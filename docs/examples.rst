========
Examples
========

SVG-to-Turtle compiler
======================

This example reads a small subset of SVG and compiles it into a Python
module that reproduces the drawing with the standard-library
:mod:`turtle` module.  It demonstrates several fluent-codegen features:

* Building a :class:`~fluent_codegen.codegen.Module` with multiple
  functions.
* Using the **E-object** system (``enames``, ``.e``) for natural-looking
  math and method calls.
* Automatic **name deduplication** — the same local name (``pos``,
  ``heading``) is used for each ``<use>`` element, and the scope
  manager appends suffixes automatically.

Supported SVG elements:

* ``<line x1=… y1=… x2=… y2=…>`` — a straight line segment.
* ``<defs>`` containing ``<line>`` elements with ``id`` attributes —
  compiled into helper functions.
* ``<use href="#id" transform="translate(tx, ty)">`` — compiled into
  code that repositions the turtle and calls the helper.

The compiler script
-------------------

.. literalinclude:: examples/svg_to_turtle/svg_to_turtle.py
   :language: python
   :caption: ``docs/examples/svg_to_turtle/svg_to_turtle.py``

Sample input
------------

A simple house shape built from ``<line>`` elements and ``<use>``
references to a reusable beam defined in ``<defs>``:

.. literalinclude:: examples/svg_to_turtle/house.svg
   :language: xml
   :caption: ``house.svg``

Generated output
----------------

Running ``python svg_to_turtle.py house.svg`` produces:

.. literalinclude:: examples/svg_to_turtle/house_output.py
   :language: python
   :caption: ``house.py`` (generated)

Key points to note in the generated code:

* ``_draw_beam(t)`` is a helper function compiled from the ``<defs>``
  element.
* Each ``<use>`` saves the turtle position, translates, calls the
  helper, and restores.
* The local variables ``pos``, ``heading`` are auto-suffixed to
  ``pos_2``, ``heading_2`` for the second ``<use>``.
