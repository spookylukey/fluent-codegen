Changelog
=========

Main additions and breaking changes listed here.

0.5.0 - 2026-03-11
------------------

Additions:

- Added ``For`` loop support with ``Block.create_for()``
- Added ``Break`` and ``Continue`` statements with ``Block.create_break()`` / ``Block.create_continue()``
- Added ``Raise`` statement with ``Block.create_raise()``
- Added ``Block.create_try()`` convenience method
- Added ``Slice`` AST node with :func:`~fluent_codegen.codegen.auto` support
- Added support for multiple ``except`` clauses, ``except ... as``, and ``finally`` on ``Try``
- ``create_for``, ``create_except``, and ``create_with`` now accept plain strings where ``Name`` / target was required

Breaking changes:

- Redesigned ``Try``: removed ``catch_exceptions`` from ``__init__``; except clauses are now added via ``Try.create_except()``
- ``Block.create_try()`` no longer takes arguments (add except clauses on the returned ``Try``)
- ``Block.create_with`` returns ``(with_statement, target)`` as output.
- Removed ``has_assignment_for_name``

0.4.0 - 2026-03-02
------------------

- Added E-object system

0.3.0 - 2026-02-24
------------------

