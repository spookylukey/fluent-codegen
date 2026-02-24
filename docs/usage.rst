Usage Guide
===========

This guide explains the core concepts of **fluent-codegen** and walks through
progressively more complex examples.


Why fluent-codegen?
-------------------

When you need to generate Python source code programmatically, the obvious
approaches have drawbacks:

* **String concatenation / templates** — easy to produce syntactically broken
  code, hard to avoid injection bugs, and painful to maintain indentation.
* **Raw ``ast`` module** — correct by construction but extremely verbose; every
  node requires a half-dozen keyword arguments.

``fluent-codegen`` sits in between: it gives you a *simplified* AST that maps
closely to Python constructs, with a **fluent chaining API** for building
expressions.  You get correctness (it emits real ``ast`` nodes) without the
verbosity.


Core Concepts
-------------

The library is built around a small number of interacting concepts:

.. contents::
   :local:
   :depth: 1


Module — the top-level container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every code-generation session starts with a :class:`~fluent_codegen.codegen.Module`.
A Module is both a :class:`~fluent_codegen.codegen.Block` (a list of statements)
and a :class:`~fluent_codegen.codegen.Scope` (a namespace for names).

.. code-block:: python

   from fluent_codegen import codegen

   module = codegen.Module()

By default the module's scope pre-reserves all Python builtins, so you can
never accidentally shadow ``str``, ``len``, etc.

When you're done building, call:

* ``module.as_python_source()`` — get a string of Python source code.
* ``module.as_ast()`` — get a ``ast.Module`` node you can ``compile()`` and
  ``exec()``.


Scope — safe name management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fluent_codegen.codegen.Scope` tracks which names are in use in a given
namespace, and guarantees you never create clashing identifiers.

The two key methods are:

``scope.create_name(requested)``
    Reserve a name and return a :class:`~fluent_codegen.codegen.Name` expression.
    If the requested name is already taken, a numeric suffix is appended
    automatically (e.g. ``x``, ``x_2``, ``x_3``, …).

``scope.name(existing)``
    Return a :class:`~fluent_codegen.codegen.Name` for a name that is *already*
    reserved (raises if not).  Use this to refer to function parameters or
    previously created names.

.. code-block:: python

   scope = codegen.Scope()
   a = scope.create_name("x")       # Name("x")
   b = scope.create_name("x")       # Name("x_2") — auto-deduplicated
   c = scope.name("x")              # Name("x") — refers to existing

Since :class:`~fluent_codegen.codegen.Module`,
:class:`~fluent_codegen.codegen.Function`, and
:class:`~fluent_codegen.codegen.Class` all inherit from
:class:`~fluent_codegen.codegen.Scope`, you typically call ``create_name`` on
those directly rather than on a bare ``Scope``.


Block — a sequence of statements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fluent_codegen.codegen.Block` is an ordered list of statements.  A
Module's body, a Function's body, and each branch of an If are all Blocks.

Blocks expose ``create_*`` factory methods that simultaneously create a
statement or sub-structure, add it to the block, and (where relevant) register
names in the scope:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Creates
   * - ``create_function(name, args)``
     - A :class:`~fluent_codegen.codegen.Function` definition
   * - ``create_class(name, bases)``
     - A :class:`~fluent_codegen.codegen.Class` definition
   * - ``create_assignment(name, value)``
     - ``name = value``
   * - ``create_annotation(name, type)``
     - ``name: type`` (bare annotation)
   * - ``create_field(name, type, default=…)``
     - Typed field (annotation ± default), useful in dataclasses
   * - ``create_import(module)``
     - ``import module``
   * - ``create_import_from(from_=…, import_=…)``
     - ``from module import name``
   * - ``create_if()``
     - An :class:`~fluent_codegen.codegen.If` statement
   * - ``create_with(expr, target)``
     - A ``with`` statement
   * - ``create_return(value)``
     - ``return value``
   * - ``create_assert(test, msg)``
     - ``assert test, msg``
   * - ``add_comment(text)``
     - A ``# text`` comment line
   * - ``add_statement(stmt)``
     - Any :class:`~fluent_codegen.codegen.Statement` or :class:`~fluent_codegen.codegen.Expression`

These factory methods are the **recommended way** to build code.  They handle
scope registration and validation for you.


Name — the bridge between Scope and Expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fluent_codegen.codegen.Name` is the central connecting piece.  It is an
:class:`~fluent_codegen.codegen.Expression`, so you can chain further operations
on it (call, attribute access, arithmetic, etc.).  But it can only be created
through a :class:`~fluent_codegen.codegen.Scope`, which guarantees the name is
defined.

.. code-block:: python

   module = codegen.Module()
   func, func_name = module.create_function("add", args=["a", "b"])

   # func_name is a Name for "add" in the module scope.
   # func is the Function object (which is also a Scope).

   a = func.name("a")   # Name for the parameter
   b = func.name("b")

   func.body.create_return(a.add(b))   # return a + b

The pattern of ``create_*`` returning a ``(thing, Name)`` tuple is pervasive:
you hold onto the ``Name`` so you can call or reference the created entity
later.


Expression — the fluent chaining API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fluent_codegen.codegen.Expression` is the base class for all
value-producing nodes.  Every Expression exposes chainable methods that produce
new Expressions:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Produces
     - Python equivalent
   * - ``.call(args, kwargs)``
     - :class:`~fluent_codegen.codegen.Call`
     - ``expr(a, b, k=v)``
   * - ``.attr(name)``
     - :class:`~fluent_codegen.codegen.Attr`
     - ``expr.name``
   * - ``.method_call(name, args, kwargs)``
     - :class:`~fluent_codegen.codegen.Call`
     - ``expr.name(a, b)``
   * - ``.subscript(index)``
     - :class:`~fluent_codegen.codegen.Subscript`
     - ``expr[index]``
   * - ``.add(other)``
     - :class:`~fluent_codegen.codegen.Add`
     - ``expr + other``
   * - ``.sub(other)``
     - :class:`~fluent_codegen.codegen.Sub`
     - ``expr - other``
   * - ``.mul(other)``
     - :class:`~fluent_codegen.codegen.Mul`
     - ``expr * other``
   * - ``.div(other)``
     - :class:`~fluent_codegen.codegen.Div`
     - ``expr / other``
   * - ``.mod(other)``
     - :class:`~fluent_codegen.codegen.Mod`
     - ``expr % other``
   * - ``.eq(other)``
     - :class:`~fluent_codegen.codegen.Equals`
     - ``expr == other``
   * - ``.ne(other)``
     - :class:`~fluent_codegen.codegen.NotEquals`
     - ``expr != other``
   * - ``.lt(other)``, ``.gt(other)``, ``.le(other)``, ``.ge(other)``
     - Comparisons
     - ``<``, ``>``, ``<=``, ``>=``
   * - ``.and_(other)``, ``.or_(other)``
     - Boolean ops
     - ``and``, ``or``
   * - ``.in_(other)``, ``.not_in(other)``
     - Membership tests
     - ``in``, ``not in``
   * - ``.starred()``
     - :class:`~fluent_codegen.codegen.Starred`
     - ``*expr``

Because every method returns a new Expression, you can chain them fluently:

.. code-block:: python

   # Generates: result.encode('utf-8').decode('ascii')
   result.method_call("encode", [codegen.String("utf-8")]) \
         .method_call("decode", [codegen.String("ascii")])


Literal values
~~~~~~~~~~~~~~

The library provides Expression subclasses for all Python literal types:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Example
   * - :class:`~fluent_codegen.codegen.String`
     - ``codegen.String("hello")``
   * - :class:`~fluent_codegen.codegen.Number`
     - ``codegen.Number(42)`` or ``codegen.Number(3.14)``
   * - :class:`~fluent_codegen.codegen.Bool`
     - ``codegen.Bool(True)``
   * - :class:`~fluent_codegen.codegen.Bytes`
     - ``codegen.Bytes(b"data")``
   * - :class:`~fluent_codegen.codegen.List`
     - ``codegen.List([codegen.Number(1), codegen.Number(2)])``
   * - :class:`~fluent_codegen.codegen.Tuple`
     - ``codegen.Tuple([codegen.String("a"), codegen.String("b")])``
   * - :class:`~fluent_codegen.codegen.Set`
     - ``codegen.Set([codegen.Number(1)])``
   * - :class:`~fluent_codegen.codegen.Dict`
     - ``codegen.Dict([(codegen.String("k"), codegen.Number(1))])``
   * - :class:`~fluent_codegen.codegen.NoneExpr`
     - ``codegen.NoneExpr()``

For convenience, the :func:`~fluent_codegen.codegen.auto` function converts a
plain Python value into the appropriate Expression:

.. code-block:: python

   codegen.auto(42)         # Number(42)
   codegen.auto("hello")    # String("hello")
   codegen.auto(None)       # NoneExpr()
   codegen.auto([1, 2, 3])  # List([Number(1), Number(2), Number(3)])

Pre-made constants are available as ``codegen.constants.None_``,
``codegen.constants.True_``, and ``codegen.constants.False_``.


Worked Examples
---------------

Hello World — a simple function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from fluent_codegen import codegen

   module = codegen.Module()
   func, func_name = module.create_function("hello", args=["name"])
   name = func.name("name")

   greeting = codegen.FStringJoin.build([
       codegen.String("Hello, "),
       name,
       codegen.String("!"),
   ])
   func.body.create_return(greeting)

   print(module.as_python_source())

Output::

   def hello(name):
       return f'Hello, {name}!'


Creating names and calling them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common pattern is to create a name (for a function, variable, or import) and
then call or reference it later:

.. code-block:: python

   module = codegen.Module()

   # Import a module and hold the Name
   _, json_name = module.create_import("json")

   func, _ = module.create_function("serialize", args=["data"])
   data = func.name("data")

   # Use the held Name to call json.dumps(data)
   result = json_name.attr("dumps").call([data])
   func.body.create_return(result)

   print(module.as_python_source())

Output::

   import json
   def serialize(data):
       return json.dumps(data)

The same pattern works with ``create_function``, ``create_class``, and
``create_import_from`` — all return a ``(object, Name)`` tuple.


Assignments and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a local variable, first reserve a name in the scope, then assign to
it:

.. code-block:: python

   module = codegen.Module()
   func, _ = module.create_function("compute", args=["x"])
   x = func.name("x")

   # Reserve the name "result" in the function scope
   result_name = func.create_name("result")

   # Assign: result = x * 2
   func.body.create_assignment(result_name, x.mul(codegen.Number(2)))

   # Return: return result + 1
   func.body.create_return(result_name.add(codegen.Number(1)))

   print(module.as_python_source())

Output::

   def compute(x):
       result = x * 2
       return result + 1


Classes and decorators
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   module = codegen.Module()
   _, dc = module.create_import_from(from_="dataclasses", import_="dataclass")

   cls, cls_name = module.create_class(
       "Point",
       decorators=[dc],
   )

   cls.body.create_field("x", codegen.Name("float", module.scope))
   cls.body.create_field("y", codegen.Name("float", module.scope))
   cls.body.create_field("z", codegen.Name("float", module.scope),
                         default=codegen.Number(0.0))

   print(module.as_python_source())

Output::

   from dataclasses import dataclass
   @dataclass
   class Point:
       x: float
       y: float
       z: float = 0.0


Control flow — if / elif / else
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~fluent_codegen.codegen.If` is built incrementally with
``create_if_branch``:

.. code-block:: python

   module = codegen.Module()
   func, _ = module.create_function("classify", args=["n"])
   n = func.name("n")

   if_stmt = func.body.create_if()

   # if n > 0:
   pos = if_stmt.create_if_branch(n.gt(codegen.Number(0)))
   pos.create_return(codegen.String("positive"))

   # elif n < 0:
   neg = if_stmt.create_if_branch(n.lt(codegen.Number(0)))
   neg.create_return(codegen.String("negative"))

   # else:
   if_stmt.else_block.create_return(codegen.String("zero"))

   print(module.as_python_source())

Output::

   def classify(n):
       if n > 0:
           return 'positive'
       elif n < 0:
           return 'negative'
       else:
           return 'zero'


With statements
~~~~~~~~~~~~~~~~

.. code-block:: python

   module = codegen.Module()
   func, _ = module.create_function("read_file", args=["path"])
   path = func.name("path")

   f_name = func.create_name("f")
   with_stmt = func.body.create_with(
       module.scope.name("open").call([path]),
       target=f_name,
   )
   with_stmt.body.create_return(f_name.method_call("read"))

   print(module.as_python_source())

Output::

   def read_file(path):
       with open(path) as f:
           return f.read()


Function arguments — positional, keyword, defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For simple cases, pass argument names as strings.  For finer control use
:class:`~fluent_codegen.codegen.FunctionArg`:

.. code-block:: python

   from fluent_codegen.codegen import FunctionArg

   module = codegen.Module()
   func, _ = module.create_function("connect", args=[
       FunctionArg.positional("host"),
       FunctionArg.positional("port", default=codegen.Number(5432)),
       FunctionArg.keyword("timeout", default=codegen.Number(30)),
       FunctionArg.keyword("ssl", default=codegen.constants.False_),
   ])

   print(module.as_python_source())

Output::

   def connect(host, port=5432, /, *, timeout=30, ssl=False):
       pass


Imports
~~~~~~~

.. code-block:: python

   module = codegen.Module()

   # import os
   _, os_name = module.create_import("os")

   # import numpy as np
   _, np_name = module.create_import("numpy", as_="np")

   # from pathlib import Path
   _, path_cls = module.create_import_from(from_="pathlib", import_="Path")

   # from collections import OrderedDict as OD
   _, od_name = module.create_import_from(
       from_="collections", import_="OrderedDict", as_="OD"
   )

Each call returns the statement and a :class:`~fluent_codegen.codegen.Name` that
you use to reference the imported entity.


String joining / f-strings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~fluent_codegen.codegen.FStringJoin` (the default
:class:`~fluent_codegen.codegen.StringJoin`) or
:class:`~fluent_codegen.codegen.ConcatJoin` to build dynamic strings:

.. code-block:: python

   greeting = codegen.FStringJoin.build([
       codegen.String("Hello, "),
       name,
       codegen.String("! You have "),
       count.method_call("__str__"),  # or any expression
       codegen.String(" items."),
   ])
   # Generates: f'Hello, {name}! You have {count.__str__()} items.'

``build()`` is smart: it merges adjacent ``String`` literals and simplifies
down to a plain ``String`` when possible.


Comments
~~~~~~~~

Add comments to any block:

.. code-block:: python

   module.add_comment("Auto-generated — do not edit.")
   module.add_comment(
       "This is a long comment that will be wrapped nicely.",
       wrap=72,
   )

Comments appear in the output of ``as_python_source()`` as ``# …`` lines.


Compiling and executing generated code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   module = codegen.Module()
   func, func_name = module.create_function("double", args=["n"])
   n = func.name("n")
   func.body.create_return(n.mul(codegen.Number(2)))

   # Compile to a code object
   code = compile(module.as_ast(), "<generated>", "exec")

   # Execute into a namespace
   ns: dict[str, object] = {}
   exec(code, ns)

   # Call the generated function
   assert ns["double"](21) == 42


Advanced Topics
---------------

Scope nesting and name safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scopes form a parent-child chain.  When a :class:`~fluent_codegen.codegen.Function`
is created inside a Module, the function's scope has the module's scope as its
parent.  ``create_name`` checks *all* ancestor scopes to avoid shadowing:

.. code-block:: python

   module = codegen.Module()
   x = module.scope.create_name("x")

   func, _ = module.create_function("f", args=[])
   y = func.create_name("x")  # Gets "x_2" because "x" is taken in parent

This prevents accidental variable shadowing in generated code.


Removing unused assignments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~fluent_codegen.remove_unused_assignments.remove_unused_assignments`
utility performs dead-code elimination on a function body:

.. code-block:: python

   from fluent_codegen.remove_unused_assignments import remove_unused_assignments

   # After building a function with potentially unused variables:
   remove_unused_assignments(func)

This iteratively removes assignments whose targets are never read, which is
useful when your code-generation logic creates variables speculatively.


AST traversal and rewriting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~fluent_codegen.codegen.rewriting_traverse` function walks the
codegen AST and applies a rewriter function to every node, allowing
post-processing transformations:

.. code-block:: python

   def my_rewriter(node):
       # Transform nodes as needed
       return node

   codegen.rewriting_traverse(module, my_rewriter)

The higher-level :func:`~fluent_codegen.codegen.simplify` repeatedly applies
a simplifier until no more changes occur — useful for optimization passes.


Security considerations
~~~~~~~~~~~~~~~~~~~~~~~~

fluent-codegen is designed with security in mind:

* It builds AST, not strings, so code-injection via string interpolation is
  impossible.
* A set of :data:`~fluent_codegen.codegen.SENSITIVE_FUNCTIONS` (``exec``,
  ``eval``, ``compile``, ``open``, etc.) is blocked — attempting to call them
  raises ``AssertionError``.
* Assertions throughout ``as_ast()`` methods validate that names are legal
  Python identifiers.

These are defence-in-depth measures. Your higher-level code should also
validate inputs before passing them to the code generator.
