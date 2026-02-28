===========
 E-objects
===========

The nature of a code generation library is that we want to write code where the
names of functions, variables or classes are not fully known ahead of time.
(Otherwise, we would just write a library!).

However, as well as dynamically created names, there will be significant amount
of code where the function calls, names and values are known and don’t depend on
the input data. For these cases, even with the “chaining” API on Expression, you
have to write lot of verbose and difficult to read code.

For example, if you wanted to calculate the hypotenuse of a triangle of width
``x`` and height ``y``, in Python it looks like this:

.. code-block:: python

   import math

   x = 3
   y = 4
   h = math.sqrt(x**2 + y**2)


To generate the above using fluent-codegen would require this:

.. code-block:: python

   mod = codegen.Module()
   _, math_lib = mod.create_import("math")
   x_name = mod.assign("x", codegen.auto(3))
   y_name = mod.assign("y", codegen.auto(4))
   h_name = mod.assign("h", math_lib.attr("sqrt").call([x_name.pow(codegen.auto(2)).add(y_name.pow(codegen.auto(2)))]))

The last line is particularly unreadable.

This is where E-objects come in. They allow you to use normal Python syntax for
this kind of code. You can create an E-object by using ``.e`` on any
``Expression``. The last line of the above code can now be re-written as
follows:

.. code-block:: python

   h_name = mod.assign("h", math_lib.e.sqrt(x_name.e**2 + y_name.e**2))


It’s important to know how this works: E-objects define methods like
``__getattr__`` and ``__add__`` etc. to override normal attribute access and
operators so that they all:

- automatically wrap arguments in ``auto()``, so that literals etc. are handled
  without ceremony.
- delegate to the appropriate methods on ``Expression`` to build the expression.
- return the result as an E-object.


Mixing types
============

It is often convenient to mix ``Expression`` and E-objects. This is supported by
various methods. Note the following:

- The “bottom layer” in the codegen module is the ``CodeGenAst`` subclasses,
  such as ``Name``, ``If``, ``Add`` etc. The constructors to these functions
  (which you don’t normally call directly) accept only ``Expression`` objects.

- The more convenient methods on ``Expression`` and ``Block``, such as the
  method chaining for building up up expressions, and utilities like
  ``Block.create_assignment`` and ``Block.assign``, all accept E-objects as well
  as ``Expression``. Similarly the methods on ``If`` for adding branches. This
  is the middle layer.

- The top layer is the E-object layer, and this allows you to freely mix not
  only E-objects and ``Expression``, but also simple Python objects and
  container objects. This also includes the ``auto()`` utility.

Note specifically that the middle layer doesn’t allow mixing in simple Python
objects. For example:

.. code-block:: python

   x = mod.assign("x", "y")

This is an error. To avoid confusion between strings and what they mean, you
have to be more explicit at this level. You could mean either:

.. code-block:: python
   x = mod.assign("x", mod.name("y"))  #  x = y

Or:

.. code-block:: python
   x = mod.assign("x", codegen.String("y"))  # x = "y"

If you need to explicitly convert from E-objects to ``Expression``, you can use
``Expression.from_e``.

TODO static typing.

Limitations
===========

No magic
--------

In general, E-objects aren’t magic, and work only by implementation of methods
that allow overriding how operators work. So, for example, the following will
work due to overriding the ``|`` operator:

.. code-block:: python

   mod.assign("y", name.e | {"key": "value"})

But the following will fail:

   mod.assign("y", {"key": "value"})

The above can be fixed by explicit use of ``auto()`` around the dict.

Unsupported operators
---------------------

Some operators cannot be overridden by implementing dunder methods:

- ``is`` and ``is not`` - these always do object identity in Python

- ``and`` and ``or`` are special short-circuiting operators in Python i.e.
  control structures. Similarly ``if/else``.

- ``in`` and ``not in`` membership operators can’t be supported.


Comparisons
-----------

Comparison operators ``==``, ``!=``, ``<`` etc do not always come out on the
“side” you expect, due to Python limitations. If you put an E-object on the
right and a Python object on the left, it comes out backwards:

.. code-block:: python

   x.e == 1   # produces `x == 1`
   1 == x.e   # produces `x == 1`

In addition, the second form will be deduced by type checkers to have type
``bool``, when in fact it produces an E-object. So the first form should be
preferred.
