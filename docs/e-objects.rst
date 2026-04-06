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

- automatically wrap arguments in :func:`~fluent_codegen.codegen.auto`, so that
  literals etc. are handled without ceremony.
- delegate to the appropriate methods on ``Expression`` to build the expression.
- return the result as an E-object.

Enames
======

As a shortcut to getting hold of an E-object, you can use the ``enames``
property which is available on ``Scope`` and ``Module``:

.. code-block:: python

   mod = codegen.Module()
   func, func_name = mod.create_function('inc', ['val'])
   func.create_return(func.enames.val + 1)

   # Output code:
   #
   #    def inc(val):
   #        return val + 1

In the above code, ``func.enames.val`` is equivalent to ``func.name('val').e``.
Like the ``Scope.name()`` method, it will raise an error of you attempt to get a
name that has not been reserved.

This is also a convenient way to get hold of builtins that are already
registered as names in the ``Module`` scope (and inherited by other scopes that
are added to ``Module`` objects):

.. code-block:: python

   mod.enames.str(1)   #  Outputs `str(1)`

   # Equivalent to:
   #
   #  mod.scope.name('str').e(1)
   #
   # or the long method chaining version:
   #
   #  mod.scope.name('str').call([codegen.Number(1)]).e

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
  as ``Expression``. Similarly the methods on ``If`` and ``Try_`` for adding
  branches. This is the middle layer.

- The top layer is the E-object layer, and this allows you to freely mix not
  only E-objects and ``Expression``, but also simple Python objects and
  container objects. This also includes the :func:`~fluent_codegen.codegen.auto`
  utility.

Note specifically that the middle layer doesn’t allow mixing in simple Python
objects. For example:

.. code-block:: python

   x = mod.assign("x", "y")

This is an error (both at runtime and if you use a static type checker) . To
avoid confusion between strings and what they mean, you have to be more explicit
at this level. You could mean either:

.. code-block:: python

   x = mod.assign("x", mod.name("y"))  #  x = y

Or:

.. code-block:: python

   x = mod.assign("x", codegen.String("y"))  # x = "y"


Similarly there would be ambiguity over the difference between ``None`` and
``auto(None)`` if the middle layer functions automatically wrapped Python
objects with :func:`~fluent_codegen.codegen.auto`.

If you need to explicitly convert from E-objects to ``Expression``, you can use
``Expression.from_e``.

Note that E-objects and Expression objects are very different, and while you can
mix them in some of the calls, you cannot use an E-object as if it were an
``Expression``. For example, if you have an ``Expression``, you can use
``.attr("foo")`` to generate an attribute access to the ``.foo`` attribute. With
an E-object, however, ``.attr("foo")`` will generate a method call to
``.attr("foo")``! For this reason, it can be helpful to keep the ``.e`` call
visible in your code, or use a name convention like ``…_e`` to remind you that an
object is an E-object, not an ``Expression``.



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

.. code-block:: python

   mod.assign("y", {"key": "value"})

The above can be fixed by explicit use of :func:`~fluent_codegen.codegen.auto` around the dict.

Static typing issues
--------------------

E-objects allow you to write code that looks like normal Python code, and while
your static type checker is checking it, it is doing so in a very different way
to normal Python.

For example, take the following Python code:


.. code-block:: python

   import math
   import decimal

   print(math.sqrt("2"))
   print(math.arctan(decimal.Decimal(1)))


A type checker will immediately tell you that ``"2"`` is an invalid input to the
``sqrt()`` function, and that ``arctan()`` doesn’t exist at all. But consider the
equivalent using E-objects:

.. code-block:: python

   mod = codegen.Module()

   _, math_lib = mod.create_import("math")
   _, decimal_lib = mod.create_import("decimal")

   mod.enames.print(math_lib.e.sqrt("2"))
   mod.enames.print(math_lib.e.arctan(decimal_lib.e.Decimal(1)))


This looks similar, but gives no type check errors. All the objects involved are
correctly inferred as having type ``E``, and ``E`` objects allows any method to be
called with any ``E`` objects as values.

This is exactly the same as the equivalent without E-objects:

.. code-block:: python

   mod.scope.name("print").call([math_lib.attr("sqrt").call([codegen.String("2")])])

However, the use of strings like ``"print"`` and ``"sqrt"`` may make you more
aware of the code generation and the limited nature of type checking, while
E-objects can make you think there is more checking than there really is.

You do still get type checking for incorrect usage of the E-object API itself.
For example, if you do this:

.. code-block:: python

   import decimal

   mod.enames.print(math_lib.e.sqrt(decimal.Decimal(1)))

…you will get an error (something like “Argument of type "Decimal" cannot be
assigned to parameter "args" of type "ELike" in function "__call__"”), informing
you that ``Decimal`` objects can’t be auto-converted to ``Expression``, unlike
integers and floats.


Unsupported operators
---------------------

Some operators cannot be overridden by implementing dunder methods:

- ``is`` and ``is not`` - these always do object identity in Python

- ``and`` and ``or`` are special short-circuiting operators in Python i.e.
  control structures. Similarly ``if/else``.

- ``in`` and ``not in`` membership operators can’t be supported.

- star-unpacking, like ``foo(*x)`` or ``foo(**x)``, is not supported (and will
  produce an infinite loop if you attempt it…)

For these, you need to fall back to converting to ``Expression`` with
``Expression.from_e``, and using method chaining.

Comparisons
-----------

Comparison operators ``==``, ``!=``, ``<`` etc do not always come out on the
“side” you expect, due to Python limitations. If you put an E-object on the
right and a Python object on the left, it comes out backwards:

.. code-block:: python

   x.e == 1   #      produces `x == 1`
   1 == x.e   # also produces `x == 1`

In addition, the second form will be deduced by type checkers to have type
``bool``, when in fact it produces an E-object. So the first form should be
preferred.
