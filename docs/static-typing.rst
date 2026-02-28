=============
Static typing
=============

fluent-codegen has been designed to work well static type checks to ensure that
you are passing appropriate objects to different methods. In general, static
type errors will indicate a misuse of the API, and it is assumed that you will
be using a static type checker. If you pass the wrong kind of object, such as
passing a string where an Expression is expected, often you will only get a
runtime error later in the process – at the point where you are trying to
convert to AST or Python source.

In terms of the Python code you are generating, fluent-codegen provides
little-to-no protection for type errors, or name errors resulting from
attributes that don’t exist. For example, you will get no warnings if you do:

.. code-block:: python

   mod.assign("x", codegen.Number(1).e + codegen.String("hello").e)  # x = 1 + "hello"

Or:

.. code-block:: python

   _, math_lib = mod.create_import('math')

   math_lib.e.my_math_func(1, 2)   # No such function in `math`


So you will need other ways to ensure and test that your output is correct.

In addition to unit tests, a useful technique is to run a static type checker on
the output. `ty <https://docs.astral.sh/ty/>`_ is a fast type checker that can
be useful for this purpose.
