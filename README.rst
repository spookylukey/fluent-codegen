fluent-codegen
==============

A Python library for generating Python code via AST construction.

`Documentation <https://fluent-codegen.readthedocs.io/en/latest/>`__

Overview
--------

``fluent-codegen`` provides a set of classes that represent simplified Python
constructs (functions, assignments, expressions, control flow, etc.) and can
generate real Python `ast <https://docs.python.org/3/library/ast.html>`_ nodes.
This lets you build correct Python code programmatically without manipulating
raw AST or worrying about string interpolation pitfalls.

Originally extracted from
`fluent-compiler <https://github.com/django-ftl/fluent-compiler>`__,
where it was used to compile Fluent localization files into Python
bytecode.

Key features
------------

-  **Safe by construction** — builds AST, not strings, eliminating
   injection bugs
-  **Scope management** — automatic name deduplication and scope
   tracking
-  **Simplified API** — high-level classes (``Function``, ``If``,
   ``Try``, ``StringJoin``, etc.) that map to Python constructs without
   requiring knowledge of the raw ``ast`` module, plus two levels
   of helpers for building up expressions:

   - a `chaining API on “Expression” nodes
     <https://fluent-codegen.readthedocs.io/en/latest/usage.html#expression-the-fluent-chaining-api>`_

   - the `E-objects system for using something closer to Python syntax
     <https://fluent-codegen.readthedocs.io/en/latest/e-objects.html>`_

-  **Security guardrails** — blocks calls to sensitive builtins
   (``exec``, ``eval``, etc.)

Installation
------------

.. code:: bash

   pip install fluent-codegen

Requires Python 3.12+.

Quick example
-------------

This builds a FizzBuzz function entirely via the codegen API, using
fluent method-chaining for expressions:

.. code:: python

   from fluent_codegen import codegen

   # 1. Create a module and a function inside it
   module = codegen.Module()
   func, _ = module.create_function("fizzbuzz", args=["n"])

   # 2. A Name reference to the "n" parameter (Function *is* a Scope)
   n = func.name("n")

   # 3. Build an if / elif / else chain
   if_stmt = func.body.create_if()

   #    if n % 15 == 0: return "FizzBuzz"   — fluent chaining
   branch = if_stmt.create_if_branch(n.mod(codegen.Number(15)).eq(codegen.Number(0)))
   branch.create_return(codegen.String("FizzBuzz"))

   #    elif n % 3 == 0: return "Fizz"
   branch = if_stmt.create_if_branch(n.mod(codegen.Number(3)).eq(codegen.Number(0)))
   branch.create_return(codegen.String("Fizz"))

   #    elif n % 5 == 0: return "Buzz"
   branch = if_stmt.create_if_branch(n.mod(codegen.Number(5)).eq(codegen.Number(0)))
   branch.create_return(codegen.String("Buzz"))

   #    else: return str(n)
   if_stmt.else_block.create_return(module.scope.name("str").call([n]))

   # 4. Inspect the generated source
   print(module.as_python_source())
   # def fizzbuzz(n):
   #     if n % 15 == 0:
   #         return 'FizzBuzz'
   #     elif n % 3 == 0:
   #         return 'Fizz'
   #     elif n % 5 == 0:
   #         return 'Buzz'
   #     else:
   #         return str(n)

   # 5. Compile, execute, and call the generated function
   code = compile(module.as_ast(), "<fizzbuzz>", "exec")
   ns: dict[str, object] = {}
   exec(code, ns)
   fizzbuzz = ns["fizzbuzz"]
   assert fizzbuzz(15) == "FizzBuzz"
   assert fizzbuzz(9)  == "Fizz"
   assert fizzbuzz(10) == "Buzz"
   assert fizzbuzz(7)  == "7"

Even simpler with E-objects
---------------------------

The example above uses the **method-chaining** API (``n.mod(...).eq(...)``),
which maps one-to-one to AST nodes. For expression-heavy code, where you know
the names of functions/methods/attributes statically, the **E-object** API lets
you use normal Python operators instead — the library intercepts them and builds
the AST for you.

Here's the same FizzBuzz with E-objects:

.. code:: python

   from fluent_codegen import codegen

   module = codegen.Module()
   func, _ = module.create_function("fizzbuzz", args=["n"])
   n = func.name("n")

   if_stmt = func.body.create_if()

   # n.e enters "E-object mode" — then % and == are Python operators
   branch = if_stmt.create_if_branch(n.e % 15 == 0)
   branch.create_return(codegen.String("FizzBuzz"))

   branch = if_stmt.create_if_branch(n.e % 3 == 0)
   branch.create_return(codegen.String("Fizz"))

   branch = if_stmt.create_if_branch(n.e % 5 == 0)
   branch.create_return(codegen.String("Buzz"))

   # Convenient access to builtins as E-objects via `Scope.enames`
   if_stmt.else_block.create_return(module.enames.str(n))

The generated output is identical. The key difference is readability:
``n.e % 15 == 0`` vs ``n.mod(codegen.Number(15)).eq(codegen.Number(0))``.

E-objects really shine for math-heavy expressions:

.. code:: python

   module = codegen.Module()
   _, math_lib = module.create_import("math")
   func, _ = module.create_function("distance", args=["x", "y"])
   x = func.name("x")
   y = func.name("y")

   # E-object — reads like the code it generates
   func.body.create_return(math_lib.e.sqrt(x.e ** 2 + y.e ** 2))

   print(module.as_python_source())
   # import math
   # def distance(x, y):
   #     return math.sqrt(x ** 2 + y ** 2)

Compare with the equivalent method-chaining version:

.. code:: python

   func.body.create_return(
       math_lib.attr("sqrt").call([
           x.pow(codegen.Number(2)).add(y.pow(codegen.Number(2)))
       ])
   )


License
-------

Apache License 2.0

AI/LLM usage
------------

This project has been lovingly created by a human! The original API design came
from fluent-compiler and was all written by hand.

Since then, it has been turned into a separate project with considerable help
from a coding agent. The agent was used for a variety of tasks, including:

- extracting the code from fluent-compiler and project “scaffolding” work
- adding documentation
- continuing the existing patterns to cover more Python syntax
- researching which common Python syntax the project did not yet cover
- improving test coverage

At the same time, the code created was checked carefully and a lot of work was
done without the agent to ensure the API worked exactly the way I wanted.
Commits created mainly by the agent are obvious in the git history.

For external contributions, I prefer either code written by a human, or a
request written by a human, since it is easier for me to prompt the coding agent
according to my own standards than to review the results of someone else
prompting an AI.
