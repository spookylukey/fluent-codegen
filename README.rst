fluent-codegen
==============

A Python library for generating Python code via AST construction.

`Documentation <https://fluent-codegen.readthedocs.io/en/latest/>`__

Overview
--------

``fluent-codegen`` provides a set of classes that represent simplified
Python constructs (functions, assignments, expressions, control flow,
etc.) and can generate real Python ``ast`` nodes. This lets you build
correct Python code programmatically without manipulating raw AST or
worrying about string interpolation pitfalls.

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
   requiring knowledge of the raw ``ast`` module
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

License
-------

Apache License 2.0
