fluent-codegen
==============

A Python library for generating Python code via AST construction.

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

.. code:: python

   from fluent_codegen import codegen

   # Create a module with a function
   module = codegen.Module()
   func_name = module.scope.reserve_name("greet")
   func = codegen.Function(func_name, args=["name"], parent_scope=module.scope)
   func.add_return(
       codegen.StringJoin.build([
           codegen.String("Hello, "),
           codegen.VariableReference("name", func),
           codegen.String("!"),
       ])
   )
   module.add_function(func_name, func)

   # Compile and execute
   import ast, compile
   code = compile(ast.fix_missing_locations(module.as_ast()), "<generated>", "exec")
   namespace = {}
   exec(code, namespace)
   print(namespace["greet"]("World"))  # Hello, World!

License
-------

Apache License 2.0
