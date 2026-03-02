---------------
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
useful when your code-generation logic creates variables speculatively. It is
only safe to use if the statements that create the assignments have no side effects


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
