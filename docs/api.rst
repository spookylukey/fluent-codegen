API Reference
=============

.. module:: fluent_codegen.codegen

This page documents the complete public API of ``fluent_codegen.codegen``.


Top-level containers
--------------------

.. autoclass:: Module
   :members:
   :show-inheritance:

.. autoclass:: Block
   :members:
   :show-inheritance:


Scope and name management
-------------------------

.. autoclass:: Scope
   :members:

.. autoclass:: Name
   :members:
   :show-inheritance:


Functions and classes
---------------------

.. autoclass:: Function
   :members:
   :show-inheritance:

.. autoclass:: Class
   :members:
   :show-inheritance:

.. autoclass:: FunctionArg
   :members:

.. autoclass:: ArgKind
   :members:
   :undoc-members:


Statements
----------

.. autoclass:: Statement
   :members:
   :show-inheritance:

.. autoclass:: Assignment
   :members:
   :show-inheritance:

.. autoclass:: Annotation
   :members:
   :show-inheritance:

.. autoclass:: Return
   :members:
   :show-inheritance:

.. autoclass:: Assert
   :members:
   :show-inheritance:

.. autoclass:: If
   :members:
   :show-inheritance:

.. autoclass:: With
   :members:
   :show-inheritance:

.. autoclass:: Try
   :members:
   :show-inheritance:

.. autoclass:: Import
   :members:
   :show-inheritance:

.. autoclass:: ImportFrom
   :members:
   :show-inheritance:


Expressions
-----------

.. autoclass:: Expression
   :members:
   :show-inheritance:


Literal types
~~~~~~~~~~~~~

.. autoclass:: String
   :members:
   :show-inheritance:

.. autoclass:: Number
   :members:
   :show-inheritance:

.. autoclass:: Bool
   :members:
   :show-inheritance:

.. autoclass:: Bytes
   :members:
   :show-inheritance:

.. autoclass:: NoneExpr
   :members:
   :show-inheritance:

.. autoclass:: List
   :members:
   :show-inheritance:

.. autoclass:: Tuple
   :members:
   :show-inheritance:

.. autoclass:: Set
   :members:
   :show-inheritance:

.. autoclass:: Dict
   :members:
   :show-inheritance:


Access and call expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Attr
   :members:
   :show-inheritance:

.. autoclass:: Call
   :members:
   :show-inheritance:

.. autoclass:: Subscript
   :members:
   :show-inheritance:

.. autoclass:: Starred
   :members:
   :show-inheritance:


String joining
~~~~~~~~~~~~~~

.. autoclass:: StringJoinBase
   :members:
   :show-inheritance:

.. autoclass:: FStringJoin
   :members:
   :show-inheritance:

.. autoclass:: ConcatJoin
   :members:
   :show-inheritance:

.. data:: StringJoin

   Alias for :class:`FStringJoin` (the default string-join strategy).


Arithmetic operators
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Add
   :show-inheritance:

.. autoclass:: Sub
   :show-inheritance:

.. autoclass:: Mul
   :show-inheritance:

.. autoclass:: Div
   :show-inheritance:

.. autoclass:: FloorDiv
   :show-inheritance:

.. autoclass:: Mod
   :show-inheritance:

.. autoclass:: Pow
   :show-inheritance:

.. autoclass:: MatMul
   :show-inheritance:


Comparison operators
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Equals
   :show-inheritance:

.. autoclass:: NotEquals
   :show-inheritance:

.. autoclass:: Lt
   :show-inheritance:

.. autoclass:: Gt
   :show-inheritance:

.. autoclass:: LtE
   :show-inheritance:

.. autoclass:: GtE
   :show-inheritance:

.. autoclass:: In
   :show-inheritance:

.. autoclass:: NotIn
   :show-inheritance:


Boolean operators
~~~~~~~~~~~~~~~~~

.. autoclass:: And
   :show-inheritance:

.. autoclass:: Or
   :show-inheritance:


Base classes
~~~~~~~~~~~~

.. autoclass:: BinaryOperator
   :members:
   :show-inheritance:

.. autoclass:: ArithOp
   :members:
   :show-inheritance:

.. autoclass:: CompareOp
   :members:
   :show-inheritance:

.. autoclass:: BoolOp
   :members:
   :show-inheritance:


Utility functions
-----------------

.. autofunction:: auto

.. autoclass:: constants
   :members:
   :undoc-members:

.. autofunction:: function_call

.. autofunction:: method_call

.. autofunction:: cleanup_name

.. autofunction:: traverse

.. autofunction:: simplify

.. autofunction:: rewriting_traverse

.. autofunction:: morph_into

.. autodata:: SENSITIVE_FUNCTIONS


.. module:: fluent_codegen.remove_unused_assignments

Dead-code elimination
---------------------

.. autofunction:: remove_unused_assignments
