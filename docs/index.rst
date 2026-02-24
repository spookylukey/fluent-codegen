fluent-codegen
==============

**fluent-codegen** is a Python library for generating Python code via AST
construction.  Instead of manipulating raw ``ast`` nodes or concatenating
strings, you work with high-level objects like :class:`~fluent_codegen.codegen.Module`,
:class:`~fluent_codegen.codegen.Function`, and :class:`~fluent_codegen.codegen.Expression`
that produce correct Python source or compilable AST.

.. code-block:: python

   from fluent_codegen import codegen

   module = codegen.Module()
   func, func_name = module.create_function("greet", args=["name"])
   name = func.name("name")
   func.body.create_return(
       codegen.String("Hello, ").add(name).add(codegen.String("!"))
   )
   print(module.as_python_source())
   # def greet(name):
   #     return f'Hello, {name}!'

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
