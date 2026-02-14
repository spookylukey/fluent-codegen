# fluent-codegen Migration Plan

Migrate from `fluent-compiler` (Fluent localization compiler) to `fluent-codegen` (general-purpose Python code generation library), keeping only the codegen module and its dependencies.

## Phase 1: Strip down to codegen core

- [x] Read ARCHITECTURE.rst and understand the codebase
- [ ] Delete fluent-specific source modules: `compiler.py`, `bundle.py`, `builtins.py`, `escapers.py`, `errors.py`, `resource.py`, `runtime.py`, `source.py`, `types.py`
- [ ] Remove fluent-specific dependencies from `codegen.py` (FtlSource) and `utils.py`
- [ ] Delete fluent-specific tests: `tests/format/`, `test_bundle.py`, `test_compiler.py`, `test_types.py`, `test_utils.py`, `tests/utils.py`
- [ ] Delete `compat.py` (use stdlib directly for Python 3.12+)
- [ ] Delete docs folder (all fluent-specific)
- [ ] Delete `ARCHITECTURE.rst`, `CHANGELOG.rst`, `CONTRIBUTING.rst`, `RELEASE.rst`, `MANIFEST.in`, `release.sh`, `tools/`
- [ ] Rename package: `fluent_compiler` → `fluent_codegen`
- [ ] Update `pyproject.toml`: name, description, dependencies, classifiers, Python 3.12+
- [ ] Rewrite `README.rst` → `README.md`
- [ ] Update `__init__.py`
- [ ] Ensure tests pass with pytest
- [ ] Git commit

## Phase 2: Cleanups

- [ ] Rewrite `test_codegen.py` to use pytest style instead of unittest
- [ ] Remove `as_multiple_module_ast` from `Module`
- [ ] Remove `Decimal` support from `Number`
- [ ] Remove any other fluent-specific remnants in codegen
- [ ] Ensure 100% test coverage (add `pytest-cov`)
- [ ] Git commit

## Phase 3: Modern tooling

- [ ] Configure ruff for Python 3.12+ (update existing config)
- [ ] Run pyupgrade via ruff UP rules for Python 3.12+
- [ ] Run ruff fix & format
- [ ] Git commit
