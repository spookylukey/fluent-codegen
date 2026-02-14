# fluent-codegen Migration Plan

Migrate from `fluent-compiler` (Fluent localization compiler) to `fluent-codegen` (general-purpose Python code generation library), keeping only the codegen module and its dependencies.

## Phase 1: Strip down to codegen core

- [x] Read ARCHITECTURE.rst and understand the codebase
- [x] Delete fluent-specific source modules: `compiler.py`, `bundle.py`, `builtins.py`, `escapers.py`, `errors.py`, `resource.py`, `runtime.py`, `source.py`, `types.py`
- [x] Remove fluent-specific dependencies from `codegen.py` (FtlSource) and `utils.py`
- [x] Delete fluent-specific tests: `tests/format/`, `test_bundle.py`, `test_compiler.py`, `test_types.py`, `test_utils.py`, `tests/utils.py`
- [x] Delete `compat.py` (use stdlib directly for Python 3.12+)
- [x] Delete docs folder (all fluent-specific)
- [x] Delete `ARCHITECTURE.rst`, `CHANGELOG.rst`, `CONTRIBUTING.rst`, `RELEASE.rst`, `MANIFEST.in`, `release.sh`, `tools/`
- [x] Rename package: `fluent-compiler` → `fluent-codegen`
- [x] Update `pyproject.toml`: name, description, dependencies, classifiers, Python 3.12+
- [x] Rewrite `README.rst` → `README.md`
- [x] Update `__init__.py`
- [x] Ensure tests pass with pytest
- [x] Git commit

## Phase 2: Cleanups

- [x] Rewrite `test_codegen.py` to use pytest style instead of unittest
- [x] Remove `as_multiple_module_ast` from `Module`
- [x] Remove `Decimal` support from `Number`
- [x] Remove any other fluent-specific remnants in codegen (FtlSource)
- [x] Ensure 99% test coverage (remaining 1% = abstract method bodies + platform branch)
- [x] Git commit

## Phase 3: Modern tooling

- [x] Configure ruff for Python 3.12+ (update existing config)
- [x] Run pyupgrade via ruff UP rules for Python 3.12+
- [x] Run ruff fix & format
- [x] Git commit
