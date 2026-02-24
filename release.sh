#!/bin/sh
set -e

die()  { echo "ERROR: $*" >&2; exit 1; }

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree is dirty — commit or stash changes first"
fi

# Pre-flight checks
uv sync
pre-commit run --all --all-files || exit 1
pytest || exit 1
pyright || exit 1

# Build
umask 000
rm -rf build dist
git ls-tree --full-tree --name-only -r HEAD | xargs chmod ugo+r

uv build --sdist --wheel || exit 1
uv publish || exit 1

# Tag and push
VERSION=$(uv version --short | tr -d '\n') || exit 1

git tag "$VERSION" || exit 1
git push || exit 1
git push --tags || exit 1
