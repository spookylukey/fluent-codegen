======================
Releasing new versions
======================

To do a release::

  uv version <newversion>

Update ``docs/changelog.rst``.

Update ``docs/conf.py`` with new version number.

Add changes to git and commit with::

  git commit -m "Version bump $(uv version)"

Then do ``./release.sh``
