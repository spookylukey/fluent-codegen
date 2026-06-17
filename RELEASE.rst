======================
Releasing new versions
======================

To do a release::

  uv version <newversion>

Update ``docs/changelog.rst``.

Add changes to git and commit with::

  git commit -m "Version bump $(uv version)"

Then do ``./release.sh``
