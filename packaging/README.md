# Packaging

`packaging/package.sh` → `dist/nz-dz-helper-<version>.tar.gz` (+ `.sha256`). Python, so no compiled binary: the launcher (`nz-dz-helper`) builds a virtualenv in the data directory on first run. OpenRC service and systemd user unit included. Only git-tracked files are packed, so `.env`, databases, cookies, caches and logs cannot end up in it. No ebuild (by choice).

Install from the tarball: `./install.sh` (under `/usr/local`, `PREFIX=$HOME/.local` works without root), `DESTDIR=… ./install.sh` to stage, `./install.sh uninstall` to remove
what it installed (it records a manifest). Existing files in `/etc` are never overwritten; the new copy is written as `<name>.new`.
Checked: install/uninstall in a DESTDIR; the launcher stops with a clear message when `.env` is missing. The virtualenv step and the OpenRC script were not run.
