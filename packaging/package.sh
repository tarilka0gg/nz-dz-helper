#!/bin/bash
# package.sh - pack nz-dz-helper: dist/nz-dz-helper-<version>.tar.gz (+ .sha256).
# Python app, so there is no compiled binary: the tarball has the code, a launcher that builds its virtualenv
# on first run, OpenRC and systemd service files, and install.sh (install / uninstall, PREFIX, DESTDIR).
# Only files tracked by git are packed, so .env, nz_solver.db, cookies, caches and logs can never end up in it.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=nz-dz-helper
VERSION=${VERSION:-0.1.0+git$(git rev-parse --short HEAD)}
D=dist/$NAME-$VERSION
rm -rf "${D:?}"

APPDIR=$D/prefix/share/$NAME
mkdir -p "$APPDIR"
for f in $(git ls-files src config.yaml requirements.txt .env.example README.md nz-api-notes.md); do
    install -Dm644 "$f" "$APPDIR/$f"
done
install -Dm755 packaging/nz-dz-helper "$D/prefix/bin/nz-dz-helper"
install -Dm644 packaging/nz-dz-helper.service "$APPDIR/nz-dz-helper.service"
install -Dm755 packaging/etc/init.d/nz-dz-helper "$D/etc/init.d/nz-dz-helper"
install -Dm644 packaging/etc/conf.d/nz-dz-helper "$D/etc/conf.d/nz-dz-helper"
install -m755 packaging/install.sh "$D/install.sh"
cat > "$D/POST-INSTALL.txt" <<'TXT'
nz-dz-helper keeps its .env, config.yaml, caches and virtualenv in a data directory, not in the install:
  per user:   mkdir -p ~/.local/share/nz-dz-helper; cp /usr/local/share/nz-dz-helper/.env.example ~/.local/share/nz-dz-helper/.env
              (fill it in), then run `nz-dz-helper`; a systemd user unit is in /usr/local/share/nz-dz-helper/.
  OpenRC:     useradd -r -m -d /var/lib/nz-dz-helper nzdz; put .env in /var/lib/nz-dz-helper (owner nzdz, mode 600);
              rc-update add nz-dz-helper default; rc-service nz-dz-helper start
The first run needs Python 3 with venv and network access to install the requirements.
TXT

OUT=dist/$NAME-$VERSION.tar.gz
tar -C dist --owner=0 --group=0 -czf "$OUT" "$NAME-$VERSION"
(cd dist && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo "$OUT"
