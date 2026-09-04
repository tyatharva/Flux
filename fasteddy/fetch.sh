#!/usr/bin/env bash
# Fetch NCAR FastEddy at the pinned release and apply the project's patch series.
#
#   fasteddy/fetch.sh                  -> ./FastEddy-model-5.0.1/ (the path every build script expects)
#   FE_DIR=/elsewhere fasteddy/fetch.sh
#   FE_FETCH=tarball fasteddy/fetch.sh -> use the GitHub tag tarball instead of git
#
# If the destination already exists it is VERIFIED against fasteddy/MANIFEST.sha256 and
# left alone, so a checkout that has been edited by hand is refused rather than built.
# The manifest covers every source file of the patched tree (build products excluded), so
# passing it means the tree is the one every published result was produced from.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="${FE_DIR:-$ROOT/FastEddy-model-5.0.1}"
# shellcheck source=UPSTREAM
. "$HERE/UPSTREAM"

verify() {  # $1 = tree
  ( cd "$1" && sha256sum -c --quiet "$HERE/MANIFEST.sha256" ) \
    || { echo "FATAL: $1 does not match fasteddy/MANIFEST.sha256 (edited, or a different release)." >&2; return 1; }
  echo "  verified: $(wc -l < "$HERE/MANIFEST.sha256") files match fasteddy/MANIFEST.sha256"
}

series="$(cat "$HERE"/patches/*.patch | sha256sum | cut -c1-64)"
[ "$series" = "$SERIES_SHA256" ] \
  || { echo "FATAL: patch series sha256 $series != UPSTREAM's $SERIES_SHA256; update UPSTREAM." >&2; exit 1; }

if [ -e "$DEST" ]; then
  echo "== $DEST exists; verifying"
  verify "$DEST"
  exit 0
fi

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
echo "== fetching FastEddy $UPSTREAM_TAG ($UPSTREAM_SHA)"
if [ "${FE_FETCH:-git}" = git ] && command -v git >/dev/null 2>&1; then
  git -c advice.detachedHead=false clone -q --branch "$UPSTREAM_TAG" --depth 1 "$UPSTREAM_URL" "$tmp/src"
  got="$(git -C "$tmp/src" rev-parse HEAD)"
  [ "$got" = "$UPSTREAM_SHA" ] || { echo "FATAL: tag $UPSTREAM_TAG is at $got, pinned $UPSTREAM_SHA" >&2; exit 1; }
  rm -rf "$tmp/src/.git"
else
  ( cd "$tmp" && curl -fsSL "$TARBALL_URL" | tar -xz ) || wget -qO- "$TARBALL_URL" | tar -xz -C "$tmp"
  mv "$tmp"/FastEddy-model-* "$tmp/src"
fi

echo "== applying $(ls "$HERE"/patches/*.patch | wc -l) patches"
for p in "$HERE"/patches/*.patch; do
  patch -d "$tmp/src" -p1 -s --no-backup-if-mismatch < "$p" \
    || { echo "FATAL: $(basename "$p") did not apply" >&2; exit 1; }
  echo "  $(basename "$p")"
done
verify "$tmp/src"
mkdir -p "$(dirname "$DEST")"; mv "$tmp/src" "$DEST"
echo "== $DEST ready: FastEddy $UPSTREAM_TAG + $(ls "$HERE"/patches | wc -l) patches (series $series)"
