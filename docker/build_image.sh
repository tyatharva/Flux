#!/usr/bin/env bash
# Build the portable seed image. THE TAG IS THE COMMIT.
#
#   docker/build_image.sh                    -> flux-seeds:<flux-sha>-fe<fasteddy-sha>
#   docker/build_image.sh --push ghcr.io/me  -> also tags and pushes there
#
# WHY THE TAG IS THE COMMIT AND NOT `latest`. The whole argument for baking the code in is
# that a rented machine cannot pull the wrong commit -- there is nothing on it to pull.
# That only holds if the tag names the commit; `latest` reintroduces exactly the ambiguity
# the design removes, one layer up. `latest` IS also applied, for convenience, and the
# banner every run prints comes from IMAGE_PROVENANCE.txt inside the image rather than
# from the tag, so a mislabelled pull is still self-identifying.
#
# BOTH INPUTS ARE PINNED. This repository by its commit, and FastEddy by fasteddy/UPSTREAM
# (NCAR v5.0.1 + the patch series, which the image fetches and verifies itself). A dirty
# tree is refused unless FLUX_ALLOW_DIRTY=1, because an image tagged with a commit whose
# tree it does not contain is worse than an untagged one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
NAME="${IMAGE_NAME:-flux-seeds}"
PUSH_TO=""
if [ "${1:-}" = "--push" ]; then PUSH_TO="${2:?--push needs a registry prefix, e.g. ghcr.io/tyatharva}"; shift 2; fi

dirty(){ [ -n "$(git -C "$1" status --porcelain 2>/dev/null)" ] && echo "-dirty" || echo ""; }
SHA="$(git -C "$ROOT" rev-parse --short=12 HEAD)$(dirty "$ROOT")"
# FastEddy's identity is the pinned release plus the patch series, both recorded in
# fasteddy/UPSTREAM; the image fetches and verifies that tree itself (fasteddy/fetch.sh).
. "$ROOT/fasteddy/UPSTREAM"
FESHA="${UPSTREAM_TAG}-p$(cat "$ROOT"/fasteddy/patches/*.patch | sha256sum | cut -c1-12)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

case "$SHA" in
  *dirty*)
    if [ "${FLUX_ALLOW_DIRTY:-0}" != "1" ]; then
      echo "REFUSED: uncommitted changes." >&2
      git -C "$ROOT" status --short | sed 's/^/  flux     /' >&2
      echo "  The tag would name a commit whose tree the image does not contain." >&2
      echo "  Commit, or set FLUX_ALLOW_DIRTY=1 to build a '-dirty' tag anyway." >&2
      exit 1
    fi
    echo "  FLUX_ALLOW_DIRTY=1: building a -dirty tag" >&2 ;;
esac

TAG="${NAME}:${SHA}-fe${FESHA}"
echo "=== building ${TAG} ==="
echo "  flux     ${SHA}"
echo "  fasteddy ${FESHA}  (NCAR ${UPSTREAM_TAG} ${UPSTREAM_SHA:0:12} + $(ls "$ROOT"/fasteddy/patches | wc -l) patches)"
echo "  date     ${BUILD_DATE}"
echo "  context  $(du -sh --exclude=.git --exclude=runs --exclude=data/hrrr --exclude=data/raw . 2>/dev/null | cut -f1) (before .dockerignore)"

# BuildKit is enabled on this machine but the buildx component is absent, so the classic
# builder is selected explicitly rather than left to fail with "buildx is missing or
# broken". The classic builder handles multi-stage and --target fine; what it does NOT
# support is per-Dockerfile ignore files, which is why there is one .dockerignore.
DOCKER_BUILDKIT=0 docker build \
  -f Dockerfile.blackwell \
  --build-arg GIT_COMMIT="${SHA}" \
  --build-arg GIT_COMMIT_FE="${FESHA}" \
  --build-arg BUILD_DATE="${BUILD_DATE}" \
  ${FE_GENCODE:+--build-arg FE_GENCODE="${FE_GENCODE}"} \
  -t "${TAG}" -t "${NAME}:latest" \
  "$@" .

echo
echo "=== built ${TAG} ==="
docker run --rm --entrypoint cat "${TAG}" /flux/IMAGE_PROVENANCE.txt | sed 's/^/  /'
echo
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | grep "^  ${NAME}:" || true
if [ -n "$PUSH_TO" ]; then
  echo; echo "=== pushing to ${PUSH_TO} ==="
  for t in "${SHA}-fe${FESHA}" latest; do
    docker tag "${NAME}:${t}" "${PUSH_TO}/${NAME}:${t}"
    docker push "${PUSH_TO}/${NAME}:${t}"
  done
  docker inspect --format='  digest {{index .RepoDigests 0}}' "${PUSH_TO}/${NAME}:latest" || true
fi
cat <<EOF

RUN IT
  docker run --gpus all -v /out:/out ${TAG} run_seeds --gpu-count 16

CHECK IT FIRST (seconds, no seeds)
  docker run --gpus all -v /out:/out ${TAG} verify
EOF
