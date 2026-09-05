#!/usr/bin/env bash
# Fetch the large assets that are not in git from the Hugging Face dataset repository, verify
# them against assets/SHA256SUMS, and put each at the path the code expects.
#
#   bin/fetch_assets.sh corpus              corpus/corpus_cone.h5, corpus/corpus_raw.h5      (76 MB)
#   bin/fetch_assets.sh pairs               corpus/pairs_npz/ and corpus/logs/, unpacked from two tars (55 MB)
#   bin/fetch_assets.sh seeds               seeds/*/return/seed_restart.nc, 30 files           (2.1 GB)
#   bin/fetch_assets.sh weights             the FNO and CFM checkpoints                        (610 MB)
#   bin/fetch_assets.sh predictions         the audited test-split outputs + FNO val preds     (200 MB)
#   bin/fetch_assets.sh all
#
# Uses `huggingface-cli` if present (pip install huggingface_hub), otherwise curl. Every file
# is checked against assets/SHA256SUMS after download; a mismatch is fatal and the file is
# removed. Re-running skips files that are already present and correct.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
REPO="${FLUX_HF_REPO:-tyatharva/flux-kegonsa}"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"
SUMS="$ROOT/assets/SHA256SUMS"
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 64; }

want() {  # group -> regex on the HF path
  case "$1" in
    corpus)      echo '^corpus/corpus_(cone|raw)\.h5$' ;;
    pairs)       echo '^corpus/(pairs_npz|logs)\.tar$' ;;
    seeds)       echo '^seeds/' ;;
    weights)     echo '^weights/' ;;
    predictions) echo '^predictions/' ;;
    all)         echo '.' ;;
    *) echo "unknown group '$1'" >&2; exit 64 ;;
  esac
}

fetch_one() {  # hf_path local_path sha
  local hf="$1" local="$2" sha="$3"
  if [ -f "$local" ] && echo "$sha  $local" | sha256sum -c --quiet - 2>/dev/null; then
    echo "  ok      $local"; return 0
  fi
  mkdir -p "$(dirname "$local")"
  echo "  fetch   $hf -> $local"
  if command -v hf >/dev/null 2>&1; then
    tmp="$(mktemp -d)"
    hf download --repo-type dataset "$REPO" "$hf" --local-dir "$tmp" >/dev/null
    mv "$tmp/$hf" "$local"; rm -rf "$tmp"
  else
    curl -fL --retry 3 -o "$local" "$BASE/$hf"
  fi
  echo "$sha  $local" | sha256sum -c --quiet - || { echo "FATAL: checksum mismatch on $local; removed" >&2; rm -f "$local"; exit 1; }
  echo "  verified $local"
}

fetch_tar() {  # hf_path local_dir sha : a directory that travels as one tar
  local hf="$1" dir="${2%/}" sha="$3" tar="${2%/}.tar"
  if [ -d "$dir" ]; then echo "  ok      $dir/ ($(ls "$dir" | wc -l) files, already unpacked)"; return 0; fi
  mkdir -p "$(dirname "$dir")"
  echo "  fetch   $hf -> $tar"
  curl -fL --retry 3 -o "$tar" "$BASE/$hf"
  echo "$sha  $tar" | sha256sum -c --quiet - || { echo "FATAL: checksum mismatch on $tar; removed" >&2; rm -f "$tar"; exit 1; }
  tar -xf "$tar" -C "$(dirname "$dir")" && rm -f "$tar"
  echo "  unpacked $dir/ ($(ls "$dir" | wc -l) files)"
}

n=0
for g in "$@"; do
  rx="$(want "$g")"
  while read -r sha hf local; do
    [ -n "$sha" ] && [ "${sha:0:1}" != "#" ] || continue
    echo "$hf" | grep -Eq "$rx" || continue
    case "$local" in
      */) fetch_tar "$hf" "$local" "$sha" ;;
      *)  fetch_one "$hf" "$local" "$sha" ;;
    esac
    n=$((n+1))
  done < "$SUMS"
done
echo "done: $n file(s) checked against assets/SHA256SUMS"
