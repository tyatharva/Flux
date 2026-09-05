#!/usr/bin/env bash
# Fetch the large assets that are not in git from the Hugging Face dataset repository, verify
# them against assets/SHA256SUMS, and put each at the path the code expects.
#
#   bin/fetch_assets.sh corpus              corpus/corpus_cone.h5, corpus/corpus_raw.h5      (76 MB)
#   bin/fetch_assets.sh pairs               corpus/pairs_npz/ (the 1366 source records)       (55 MB)
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
    pairs)       echo '^corpus/pairs_npz\.tar$' ;;
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
  if command -v huggingface-cli >/dev/null 2>&1; then
    tmp="$(mktemp -d)"
    huggingface-cli download --repo-type dataset "$REPO" "$hf" --local-dir "$tmp" >/dev/null
    mv "$tmp/$hf" "$local"; rm -rf "$tmp"
  else
    curl -fL --retry 3 -o "$local" "$BASE/$hf"
  fi
  echo "$sha  $local" | sha256sum -c --quiet - || { echo "FATAL: checksum mismatch on $local; removed" >&2; rm -f "$local"; exit 1; }
  echo "  verified $local"
}

n=0
for g in "$@"; do
  rx="$(want "$g")"
  while read -r sha hf local; do
    [ -n "$sha" ] && [ "${sha:0:1}" != "#" ] || continue
    echo "$hf" | grep -Eq "$rx" || continue
    fetch_one "$hf" "$local" "$sha"; n=$((n+1))
  done < "$SUMS"
  if [ "$g" = pairs ] || [ "$g" = all ]; then
    # the 1366 source records travel as one tar; unpack beside the .h5 files
    if [ ! -d corpus/pairs_npz ]; then
      curl -fL --retry 3 -o corpus/pairs_npz.tar "$BASE/corpus/pairs_npz.tar" && tar -xf corpus/pairs_npz.tar -C corpus && rm corpus/pairs_npz.tar
      echo "  unpacked corpus/pairs_npz/ ($(ls corpus/pairs_npz | wc -l) files)"
    fi
  fi
done
echo "done: $n file(s) checked against assets/SHA256SUMS"
