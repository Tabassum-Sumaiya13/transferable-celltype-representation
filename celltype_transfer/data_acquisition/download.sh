#!/usr/bin/env bash
# Stage 0.2 / 0.3 - fetch the four new cohorts.
# Resumable (curl -C -), retrying, and safe to re-run: an already-complete file is skipped.
set -u
D="$(cd "$(dirname "$0")/../.." && pwd)/Datasets"

get () {  # get <dest-relative-path> <url> <expected-bytes>
  local out="$D/$1" url="$2" want="$3"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    local have; have=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then echo "OK    $1 (already complete)"; return 0; fi
    echo "RESUME $1 ($have / $want)"
  else
    echo "START $1 ($want bytes)"
  fi
  curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$out" "$url" || {
    echo "FAIL  $1"; return 1; }
  local got; got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then echo "DONE  $1"; else echo "SIZE MISMATCH $1: $got != $want"; return 1; fi
}

MEND="https://data.mendeley.com/public-files/datasets/3gmvy3bcmk/files"

case "${1:-all}" in
  small)
    get "Phillips/Raw_df_CODEX.csv" "$MEND/3a4d5d06-1042-47a2-bbb4-86195bbac0cf/file_downloaded" 99449956
    get "Phillips/TMA_key.xlsx"     "$MEND/fe33deb5-9b4f-400f-a2e0-7b5c5b88d06b/file_downloaded" 10633
    get "Risom/DataTables.zip"      "https://zenodo.org/records/5945388/files/DataTables.zip?download=1" 65495275
    ;;
  big)
    get "Sorin/LungData.zip"        "https://zenodo.org/records/7760826/files/LungData.zip?download=1" 2088454891
    get "Danenberg/MBTMEStrIMCPublic.zip" "https://zenodo.org/records/6036188/files/MBTMEStrIMCPublic.zip?download=1" 6650249371
    ;;
  *)
    "$0" small && "$0" big
    ;;
esac
echo "=== ${1:-all} finished ==="
