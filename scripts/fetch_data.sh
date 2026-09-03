#!/usr/bin/env bash
#
# Prepare the two pieces of data that a fresh clone does not have.
#
# Both are deliberately absent from version control and both are needed before
# the platform can show everything it does:
#
#   1. The three road graphs (~40 MB of GraphML). data/osm/networks.json records
#      the exact OSMnx recipe; `qroute osm fetch` rebuilds them from it. Without
#      them the service still runs — every benchmark instance, every one of the
#      eleven solvers /api/algorithms lists, and the whole /api surface are
#      unaffected — but the map is empty and /api/health reports "networks": 0.
#
#   2. The definitive benchmark's per-run log. It is 43 MB uncompressed, so it
#      is committed as results/runs/main/rows.jsonl.gz. /api/benchmarks looks for
#      the uncompressed rows.jsonl and lists nothing without it, so the results
#      view in the UI is blank on a clean checkout until this has been run.
#
# The script is idempotent: a graph that is already on disk is not re-downloaded
# (pass --force to override), and a log that is already expanded is left alone.
# Run it as often as you like.
#
# Usage:
#   scripts/fetch_data.sh              # both steps
#   scripts/fetch_data.sh --force      # re-download the road graphs as well
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORCE=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE="--force" ;;
        # The help text is this file's own header block, printed with the
        # comment markers stripped. Extracted by "every comment line after the
        # shebang, up to the first line that is not one" rather than by a line
        # range: a hard-coded range silently truncates the help the first time
        # somebody adds a paragraph above it.
        -h|--help)
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# Activate the checkout's virtualenv only when the caller has not already put
# themselves in one. Someone running this inside conda, a container, or a venv
# of their own has made a choice, and silently overriding it is how a script
# ends up producing results from an environment nobody intended.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if ! command -v qroute > /dev/null 2>&1; then
    echo "error: the 'qroute' command is not on PATH." >&2
    echo "  Install the package first:  pip install -e '.[dev]'" >&2
    exit 1
fi

# Anchor the data root explicitly. `qroute osm fetch` resolves its default
# output directory relative to the working directory, and this script has
# already cd'd to the repository root, but saying so makes the behaviour
# independent of that and lets an operator point the whole thing elsewhere.
export QROUTE_DATA="${QROUTE_DATA:-$ROOT/data}"

echo "==> data root: $QROUTE_DATA"

# ---------------------------------------------------------------------------
# 1. Expand the committed benchmark log
# ---------------------------------------------------------------------------
# `python -m gzip` rather than the gzip binary: python is guaranteed to be here
# (the check above passed) and gzip is not, on every platform this may run on.
# It reads standard input and writes standard output when given no file
# argument, and it never deletes its input, so the committed .gz survives.
echo "==> expanding committed benchmark logs"
expanded=0
skipped=0
shopt -s nullglob
for gz in results/runs/*/rows.jsonl.gz; do
    plain="${gz%.gz}"
    if [[ -f "$plain" ]]; then
        echo "    $plain already present"
        skipped=$((skipped + 1))
        continue
    fi
    # Decompress to a temporary file beside the target and rename only once it
    # is complete. Writing straight to $plain looks simpler and is a trap: a
    # truncated or corrupt archive leaves a half-written rows.jsonl on disk,
    # every later run of this script reports it as "already present" and skips
    # it, and nothing downstream complains — the report reader and
    # /api/benchmarks both tolerate a short final line by design, so the
    # platform would go on serving a fraction of the sweep as though it were
    # all of it. A benchmark that is quietly wrong is worse than one that is
    # obviously missing, so this fails loudly and leaves no partial file.
    tmp="$plain.partial.$$"
    err="$tmp.err"
    if ! python -m gzip -d < "$gz" > "$tmp" 2> "$err"; then
        echo "error: $gz is corrupt or truncated; nothing was written." >&2
        # The last line of the traceback is the reason in one sentence; the
        # rest of it is interpreter internals that help nobody here.
        tail -n 1 "$err" | sed 's/^/  /' >&2
        echo "  Restore the file and try again:  git checkout -- $gz" >&2
        rm -f "$tmp" "$err"
        exit 1
    fi
    rm -f "$err"
    mv "$tmp" "$plain"
    echo "    wrote $plain ($(wc -l < "$plain" | tr -d ' ') runs)"
    expanded=$((expanded + 1))
done
shopt -u nullglob
if (( expanded == 0 && skipped == 0 )); then
    echo "    no results/runs/*/rows.jsonl.gz found; nothing to expand"
fi

# ---------------------------------------------------------------------------
# 2. Rebuild the road graphs
# ---------------------------------------------------------------------------
# This needs outbound internet access to the Overpass API and takes roughly a
# minute per network. It is the slow step, so it goes last: if it fails, step 1
# has already succeeded and re-running the script will not redo it.
echo "==> rebuilding road graphs from OpenStreetMap (about a minute each)"
qroute osm fetch --out-dir "$QROUTE_DATA/osm" ${FORCE:+$FORCE}

echo "==> done"
echo "    Start the platform with:  qroute serve"
echo "    Then check:               curl -fsS http://127.0.0.1:8000/api/health"
