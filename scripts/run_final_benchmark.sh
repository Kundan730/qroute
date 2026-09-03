#!/usr/bin/env bash
#
# Definitive benchmark run for the SIH 26137 submission.
#
# Run this on an otherwise idle machine: the protocol pins every solver to one
# thread and compares by wall clock, so background load does not slow the sweep
# down evenly, it slows down whichever solver happened to be scheduled against
# it. A sweep taken on a busy laptop is not a worse measurement, it is a
# different one, and the published tables would no longer describe it.
#
# Expect roughly an hour on ten cores. configs/main.yaml is 17 instances x 9
# solvers x 10 seeds at a 20 second budget each, and every run gets its full
# budget. `workers` is deliberately unset in that config so the runner sizes
# itself to this host; see the comment there.
#
# Usage:
#   scripts/run_final_benchmark.sh                    # configs/main.yaml
#   scripts/run_final_benchmark.sh configs/quick.yaml # a two-minute smoke sweep
#
# WHY THIS SCRIPT DRIVES `qroute bench` AND NOT THE RUNNER DIRECTLY
# ----------------------------------------------------------------
# It used to embed a Python program in a heredoc that constructed
# BenchmarkRunner itself. That form cannot work, and fails in a way that looks
# like success. BenchmarkRunner uses a spawn-based process pool, and a spawned
# child re-executes the parent's __main__ before it runs anything of ours (see
# the module docstring of qroute/api/runs.py). Under `python - <<'PY'` the
# parent's __main__ is "<stdin>", so every child dies with
#
#   FileNotFoundError: [Errno 2] No such file or directory: '.../<stdin>'
#
# and the sweep records `status=worker_died` for every single run. Measured on
# configs/quick.yaml scaled down to eight runs: ok=0 failed=8, and the script
# still exited 0 and wrote a report full of empty tables.
#
# `qroute bench` is the same runner reached through the installed console
# script, whose __main__ is a real file behind an `if __name__ == "__main__"`
# guard, so the children start correctly. The same eight runs through it:
# ok=8 failed=0. Using it also means this script and the README document one
# command rather than two ways of saying the same thing.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/main.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "error: no such config: $CONFIG" >&2
    echo "  available: $(ls configs/*.yaml | tr '\n' ' ')" >&2
    exit 2
fi

# Activate the checkout's virtualenv only when the caller has not already put
# themselves in one. Someone running this inside conda, a container, or a venv
# of their own has made a choice, and silently overriding it is how a benchmark
# ends up reporting numbers from an environment nobody intended.
if [[ -z "${VIRTUAL_ENV:-}" && -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if ! command -v qroute > /dev/null 2>&1; then
    echo "error: the 'qroute' command is not on PATH." >&2
    echo "  Install the package first:  pip install -e '.[dev,baselines]'" >&2
    exit 1
fi

# Read the configuration once, through the same object the runner will use, and
# answer everything this script needs to know about it. Nothing is guessed from
# the file's text: the output directory is not assumed to be results/runs/main
# (a config with a different name or output_dir would otherwise have its report
# generated from whatever happened to be in the hard-coded directory), and "does
# it use pyvrp" comes from the parsed algorithm list rather than from grepping
# for the word, which would also fire on a comment that merely mentions it.
#
# A failure here — unreadable YAML, a field the config class rejects — aborts
# the script through `set -e` before anything has been written, which is the
# right moment for it to fail.
#
# This is a plain `python -c`, not a heredoc: it starts no subprocesses, so the
# __main__ hazard described at the top does not apply to it.
CONFIG_FACTS="$(python -c "
import sys
from pathlib import Path
from qroute.benchmark import BenchmarkConfig
cfg = BenchmarkConfig.from_yaml(sys.argv[1])
print(Path(cfg.output_dir) / cfg.name)
print('yes' if 'pyvrp' in cfg.algorithms else 'no')
print(len(cfg.instances))
print(len(cfg.algorithms))
" "$CONFIG")"
OUTPUT_DIR="$(printf '%s\n' "$CONFIG_FACTS" | sed -n 1p)"
WANTS_PYVRP="$(printf '%s\n' "$CONFIG_FACTS" | sed -n 2p)"
N_INSTANCES="$(printf '%s\n' "$CONFIG_FACTS" | sed -n 3p)"
N_ALGORITHMS="$(printf '%s\n' "$CONFIG_FACTS" | sed -n 4p)"

# A config that names no instances or no algorithms parses happily and produces
# a sweep of nothing. Left to run, it creates $OUTPUT_DIR, writes an empty
# rows.jsonl and a summary.json beside it, and only then fails in the reporting
# step — so a typo in a config file leaves a directory of empty results sitting
# in results/runs/ looking like a real sweep that went wrong. Refuse first, and
# write nothing.
if (( N_INSTANCES == 0 || N_ALGORITHMS == 0 )); then
    echo "error: $CONFIG defines $N_INSTANCES instance(s) and $N_ALGORITHMS algorithm(s);" >&2
    echo "  a sweep needs at least one of each. Nothing was written." >&2
    exit 2
fi

# PyVRP is an optional extra, so warn rather than refuse: a sweep without it is
# still a valid sweep, it simply has one fewer column than the published one and
# should not be compared against it row for row.
if [[ "$WANTS_PYVRP" == "yes" ]] && ! python -c "import pyvrp" > /dev/null 2>&1; then
    echo "warning: $CONFIG asks for pyvrp and it is not installed." >&2
    echo "         Those runs will be recorded as failures, and the sweep" >&2
    echo "         will not be comparable with the published tables." >&2
    echo "         Install it with:  pip install -e '.[baselines]'" >&2
fi

echo "config:      $CONFIG"
echo "environment: $(python -c 'import sys; print(sys.executable)')"
echo "output:      $OUTPUT_DIR"
echo "load average before start:"
uptime
echo

# `qroute bench` refuses to write over an existing result set unless --force is
# given; that refusal is deliberate and is not suppressed here. If it stops the
# sweep, either pass a different --name or move the previous results aside on
# purpose.
qroute bench --config "$CONFIG"

if [[ ! -f "$OUTPUT_DIR/rows.jsonl" ]]; then
    echo "error: expected $OUTPUT_DIR/rows.jsonl after the sweep; it is not there." >&2
    echo "  The sweep's own output above says where it actually wrote." >&2
    exit 1
fi

# `qroute report` is the only reporting path. It is the same code that produced
# the tables in the submission, so a number printed in this terminal is the
# number in the document. It is not wrapped in a fallback: if it fails, the
# sweep's results are on disk and intact, and the right response is to read the
# error rather than to silently produce a report by some other route. (The
# fallback this script used to carry called qroute.benchmark.report.write_all
# and qroute.benchmark.plots.write_all, neither of which exists, so it could
# only ever have printed "reporting layer unavailable" and exited 0.)
#
# --out is deliberately not passed: `--format markdown` without it hands the
# work to qroute.benchmark.report.build_report, which writes report.md and the
# full set of CSVs into the result directory. Passing --out switches to the
# CLI's smaller built-in renderer and writes a single file instead.
echo
echo "==> generating tables and figures in $OUTPUT_DIR"
qroute report "$OUTPUT_DIR" --format markdown --plots > /dev/null

echo
echo "==> done. Results in $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
