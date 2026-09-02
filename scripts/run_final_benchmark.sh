#!/usr/bin/env bash
# Definitive benchmark run for the SIH 26137 submission.
#
# Run this on an otherwise idle machine: the protocol pins every solver to one
# thread and compares by wall clock, so background load invalidates the results.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Load average before start:"; uptime

python -u - <<'PY'
import warnings, time
warnings.filterwarnings("ignore")
from qroute.benchmark import BenchmarkConfig, BenchmarkRunner

cfg = BenchmarkConfig.from_yaml("configs/main.yaml")
started = time.time()

def progress(p):
    row = p["row"]
    if p["done"] % 40 == 0 or p["done"] == p["total"] or row.get("status") != "ok":
        print(f"{p['done']:5d}/{p['total']} {p['elapsed']:7.0f}s "
              f"{row['algorithm']}@{row['instance']} status={row.get('status')} "
              f"gap={row.get('gap')}", flush=True)

result = BenchmarkRunner(cfg, progress=progress).run()
print(f"DONE in {time.time()-started:.0f}s -> {result['output_dir']}", flush=True)
print(f"ok={result['summary']['n_ok']} failed={result['summary']['n_failed']}", flush=True)
for f in result["summary"]["failures"][:10]:
    print("  FAILURE:", f, flush=True)
PY

echo "Generating tables and figures"
python -u -m qroute.cli.main report results/runs/main --format markdown --plots || \
  python -u - <<'PY'
from qroute.benchmark.runner import load_results, BenchmarkRunner
rows = load_results("results/runs/main/rows.jsonl")
summary = BenchmarkRunner.summarise(rows)
try:
    from qroute.benchmark import report, plots
    report.write_all(rows, summary, "results/runs/main")
    plots.write_all(rows, summary, "results/runs/main/figures")
except Exception as exc:
    print("reporting layer unavailable:", exc)
PY
echo "Done. Results in results/runs/main"
