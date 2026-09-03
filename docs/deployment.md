# Deployment

How to install, run, operate and debug qroute, written for someone who has not
seen the project before.

Every command below was run on the development machine (macOS 26, Apple
silicon, Python 3.13.7, Node 22.16.0) unless it is marked otherwise. Where a
command could not be executed here, it says so rather than implying it was
tested.

**Docker is not installed on the machine this documentation was written on, so
the image has never been built. The Dockerfile is written to be read and
checked, not to be taken on trust.** Its base-image and wheel-availability
claims were verified against PyPI directly (the exact commands are in the
Dockerfile's own comments and are reproduced under
[§13](#13-verifying-the-image-without-building-it)),
but nothing here should be read as "this image builds and runs". The first
person with a Docker daemon should build it and correct this paragraph.

---

## 1. What is being deployed

One process. `qroute serve` starts a FastAPI application under uvicorn that
serves, on a single port:

* `/api/...` — the HTTP interface: instances, algorithms, runs (with a
  Server-Sent Events stream per run), road networks, traffic and benchmark
  results.
* `/docs` — the generated OpenAPI documentation.
* `/` — the built React application, when `frontend/dist` exists. When it does
  not, `/` returns a short JSON document saying how to build it, and everything
  under `/api` still works.

Solver runs do **not** execute in the web process. Each one gets its own
operating-system process, forked from a fork server that is started during
application startup. That is why the server stays responsive during a
twenty-second solve, why a run can be cancelled, and why a segfault in a
compiled kernel takes down only that run. It also means the process count you
will see is `1 + 1 + n_active_runs` (server, fork server, workers), which is
normal and not a leak.

There is no database, no message queue and no external service. State is one
process's memory plus files on disk.

---

## 2. Prerequisites

| | Version used here | Notes |
| --- | --- | --- |
| Python | 3.13.7 | `requires-python = ">=3.11"`. 3.12 and 3.13 both have complete wheel coverage for every dependency on Linux; see §13. |
| Node | 22.16.0 | Only needed to build the front end. Not needed to run it. |
| npm | 10.9.2 | |
| Disk | ~1.0 GB | Measured on 2026-09-03: 972 MB for the whole tree — 694 MB of virtualenv (673 MB of it `site-packages`), 131 MB of `node_modules`, 56 MB of `data/` and 63 MB of `results/`. |
| RAM | 2 GB minimum, 4 GB comfortable | ~550 MB is the three preloaded road graphs; set `QROUTE_API_PRELOAD=none` to avoid paying it. |
| Network | For installation, and for `qroute osm fetch` | The service itself makes no outbound calls at run time. |

No compiler and no system packages are required: every dependency ships a
binary wheel for Linux and macOS on both x86-64 and arm64.

---

## 3. Local development

From a clean clone:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,baselines]"
```

`dev` adds pytest, ruff and httpx. `baselines` adds PyVRP, the state-of-the-art
reference solver that appears in every published results table. PyVRP is
optional — every call into it is guarded and raises `PyVRPUnavailable` with an
install hint rather than silently substituting another solver — but without it
four tests in `tests/test_exact.py` skip themselves and `configs/main.yaml`
cannot be reproduced, because it lists `pyvrp` among its algorithms.

Then fill in the two things a clone does not carry:

```bash
scripts/fetch_data.sh
```

That script does two independent, idempotent things, and explains both when run
with `--help`:

1. Expands `results/runs/main/rows.jsonl.gz` (4.9 MB) into `rows.jsonl`
   (43 MB, 1,530 rows). `/api/benchmarks` looks for the **uncompressed** file
   and lists nothing without it, so the results view in the UI is blank until
   this has been done.
2. Runs `qroute osm fetch`, which rebuilds the three road graphs from the
   recipe in `data/osm/networks.json`. This needs internet access and takes
   about a minute per network. A graph already on disk is skipped; pass
   `--force` to re-download.

Build the front end and start the server:

```bash
cd frontend && npm ci && npm run build && cd ..
qroute serve
```

Open <http://127.0.0.1:8000>.

For front-end work, run the two halves separately. Vite proxies `/api` to the
API process, so the browser stays same-origin and `EventSource` needs no
special handling:

```bash
qroute serve            # terminal 1
cd frontend && npm run dev   # terminal 2, http://127.0.0.1:5173
```

Point the proxy elsewhere with `QROUTE_API=http://host:port npm run dev`.

Run the tests:

```bash
python -m pytest tests/ -q
```

Observed here on 2026-09-03: exit 0, 526 tests collected, 525 passed and one
skipped, in a little over six minutes on ten cores (375 s measured on a
slightly smaller revision of the suite). Much of that time is solvers running
against real wall-clock budgets, so a smaller machine takes proportionally
longer rather than failing. The count grows as the suite does; it is recorded
here so that a run which is *quietly smaller* — a collection error that hid a
file, say — is visible as well as a run that is red.

---

## 4. Production, option A: Docker Compose

This is the recommended single-host deployment. It is one command to build, one
to run, and the image carries its own Python, its own built front end and the
committed benchmark results.

```bash
docker compose build                                  # slow the first time
docker compose --profile setup run --rm osm-fetch     # optional, ~3 minutes
docker compose up -d
curl -fsS http://127.0.0.1:8000/api/health
```

`docker-compose.yml` publishes the port on `127.0.0.1` only. See
[§7](#7-exposing-it-to-a-network) before changing that.

Useful operations:

```bash
docker compose logs -f qroute        # follow the log
docker compose ps                    # includes the HEALTHCHECK state
docker compose restart qroute
docker compose down                  # stop; the road-graph volume survives
docker compose down -v               # stop and discard the road-graph volume
docker compose exec qroute qroute version
```

### How the road graphs are handled, and why

The three road graphs are about 40 MB of GraphML. They are not in version
control: `data/osm/networks.json` records the exact OSMnx recipe and
`qroute osm fetch` regenerates them. A deployment has to decide what to do
about that. The three options and the reasoning:

* **Fetch at build time.** Rejected. It would make every image build depend on
  a live Overpass API, add about three minutes to the build, and — worse —
  produce a *different image from the same Dockerfile* every time somebody
  edits OpenStreetMap. An image you cannot rebuild identically is not much of
  an artefact. A build behind a firewall, or during an Overpass outage, would
  simply fail.
* **Bake in a pre-fetched copy.** Rejected. It means committing 40 MB of
  regenerable data to git to get it into the build context, which is exactly
  what `.gitignore` exists to prevent.
* **A named volume, filled by an explicit one-shot command.** Chosen. The
  `osm-graphs` volume is mounted at `/app/data/osm`. Docker populates a new
  named volume from the image's copy of that directory, which holds
  `networks.json` and `index.json` and is owned by uid 10001, so the container
  can write into it as its non-root user with no `chown` on the host. The
  `osm-fetch` service — behind a `setup` profile, so `docker compose up` can
  never trigger a download — fills it once. Every later `up` reuses it.

Until it has been filled, the service **starts and works**. This was verified
directly, by running the server against a data root containing `benchmarks/`
and no `osm/`:

```
WARNING qroute.api  no road graphs under .../data/osm; the map and network
                    endpoints will be empty until `qroute osm fetch` has been run
```

`/api/health` reports `"networks": 0` and `"network_ids": []`, `/api/networks`
returns `[]`, and a solve submitted to `/api/runs` completed normally and
reached the best known solution. The degradation is confined to the map and the
road-network demonstration; the benchmark instances, every one of the eleven
solvers `/api/algorithms` lists and the rest of the API are unaffected.

One consequence of the volume approach worth knowing: an existing named volume
is **not** refreshed from a newer image. If `networks.json` ever changes, run
`docker compose down -v` and fetch again.

### What is in the image and what is not

| In | Not in |
| --- | --- |
| Python 3.13 + the installed package and its dependencies | Node, npm, `node_modules` |
| PyVRP (the `baselines` extra) | the test suite, `docs/`, `.git` |
| the built front end at `/app/frontend/dist` | the road graphs (volume; see above) |
| `data/benchmarks` — 277 files, 1.4 MB | a compiler or any `-dev` package |
| `configs/` and `results/runs/` including the expanded `rows.jsonl` | |
| a pre-warmed numba kernel cache | |

Expect roughly 1.3–1.6 GB. The bulk is unavoidable: llvmlite (125 MB), SciPy
(81 MB), pyogrio (74 MB) and OR-Tools (66 MB) alone account for most of it,
measured in the development virtualenv.

---

## 5. Production, option B: systemd, without Docker

For a plain VM. Adjust the paths; this assumes the checkout is at
`/srv/qroute` and is owned by a `qroute` service user.

```bash
# The clone comes first: useradd --create-home would leave /srv/qroute
# non-empty, and git refuses to clone into that.
sudo git clone <repository> /srv/qroute
sudo useradd --system --home-dir /srv/qroute --shell /usr/sbin/nologin qroute
sudo chown -R qroute:qroute /srv/qroute
cd /srv/qroute

# A non-editable install, unlike the development one: production should run the
# package as installed, not a link back into a working tree someone might edit.
# The systemd unit below sets QROUTE_DATA absolutely, which is what makes that
# safe — a package in site-packages has no data directory beside it.
sudo -u qroute python3 -m venv .venv
sudo -u qroute .venv/bin/pip install ".[baselines]"

sudo -u qroute .venv/bin/qroute osm fetch --out-dir /srv/qroute/data/osm   # optional, see §4
sudo -u qroute .venv/bin/python -m gzip -d results/runs/main/rows.jsonl.gz

cd frontend && sudo -u qroute npm ci && sudo -u qroute npm run build
```

`/etc/systemd/system/qroute.service`:

```ini
[Unit]
Description=qroute — quantum-inspired route optimisation API and web UI
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=qroute
Group=qroute
WorkingDirectory=/srv/qroute

# Absolute paths, so the service does not depend on its working directory.
Environment=QROUTE_DATA=/srv/qroute/data
Environment=QROUTE_RESULTS=/srv/qroute/results/runs
Environment=QROUTE_FRONTEND=/srv/qroute/frontend/dist
Environment=QROUTE_LOG_FORMAT=json
Environment=QROUTE_MAX_ACTIVE_RUNS=4

ExecStart=/srv/qroute/.venv/bin/qroute serve --host 127.0.0.1 --port 8000

# The server forks a fork server, which forks one process per solver run.
# KillMode=control-group (the default) is what makes a stop take the whole
# tree with it rather than leaving workers behind.
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

# Hardening. The service writes only to its results directory and to the road
# graph directory during a fetch; everything else can be read-only.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/srv/qroute/results /srv/qroute/data/osm

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qroute
systemctl status qroute
journalctl -u qroute -f
```

**This unit file has not been run.** It is written from the process's actual
behaviour — the paths it reads, the directories it writes, the child processes
it creates — but no systemd host was available here. Treat the hardening
directives in particular as a starting point: if the service fails to start
with a permission error, relax `ProtectSystem` first.

A reverse proxy in front of it must not buffer responses, or the live
convergence stream arrives in one lump when the run finishes. For nginx:

```nginx
# This is the override, not a whole server block: it must sit alongside a
# catch-all `location / { proxy_pass http://127.0.0.1:8000; }`, or nginx will
# serve /api/runs/ and 404 everything else. The prefix is chosen to cover
# /api/runs/{run_id}/stream, which is the only streaming route.
location /api/runs/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;   # longer than QROUTE_MAX_RUN_SECONDS
    proxy_set_header Connection '';
    proxy_http_version 1.1;
}
```

**Untested.** No nginx was available here. The directives are the standard ones
for Server-Sent Events and the prefix was checked against the route table in
`qroute/api/app.py`, but the block has not been put in front of a running
server.

---

## 6. Environment variables

All of these are parsed once, at first use, by `qroute/config.py`. A value that
is present but unusable raises `ConfigError` and stops the process with a
message naming the variable — it never falls back to the default silently. An
explicitly empty variable is treated as unset.

| Variable | Default | What it does |
| --- | --- | --- |
| `QROUTE_DATA` | the `data` directory beside the installed package, else the nearest one at or above the working directory | Root of the benchmark instances (`benchmarks/`) and the road graphs (`osm/`). |
| `QROUTE_RESULTS` | `<QROUTE_DATA>/../results/runs` | Where `qroute bench` writes and `/api/benchmarks` reads. Missing is not an error; it means no benchmark has been run. |
| `QROUTE_FRONTEND` | `<QROUTE_DATA>/../frontend/dist` | The built single-page application. Missing is not an error; `/` returns a JSON stub instead. |
| `QROUTE_HOST` | `127.0.0.1` | Interface to bind — **but see the note below.** |
| `QROUTE_PORT` | `8000` | Port to bind — **but see the note below.** |
| `QROUTE_LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `QROUTE_LOG_FORMAT` | `text` | `text` for a human, `json` for one JSON object per line. |
| `QROUTE_REQUEST_LOG` | `1` | One log line per HTTP request with its status and server-side duration. |
| `QROUTE_CORS_ORIGINS` | *(empty — same-origin only)* | Comma-separated origins allowed to call the API cross-origin. |
| `QROUTE_CORS_ORIGIN_REGEX` | *(unset)* | A regular expression matched against the whole `Origin` header, for ephemeral preview deployments. |
| `QROUTE_MAX_ACTIVE_RUNS` | `4` | How many solver processes may run at once. Each pins one core. Range 1–64. |
| `QROUTE_MAX_RUN_SECONDS` | `300` | Longest wall-clock budget one run may hold a worker for. A request asking for more is clamped and told so, rather than rejected. The wire schema independently caps the field at 600 s. |
| `QROUTE_API_PRELOAD` | `all` | Which road graphs to load at startup: `all`, `none`, `first`, or a comma-separated list of network ids. |

Two variables are read elsewhere and are worth knowing about:

* `NUMBA_CACHE_DIR` — where numba writes its compiled kernels. The Docker image
  sets it to a writable path, because the default is next to the module source,
  which in that image is root-owned `site-packages`; without it every process
  would silently recompile. Verified here: setting it produced 19 cache files
  (980 KB) rather than none.
* `TOMTOM_API_KEY` — read by `qroute.traffic.sources.TomTomFlowSource`. Nothing
  in the API or the CLI constructs that source today, so setting it has no
  effect on a running service. It is listed here so that finding it in the code
  does not look like a missing configuration step.

### The `QROUTE_HOST` / `QROUTE_PORT` trap

`qroute serve` does **not** read them. Its `--host` and `--port` options have
hard-coded defaults of `127.0.0.1` and `8000`, so:

```
$ QROUTE_PORT=8123 qroute serve
qroute API qroute.api.app:app on http://127.0.0.1:8000
```

That is the observed output, not a hypothetical. Pass the flags:

```bash
qroute serve --host 0.0.0.0 --port 8123
```

The variables *are* honoured by the alternative entry point
`python -m qroute.api.app`, which reads them from the settings object. Both the
Dockerfile and the systemd unit above use explicit flags, so neither depends on
which of the two you happen to know.

---

## 7. Exposing it to a network

The default bind is loopback on purpose. The API is **unauthenticated** and a
single request can commit a CPU core for up to `QROUTE_MAX_RUN_SECONDS`. Before
putting it on a network:

* Put a reverse proxy in front of it and terminate TLS there.
* Add authentication at the proxy. The application has none and does not
  pretend to.
* Lower `QROUTE_MAX_ACTIVE_RUNS` and `QROUTE_MAX_RUN_SECONDS` to what the host
  can actually absorb. Those two are the whole of the resource control.
* In `docker-compose.yml`, change `"127.0.0.1:8000:8000"` to `"8000:8000"` only
  once the above is true.

Deliberately absent from the HTTP surface: the startup banner that lists
filesystem paths is written to the log, not exposed by any endpoint, and an
unhandled exception returns an eight-character error id and nothing else — the
traceback goes to the log under that id, where an operator can find it and a
browser cannot.

---

## 8. Where the data lives

| Path | Size | In git? | What it is |
| --- | --- | ---: | --- |
| `data/benchmarks/` | 1.4 MB | yes | 138 CVRPLIB and Solomon instances (82 CVRP, 56 VRPTW) with a best-known solution beside each: 277 files in all. `/api/health` reports the 138. |
| `data/osm/networks.json` | 2 KB | yes | The recipe `qroute osm fetch` rebuilds the road graphs from. |
| `data/osm/*.graphml` | 40 MB | **no** | The three road graphs. See §4. |
| `data/cache/` | 15 MB | **no** | osmnx's HTTP cache. Purely a fetch accelerator; delete it freely. |
| `configs/*.yaml` | 24 KB | yes | Benchmark sweep definitions. |
| `results/runs/main/rows.jsonl.gz` | 4.9 MB | yes | The definitive sweep, 1,530 runs, gzipped. |
| `results/runs/main/rows.jsonl` | 43 MB | **no** | The same, expanded. `/api/benchmarks` needs this file. |
| `results/runs/main/` (rest) | 15 MB | yes | `summary.json`, `report.md`, the CSVs and the figures. |
| `frontend/dist/` | 868 KB | **no** | The built application. Rebuilt by `npm run build`. |

### Rebuilding the road graphs

```bash
qroute osm fetch                       # all three, skipping any already present
qroute osm fetch --network delhi_connaught --force
qroute osm fetch --out-dir /srv/qroute/data/osm
```

Observed output when everything is already present:

```
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┓
┃ network               ┃ result          ┃ time ┃ note ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━┩
│ bengaluru_koramangala │ already present │      │      │
│ delhi_connaught       │ already present │      │      │
│ chennai_annanagar     │ already present │      │      │
└───────────────────────┴─────────────────┴──────┴──────┘
```

A fresh fetch prints the node and edge counts and compares them against the
counts recorded in `networks.json`. OpenStreetMap is edited continuously, so
small drift is expected and is reported as a percentage; a large deviation
means something other than normal editing changed.

Inside the container:

```bash
docker compose --profile setup run --rm osm-fetch
```

---

## 9. Reproducing the benchmark

```bash
scripts/run_final_benchmark.sh                       # configs/main.yaml
scripts/run_final_benchmark.sh configs/quick.yaml    # a two-minute smoke sweep
```

The script prints where the sweep will land before starting it, warns if the
config asks for PyVRP and PyVRP is not installed, runs `qroute bench`, and then
generates the tables and figures with `qroute report`. Equivalently, by hand:

```bash
qroute bench --config configs/main.yaml
qroute report results/runs/main --format markdown --plots
```

Read `docs/benchmarking.md` for the protocol. Two operational points:

* **Run it on an idle machine.** Every solver is pinned to one thread and
  compared by wall clock, so background load does not slow the sweep down
  evenly — it slows down whichever solver was scheduled against it.
* **`workers` is unset in `configs/main.yaml` on purpose**, so the runner sizes
  itself to the host and leaves two cores free. The published sweep used 9
  workers on a 10-core machine; `results/runs/main/meta.json` records that,
  along with the versions of everything the numbers depend on.

`qroute bench` refuses to write into a directory that already holds results.
Give a new run its own `--name`, or pass `--force`, which renames the old rows
aside rather than deleting them.

`configs/main.yaml` is about an hour on ten cores. `configs/quick.yaml` is 60
runs and about a minute; use it to check that the whole pipeline works before
committing an hour to it.

---

## 10. Checking that it is healthy

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

A healthy response, taken from a running server here:

```json
{"status":"ok","version":"0.1.0","uptime_seconds":6048.2,"networks":3,
 "instances":138,"algorithms":7,
 "network_ids":["bengaluru_koramangala","chennai_annanagar","delhi_connaught"],
 "networks_loaded":["bengaluru_koramangala","chennai_annanagar","delhi_connaught"],
 "networks_loading":[],"ortools_available":true,"pyvrp_available":true,
 "pyvrp_version":"0.14.0",
 "warmup":{"done":true,"seconds":0.426,"detail":"qpso on a 8-customer instance, best cost 347.19"},
 "active_runs":0,"worker_start_method":"forkserver","workers_primed":true,
 "worker_prime_seconds":1.199}
```

What to look at:

| Field | Healthy | If not |
| --- | --- | --- |
| `status` | `"ok"` | The only value the endpoint returns; anything else means you are not talking to this service. |
| `networks`, `network_ids` | 3 | 0 means the road graphs are absent — expected before `qroute osm fetch`, see §4. |
| `networks_loaded` vs `networks_loading` | all loaded | During the first ~30 s of startup a graph appears in `loading`. That is the background preload, not a fault. |
| `instances` | 138 | 0 means `QROUTE_DATA` is wrong. The startup banner in the log names the directory it searched. |
| `warmup.done` | `true` | `false` for the first second or two after boot while the JIT kernels compile. |
| `workers_primed` | `true` | `false` means the fork server did not start; solver runs will fail. |
| `worker_start_method` | `forkserver` | `spawn` is the documented fallback and is correct on platforms without a fork server. |
| `ortools_available`, `pyvrp_available` | `true` | `false` means that solver is not installed. The rest of the platform is unaffected; comparisons against it are not available. |
| `active_runs` | ≤ `QROUTE_MAX_ACTIVE_RUNS` | At the limit, new runs are refused with a message saying to cancel one or wait. |

The container's `HEALTHCHECK` runs the same check. It is written in Python
rather than curl because `python:slim` has no curl, and it asserts both the 200
and `status == "ok"` so that a proxy returning 200 for its own error page
cannot pass for a healthy service. Verified against a live server (exit 0) and
a stopped one (exit 1).

A deeper check, which exercises a solver process end to end:

```bash
curl -fsS -X POST http://127.0.0.1:8000/api/runs \
  -H 'content-type: application/json' \
  -d '{"instance":"A-n32-k5","algorithm":"qpso","max_seconds":2,"seed":1}'
# -> {"run_id":"...","max_seconds":2.0,"clamped":null}
sleep 4
curl -fsS http://127.0.0.1:8000/api/runs/<run_id>
# -> "state":"done", "best_cost":784.0, "bks":784.0, "feasible":true
```

Those are the values observed here. `A-n32-k5` has a best known solution of
784, and two seconds of QPSO reaches it, so this is a check with a right
answer rather than a check that something came back.

---

## 11. When it is not healthy

**Read the startup banner first.** Every start writes one, and it answers most
questions before you have to ask them (log-line prefixes trimmed here):

```
qroute 0.1.0 starting
  data root          /srv/qroute/data (present)
  benchmarks         /srv/qroute/data/benchmarks (present)
  road graphs        /srv/qroute/data/osm (missing, run `qroute osm fetch`)
  results root       /srv/qroute/results/runs (missing, no benchmark run yet)
  frontend           /srv/qroute/frontend/dist (built)
  cors origins       (none)
  max active runs    4
  max run seconds    300
  network preload    all
  log level/format   INFO/text
```

| Symptom | Cause and fix |
| --- | --- |
| `/` returns JSON instead of the application | `frontend/dist` is missing or `QROUTE_FRONTEND` points at the wrong place. The banner's `frontend` line says which. `cd frontend && npm run build`. |
| `"instances": 0`, and `qroute instances` says nothing was found | `QROUTE_DATA` is wrong or unset. The error message lists every directory it searched. |
| Map is empty, `"networks": 0` | Road graphs absent. Expected before `qroute osm fetch`; §4. |
| Benchmark view is empty but `results/runs/main` exists | `rows.jsonl` has not been expanded from `rows.jsonl.gz`. `/api/benchmarks` requires the uncompressed file. Run `scripts/fetch_data.sh`, or `python -m gzip -d results/runs/main/rows.jsonl.gz`. |
| Benchmark view works but shows fewer than 1,530 runs | `rows.jsonl` is truncated. Both the reader and `/api/benchmarks` tolerate a short final line on purpose — a killed sweep's good rows are still evidence — so a partial file produces a smaller table rather than an error. Check with `wc -l results/runs/main/rows.jsonl`; if it is not 1530, delete it and re-run `scripts/fetch_data.sh`, which expands to a temporary file and renames only on success. |
| Every run returns `status=worker_died` | The parent process's `__main__` cannot be re-executed by a spawned child. This happens under a heredoc (`python - <<'PY'`) or a REPL, and it is why `scripts/run_final_benchmark.sh` drives the installed `qroute bench` command instead. Run benchmarks through the CLI, not through an ad-hoc script. |
| Runs are refused with "already in flight" | `active_runs` has reached `QROUTE_MAX_ACTIVE_RUNS`. Cancel one, wait, or raise the limit — but only alongside more cores. |
| Live convergence arrives all at once at the end | A proxy is buffering the Server-Sent Events stream. See the nginx block in §5. |
| First solve after a restart is slow | numba is compiling. `warmup.done` in `/api/health` reports it. In the container the kernels are pre-compiled at build time; a host whose CPU differs from the build machine's will recompile once. |
| A 500 with an `error_id` | The full traceback is in the log under that id: `journalctl -u qroute | grep <error_id>`, or `docker compose logs qroute | grep <error_id>`. The same value is on the response's `X-Request-Id` header. |
| The container is `unhealthy` | `docker compose logs qroute`. If the log ends at the startup banner, the process is still preloading road graphs — the health check's 45 s start period should cover it, but a slow disk may not. |
| Memory climbing steadily | The three road graphs together settle at roughly 550 MB resident once loaded, and generated instances are cached (bounded, oldest evicted). `QROUTE_API_PRELOAD=none` avoids the graph cost until a graph is actually asked for. |

Turn up the detail with `QROUTE_LOG_LEVEL=DEBUG`, and switch to
`QROUTE_LOG_FORMAT=json` if a log shipper is reading it.

---

## 12. Continuous integration

`.github/workflows/ci.yml` is one job on `ubuntu-latest`. It installs the
package with `pip install -e ".[dev,baselines]"`, records `qroute version`,
runs the Python test suite, installs the front end with `npm ci`, type-checks
it with `tsc -b` as its own step so a type error fails on a step called
"type-check", builds it, and then starts the API and polls `/api/health` until
it answers — the same endpoint the container's health check uses. Two final
steps assert the two properties that unit tests do not cover: that `/` serves
HTML rather than the JSON stub, and that the `qroute` command works from a
directory that is not the checkout, given `QROUTE_DATA`.

Deliberately one job and one configuration. A matrix over three Python versions
and two operating systems would triple the wall clock for a project that is
deployed on exactly one configuration, and cells nobody reads are decoration
rather than evidence.

`ruff` is installed by the `dev` extra but is not a gate. Measured on
2026-09-03, `ruff check qroute tests` reports 359 findings. The great majority
are stylistic and mechanically fixable — 191 are `UP045`, the pre-PEP-604
`Optional[...]` spelling, and 277 of the 359 are `--fix`-able. Two groups are
not merely stylistic and deserve a human reading rather than a bulk rewrite:
27 `BLE001` (`except Exception:`) and 5 `S110` (`try: ... except: pass`). Some
of those are deliberate — a solver that must survive one bad instance — and
some may not be; the point is that nobody has been through them.

That is the reason there is no lint gate rather than an excuse for one. A gate
that is red on the day it lands trains everyone to ignore it, and turning it
green by running `--fix` over 277 findings would bury the 32 that need an
opinion. Both belong in their own change. The count above will drift as the
code moves; re-run the command rather than trusting the number.

---

## 13. Verifying the image without building it

Docker is not installed here, so the following are the checks that *were*
possible, and their actual results. Anyone with a daemon should run
`docker build .` and replace this section with the outcome.

**Wheel availability**, which is the failure mode that would only appear at
build time. Ask pip to resolve exactly what the image installs, for the image's
platform, so the answer is what pip would really do inside the container rather
than an assumption. This resolves the *whole* dependency closure, not a
hand-picked list of the packages that look risky — the ones that catch you out
are the transitive ones nobody thought to name:

```bash
pip install --dry-run --ignore-installed --only-binary=:all: \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_34_x86_64 \
  --python-version 3.13 --implementation cp --abi cp313 \
  --target /tmp/t --report /tmp/report.json ".[baselines]"
```

59 packages resolve on `linux/amd64` cp313, 59 on `linux/arm64` cp313 with the
`aarch64` platform tags, and 59 again on `linux/amd64` cp312 — so the 3.12
downgrade mentioned in §2 is real and not an assumption. Every one is a wheel;
`qroute` itself is the only sdist, and it is this source tree. Among them are
the compiled transitive
dependencies a shortlist would have missed — uvloop, httptools, watchfiles,
pydantic-core, pyogrio, pillow, protobuf — alongside numba 0.67.0, OR-Tools
9.15.6755, NumPy 2.5.2, SciPy 1.18.1, llvmlite 0.49.0, pyproj 3.7.2, Shapely

The strictest wheel in each closure needs `manylinux_2_28`, i.e. glibc ≥ 2.28 —
pyproj on x86-64, pyogrio on aarch64. Debian bookworm ships 2.36, which is why
the base image tag is pinned to `python:3.13-slim-bookworm` rather than a
floating `python:3.13-slim`.

This proves the dependencies are installable on the target platform. It does
not prove the image builds: that `pip install` is one of 38 instructions in the
Dockerfile, and Docker has executed none of them.

Asking for the same thing against musl finds nothing at all:

```bash
pip download --no-deps --only-binary=:all: \
  --platform musllinux_1_2_x86_64 --python-version 3.13 --implementation cp \
  -d /tmp/wheels "numba>=0.60"
ERROR: Could not find a version that satisfies the requirement numba>=0.60
```

which is the whole reason the image is not Alpine-based.

**Everything the image runs** was executed directly on the host, outside a
container, in the same order the Dockerfile runs it: `qroute version`,
`qroute solve A-n32-k5 --seconds 1` (exit 0), `python -m gzip -d` on the
committed rows file (1,530 rows recovered), the health probe against a live
server (exit 0) and against a stopped one (exit 1), and a server started
against a data root with no road graphs (starts, warns, serves).

**The build context** was measured by applying `.dockerignore`'s patterns to
the tree with Docker's own matching rules — which are *not* git's: a
`.dockerignore` pattern is anchored at the context root, so `__pycache__/`
there means "a directory of that name at the top level", not "anywhere".
Written the git way the context was 616 files and 25.6 MB and carried 97 stale
`.pyc` files into the image; with the `**/` prefixes the file now uses it is
565 files and 22.7 MB with none. Both figures come from the same simulation,
and every path a `COPY` instruction names was checked to be still present in
the filtered set. What a real `docker build` uploads has not been observed.

**Not verified**: the image builds, its layers are the size estimated above,
the `HEALTHCHECK` transitions to `healthy` in Docker's own state machine, the
named volume is populated from the image with uid 10001 ownership as expected,
and `docker compose` accepts the compose file. The compose file's YAML parses
and its structure was checked by hand; `docker compose config` was not run.
