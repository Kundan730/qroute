# Deploying qroute to Hugging Face Spaces

The whole process, start to finish. It takes about twenty minutes, most of which
is the image building on Hugging Face's side.

**No API key is required.** Nothing in the platform needs credentials to run:
the base maps come from Esri's free tile services, the benchmark instances ship
with the image, and traffic is simulated. A live traffic feed is supported and
optional; see step 7.

---

## Why Spaces

Its free CPU tier gives 2 vCPU and 16 GB of memory. The service needs about
313 MB idle and roughly 700 MB with one city loaded, so there is a great deal of
headroom. Spaces builds a Dockerfile directly, gives a public URL that can be
handed to a judge, and only sleeps after 48 hours of inactivity rather than the
15 minutes typical of other free tiers.

---

## 1. Create the Space

At <https://huggingface.co/new-space>:

| Field | Value |
| --- | --- |
| Owner | your account |
| Space name | `qroute` |
| License | MIT |
| SDK | **Docker** — not Gradio or Streamlit |
| Template | Blank |
| Hardware | CPU basic, free |
| Visibility | Public |

## 2. Clone it

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/qroute hf-qroute
cd hf-qroute
```

If you are asked for a password, use an access token from
<https://huggingface.co/settings/tokens> with write permission, not your account
password.

## 3. Copy the project in

From the directory holding this repository:

```bash
rsync -a --exclude='.git' --exclude='.venv' --exclude='node_modules' \
      --exclude='frontend/dist' --exclude='data/osm/*.graphml' \
      ./ ../hf-qroute/
```

`.venv` and `node_modules` are rebuilt inside the image, `frontend/dist` is
built during the image build, and the road graphs are fetched at runtime.

## 4. Add the Space's README

Hugging Face reads its configuration from YAML at the top of `README.md`. Copy
the prepared one over the project's own:

```bash
cp deploy/huggingface/README.md README.md
```

The block that matters is `sdk: docker` and `app_port: 8000`. The port must
match what the container listens on, which the root `Dockerfile` sets through
`QROUTE_PORT`.

## 5. Enable Git LFS for the large files

The committed benchmark log is a few megabytes and Hugging Face wants files over
10 MB in LFS:

```bash
git lfs install
git lfs track "results/runs/main/rows.jsonl.gz"
git lfs track "*.graphml"
git add .gitattributes
```

## 6. Push

```bash
git add -A
git commit -m "Deploy qroute"
git push
```

The build starts immediately. Watch it under the **Logs** tab of the Space. It
takes ten to fifteen minutes: most of it is installing SciPy, OR-Tools and
Numba, and the final step solves a benchmark instance to compile the kernels
into the image so the first visitor does not wait for them.

## 7. Optional: live traffic

Everything works without this. To replace the simulated traffic with a live
feed, add a free key from <https://developer.tomtom.com> under
**Settings → Variables and secrets → New secret**:

| Name | Value |
| --- | --- |
| `TOMTOM_API_KEY` | your key |

Use a *secret*, not a variable: variables are visible to anyone who can see the
Space. Without the key the platform falls back to simulation and labels itself
as simulated rather than pretending otherwise.

## 8. Check it worked

```bash
curl https://YOUR_USERNAME-qroute.hf.space/api/health
```

A healthy response reports `"status": "ok"` along with the number of instances
and networks it found. Then open the Space itself: the map should draw a city,
the time-of-day slider should recolour the roads, and **Generate instance**
followed by **Optimise** should produce routes in about ten seconds.

---

## Settings you may want

Set these under **Settings → Variables and secrets** as plain variables.

| Variable | Default in the image | Why change it |
| --- | --- | --- |
| `QROUTE_API_PRELOAD` | `all` | `none` starts faster and uses ~313 MB instead of ~1.3 GB. Worth setting on a constrained host. |
| `QROUTE_MAX_ACTIVE_RUNS` | `4` | Lower it if several visitors solving at once make the Space sluggish. |
| `QROUTE_MAX_RUN_SECONDS` | `300` | The longest single run a visitor may request. |
| `QROUTE_LOG_FORMAT` | `text` | `json` if you want to parse the logs. |

## When it does not work

**The build fails while installing.** Check the Logs tab. The usual cause is a
dependency with no wheel for the platform; every dependency here resolves to a
manylinux wheel, so this should not happen, but the log names the package.

**The Space builds and then shows a blank page.** The port is wrong. `app_port`
in `README.md` must equal `QROUTE_PORT` in the image, both 8000 by default.

**The map has no roads.** The road graphs are fetched on first use and need
outbound network access, which Spaces allows. Check the logs for a message from
`qroute osm fetch`. The benchmark instances work regardless, so the Solver and
Benchmark pages are unaffected.

**Everything is slow on the first solve.** The compiled kernels are baked in at
build time, but Numba keys its cache on the CPU it compiled for. If the Space
runs on a different processor than the builder, the first solve recompiles once
and every later one is fast. It is a slow first request, never an error.
