# Deploying qroute to Hugging Face Spaces

The complete process. Budget about forty minutes, most of which is the image
building on Hugging Face's side while you do something else.

## Before you start: this needs PRO

Hugging Face made Docker and Gradio Spaces a paid feature. **Static Spaces are
free; Docker Spaces are not.** qroute is a Python service with a compiled
optimiser, so it needs Docker, which means a PRO account at **$9/month**,
cancellable whenever you like.

What that buys, and why it is a reasonable trade for this project:

* 16 GB of memory against the roughly 410 MB the service needs, so memory never
  becomes something you think about.
* A public HTTPS URL a judge can open, with no server for you to administer.
* Nothing to keep patched, no firewall to configure, no certificate to renew.

If you would rather not pay, `deploy/vps/DEPLOY.md` runs the same image on a
Hetzner box for about €4/month, and `deploy/oracle/DEPLOY.md` covers Oracle's
permanently free tier. Both need a card or PayPal and rather more setup.

**No API key is required for the application itself.** Base maps come from
Esri's free tile services, the benchmark instances ship inside the image, and
traffic is simulated. A live traffic feed is optional; see step 8.

---

## 1. Subscribe to PRO

<https://huggingface.co/pricing> → **Get Pro**. Payment is handled by Stripe,
which accepts most debit as well as credit cards.

## 2. Create the Space

<https://huggingface.co/new-space>:

| Field | Value |
| --- | --- |
| Owner | your account |
| Space name | `qroute` |
| Short description | Quantum-inspired vehicle routing on real road networks |
| License | `mit` |
| Space SDK | **Docker** → **Blank** |
| Hardware | CPU basic (free with PRO) |
| Visibility | **Public** |

Public matters: a private Space cannot be opened by a judge without an account.

## 3. Choose how the code gets there

Two ways, and the first is much less work once it is set up.

### Automatically, from GitHub (recommended)

Hugging Face has no "connect your repository" button the way Vercel and Render
do, because a Space *is* a git repository: deploying means pushing to it. The
workflow at `.github/workflows/deploy-space.yml` does that push for you, so a
commit to `main` on GitHub rebuilds the Space. It also handles the two
substitutions the Space needs, which is why it exists rather than you keeping a
second remote and remembering to edit two files each time.

Set it up once:

1. Create a Hugging Face token with **write** permission at
   <https://huggingface.co/settings/tokens>.
2. On GitHub, go to **Settings → Secrets and variables → Actions → New
   repository secret**, name it `HF_TOKEN`, and paste the token.
3. Check the `HF_USERNAME` and `HF_SPACE` values at the top of the workflow
   match your Space.

Then either push a commit, or run it by hand from the **Actions** tab with
**Run workflow**. Skip to step 8; steps 4 to 7 are what the workflow automates.

### By hand

Fine for a one-off, and the rest of this section walks it.

## 3a. Clone the Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/qroute hf-qroute
cd hf-qroute
```

When it asks for a password, give an access token from
<https://huggingface.co/settings/tokens> with **write** permission. Your account
password will not work.

## 4. Copy the project across (manual route)

From the directory that holds this repository:

```bash
rsync -a --exclude='.git' --exclude='.venv' --exclude='node_modules' \
      --exclude='frontend/dist' --exclude='data/osm/*.graphml' \
      --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
      ./ ../hf-qroute/
```

Everything excluded is either rebuilt during the image build or fetched by it.

## 5. Bake the road graphs into the image (manual route)

This step is specific to Spaces and matters. Spaces gives no persistent disk on
the standard tiers, so anything fetched at runtime is lost on every restart, and
the next visitor waits several minutes looking at a map with no roads on it.

In the Space copy, turn on the build flag that puts them in the image:

```bash
cd ../hf-qroute
sed -i '' 's/^ARG BAKE_OSM=0$/ARG BAKE_OSM=1/' Dockerfile   # macOS
# on Linux: sed -i 's/^ARG BAKE_OSM=0$/ARG BAKE_OSM=1/' Dockerfile
grep -n 'ARG BAKE_OSM' Dockerfile      # confirm it now reads 1
```

It adds about 40 MB to the image and three minutes to the build, and the
container then starts ready to draw a city.

## 6. Add the Space's README (manual route)

Hugging Face reads its configuration from YAML at the top of `README.md`:

```bash
cp deploy/huggingface/README.md README.md
```

The two lines that matter are `sdk: docker` and `app_port: 8000`. The port must
match what the container listens on, which the Dockerfile sets through
`QROUTE_PORT`.

## 7. Push (manual route)

The benchmark log is over Hugging Face's 10 MB inline limit, so track it in LFS
first:

```bash
git lfs install
git lfs track "results/runs/main/rows.jsonl.gz"
git add .gitattributes
git add -A
git commit -m "Deploy qroute"
git push
```

The build starts immediately. Watch it under the Space's **Logs** tab, then
**Building**. Expect fifteen to twenty-five minutes: installing SciPy, OR-Tools
and Numba is most of it, then the frontend build, then a real benchmark solve
that compiles the optimiser's kernels into the image so your first visitor does
not pay for them, then the road graphs.

## 8. Optional: live traffic

Everything works without this; traffic is simulated from a calibrated
time-of-day profile and the interface says so. To use a live feed instead, get a
free key from <https://developer.tomtom.com> and add it under
**Settings → Variables and secrets → New secret**:

| Name | Value |
| --- | --- |
| `TOMTOM_API_KEY` | your key |

Use a **secret**, not a variable. Variables are visible to anyone who can see
the Space. Without the key the platform falls back to simulation and labels
itself accordingly rather than presenting simulated data as observed.

## 9. Check it

```bash
curl https://YOUR_USERNAME-qroute.hf.space/api/health
```

A healthy reply carries `"status": "ok"` and the instance and network counts.
Then open the Space itself and walk the demo:

1. The map draws a city with roads coloured by congestion.
2. Moving the time-of-day slider recolours them.
3. **Generate instance** places a depot and stops on real intersections.
4. **Optimise** returns routes in about ten seconds, with the search visible
   while it runs.
5. The **Benchmark** tab shows the 1,520-run results.

---

## Settings worth knowing

Under **Settings → Variables and secrets**, as plain variables:

| Variable | Default | When to change it |
| --- | --- | --- |
| `QROUTE_API_PRELOAD` | `all` | `first` starts faster; `none` uses least memory. With 16 GB, `all` is fine. |
| `QROUTE_MAX_ACTIVE_RUNS` | `4` | Lower if several visitors solving at once feels sluggish. |
| `QROUTE_MAX_RUN_SECONDS` | `300` | The longest single run a visitor may request. |

## Keeping it awake

A Space sleeps after 48 hours with no visitors and wakes on the next request,
which takes a minute or so. Before a judging window, open it yourself once to
make sure it is warm. Under **Settings** you can also disable sleeping.

## When it does not work

**The build fails while installing.** Read the Logs tab; the failing package is
named. Every dependency here resolves to a wheel on `linux/amd64`, so this
should not happen.

**It builds, then the page is blank.** The port is wrong. `app_port` in
`README.md` must equal `QROUTE_PORT` in the image. Both are 8000.

**"Space is running" but the map has no roads.** Step 5 was skipped, so the
graphs are not in the image. Either redo it, or run
`qroute osm fetch` from the Space's terminal, accepting that it is lost on the
next restart.

**Push rejected for a large file.** Step 7's LFS tracking was missed. Check with
`git lfs ls-files`.

**The first solve is slower than later ones.** Numba keys its compiled cache to
the CPU it built on, and the Space may run on a different one. It recompiles
once. Expected, never an error.
