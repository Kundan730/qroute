# qroute — a single image that serves the API and the built front end on one port.
#
# Three stages:
#   1. `frontend`     Node builds the React application into a directory of
#                     static files. Node is not in the final image.
#   2. `python-build`  A virtualenv at /opt/venv with qroute and its
#                     dependencies installed. Build tooling is not in the final
#                     image; only the finished virtualenv is copied out.
#   3. `runtime`      python:slim + /opt/venv + the data the service reads.
#
# BASE IMAGE: why Debian slim and not Alpine
# ------------------------------------------
# numba, OR-Tools, NumPy, SciPy and pyproj are all distributed as *manylinux*
# wheels, which are linked against glibc. Alpine uses musl, and there is no
# musllinux wheel for numba at any version — checked against PyPI, not assumed:
#
#   pip download --only-binary=:all: --platform musllinux_1_2_x86_64 \
#       --python-version 3.13 --implementation cp --no-deps "numba>=0.60"
#   ERROR: Could not find a version that satisfies the requirement numba>=0.60
#
# On Alpine that turns into a source build of numba and LLVM, which is neither
# quick nor reliable. Against glibc targets the *whole* dependency closure — not
# a hand-picked list of the scary ones — resolves to wheels. Checked by asking
# pip to resolve exactly what this stage installs, for the target platform:
#
#   pip install --dry-run --only-binary=:all: --python-version 3.13 \
#       --implementation cp --abi cp313 --target /tmp/t --report /tmp/r.json \
#       --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
#       --platform manylinux_2_28_x86_64 --platform manylinux_2_34_x86_64 \
#       ".[baselines]"
#
# 59 packages resolve on linux/amd64 cp313 and 59 on linux/arm64 cp313, every
# one of them a wheel, qroute itself being the only sdist (it is this source
# tree). That closure includes the compiled transitive dependencies that a list
# of the obvious ones would miss — uvloop, httptools, watchfiles, pydantic-core,
# pyogrio, pillow, protobuf — as well as numba 0.67.0, OR-Tools 9.15.6755,
# NumPy 2.5.2, SciPy 1.18.1, llvmlite 0.49.0, pyproj 3.7.2, Shapely 2.1.2,
# matplotlib 3.11.1, pandas 3.0.5 and PyVRP 0.14.0. pandas has an aarch64 wheel
# too (manylinux_2_24/2_28), so arm64 is not a reduced build.
#
# The strictest wheel in each closure requires manylinux_2_28, i.e. glibc >=
# 2.28: pyproj on x86-64 and pyogrio on aarch64. Debian bookworm ships glibc
# 2.36, so the tag is pinned to bookworm rather than left floating: a future
# default of `python:3.13-slim` moving to a distro with an older libc would
# break the install in a way that is hard to read from the error message.
#
# Python 3.13 matches the version the test suite and the benchmark were run
# under. The same resolution repeated with --python-version 3.12 --abi cp312
# also gives 59 wheels and no sdist, so a downgrade is available if it is ever
# needed; it has not been run, only resolved.

# ---------------------------------------------------------------------------
# Stage 1 — build the single-page application
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS frontend

WORKDIR /build

# The lockfile is copied on its own first so that the dependency install is a
# separate layer from the source: editing a component re-runs `npm run build`
# but not `npm ci`. `npm ci` rather than `npm install` because it installs
# exactly what package-lock.json records and fails if the two disagree, which
# is the property that makes the image reproducible.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# `npm run build` is `tsc -b && vite build`, so a TypeScript error fails the
# image build rather than shipping a front end that does not type-check.
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — install the Python package into a self-contained virtualenv
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS python-build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# A virtualenv rather than the system site-packages, because a virtualenv is a
# single self-contained directory that the runtime stage can copy in one
# instruction. Nothing pip pulled in to *build* the wheel comes with it.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src

# README.md is required: pyproject.toml declares `readme = "README.md"`, and
# the build fails without it.
COPY pyproject.toml README.md ./
COPY qroute/ ./qroute/

# Every dependency resolves to a wheel on this platform (see the note at the
# top), so no compiler or -dev package is needed and none is installed. Without
# --no-cache-dir pip would leave several hundred megabytes of downloaded wheels
# in this stage; it is discarded anyway, but the cache also slows the build.
#
# The `baselines` extra is PyVRP, the state-of-the-art reference solver that
# appears in every published results table. It is optional in pyproject.toml and
# every call into it is guarded, so the image would work without it — but a
# deployed service that cannot run the solver its own benchmark compares against
# invites the reader to wonder what else is missing. Its wheel was checked the
# same way as the rest: pyvrp 0.14.0 has manylinux_2_28 wheels for cp313 on both
# x86_64 and aarch64.
RUN pip install --no-cache-dir ".[baselines]"


# ---------------------------------------------------------------------------
# Stage 3 — the image that actually runs
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# org.opencontainers.image.source is deliberately absent: it must be the URL of
# the repository this was built from, and a placeholder there is worse than
# nothing, because tooling treats the label as authoritative provenance.
LABEL org.opencontainers.image.title="qroute" \
      org.opencontainers.image.description="Quantum-inspired vehicle route optimisation: API and web UI on one port." \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT"

# A fixed uid/gid rather than whatever the base image next allocates, so that a
# bind-mounted host directory has predictable ownership and an operator can
# `chown 10001:10001` it without first inspecting the image.
RUN groupadd --system --gid 10001 qroute \
 && useradd --system --uid 10001 --gid 10001 --no-log-init \
            --create-home --home-dir /home/qroute --shell /usr/sbin/nologin qroute

# One ENV per concern, each with its reason above it, rather than one instruction
# with comments interleaved between its continuation lines. Comments inside a
# continued instruction are legal but are an easy thing for a reader (or an
# older parser) to get wrong, and an ENV layer carries no bytes, so there is
# nothing to save by combining them.
ENV PATH="/opt/venv/bin:$PATH"

# Unbuffered, so `docker logs` shows a line when it is written rather than when
# the pipe buffer happens to fill.
ENV PYTHONUNBUFFERED=1

# site-packages is owned by root and the process runs as qroute, so any attempt
# to write a .pyc would fail. pip already byte-compiled everything during the
# install, so there is nothing to gain by trying.
ENV PYTHONDONTWRITEBYTECODE=1

# qroute.config anchors its default paths on the installed package, which here
# is inside /opt/venv/lib/... and has no data beside it. QROUTE_DATA is
# therefore not optional in this image. The other two would be derived from it
# (results and frontend default to <data>/../results/runs and
# <data>/../frontend/dist) but are set explicitly, so that `docker inspect`
# answers "where does it read from?" without anyone having to know the rule.
ENV QROUTE_DATA=/app/data
ENV QROUTE_RESULTS=/app/results/runs
ENV QROUTE_FRONTEND=/app/frontend/dist

# numba compiles its kernels on first use and caches them next to the module
# source by default, which here is under root-owned site-packages, so every
# process would silently recompile. Redirecting the cache to a writable path is
# what lets the warm-up below bake the compiled kernels into the image.
ENV NUMBA_CACHE_DIR=/app/.cache/numba

# matplotlib requires a writable configuration directory and warns on every
# import when it cannot find one.
ENV MPLCONFIGDIR=/app/.cache/matplotlib

WORKDIR /app

COPY --from=python-build --chown=root:root /opt/venv /opt/venv

# The committed benchmark instances: 277 files, 1.4 MB, and the reason the
# service has something to solve the moment it starts.
COPY --chown=qroute:qroute data/benchmarks/ ./data/benchmarks/
# data/osm here is only the manifests — networks.json (the recipe `qroute osm
# fetch` regenerates the graphs from) and index.json. The .graphml files
# themselves are excluded by .dockerignore; see docs/deployment.md for why they
# are supplied through a volume instead of being baked in.
COPY --chown=qroute:qroute data/osm/ ./data/osm/
COPY --chown=qroute:qroute configs/ ./configs/
# The definitive benchmark, so that /api/benchmarks has real numbers to serve
# and the image is a complete, self-contained account of the submission.
COPY --chown=qroute:qroute results/runs/ ./results/runs/
COPY --from=frontend --chown=qroute:qroute /build/dist/ ./frontend/dist/

# The per-run log is committed gzipped (4.9 MB) because uncompressed it is
# 43 MB. /api/benchmarks/{name} reads the uncompressed file, so expand it here
# once at build time rather than making every deployment do it. `python -m
# gzip` is used in preference to the gzip binary because python is the one tool
# this image is guaranteed to have.
RUN python -m gzip -d results/runs/main/rows.jsonl.gz \
 && mkdir -p /app/.cache/numba /app/.cache/matplotlib \
 && chown -R qroute:qroute /app

USER qroute

# Two jobs in one instruction. It is a smoke test: if the package did not
# install correctly, or a compiled kernel will not build on this platform, the
# image build fails here rather than a judge discovering it at the demo. And it
# is a warm-up: solving a real instance compiles every numba kernel on the
# critical path and writes them to NUMBA_CACHE_DIR, so the shipped image starts
# with them already compiled instead of paying ~20 s on first request.
#
# The cache is keyed on the CPU numba was compiling for, so on a host whose CPU
# differs from the build machine's, numba simply recompiles. That is a slower
# first request, never an error.
RUN qroute version \
 && qroute solve A-n32-k5 --seconds 1

# Optionally bake the road graphs into the image.
#
# By default they are left out: they are 40 MB, they are regenerable from
# data/osm/networks.json, and on a host with a persistent volume it is better to
# fetch them once at runtime than to carry them in every image layer.
#
# On a platform with an ephemeral filesystem, though, that reasoning inverts.
# Hugging Face Spaces gives no persistent disk on the standard tiers, so a
# runtime fetch is repeated on every restart and the first visitor after each
# one waits several minutes with a map that has no roads on it. Building with
# `--build-arg BAKE_OSM=1` puts them in the image instead, which costs about
# 40 MB and three minutes of build time and makes the container start ready.
ARG BAKE_OSM=0
RUN if [ "$BAKE_OSM" = "1" ]; then \
      echo "baking road graphs into the image" && qroute osm fetch; \
    else \
      echo "road graphs left out; fetch them at runtime with 'qroute osm fetch'"; \
    fi

EXPOSE 8000

# /api/health answers as soon as the application has started; it reports the
# progress of the background warm-up and network preload in its body rather
# than withholding a response, which is why a plain 200 check is meaningful
# here. start-period covers the interpreter start and the fork-server prime.
#
# Written in Python rather than curl because curl is not in python:slim and
# adding it would mean an apt layer for one line of shell. `status` is checked
# as well as the HTTP code so that a proxy returning 200 for its own error page
# cannot pass for a healthy service. The probe is deliberately one long line:
# a `\` continuation inside a JSON array is parsed by Docker before the JSON is,
# and the joined result is not reliably the string that was written.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD ["python", "-c", "import json,urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8); sys.exit(0 if r.status == 200 and json.load(r).get('status') == 'ok' else 1)"]

# 0.0.0.0 rather than the 127.0.0.1 default: inside a container, binding the
# loopback interface makes the service unreachable from the published port.
# The container boundary, not the bind address, is what limits exposure here.
#
# No flags: `qroute serve` reads QROUTE_HOST and QROUTE_PORT, set just above.
# A host that dictates a port - Hugging Face Spaces, Cloud Run, Heroku - can
# then override the variable instead of needing this file edited.
ENV QROUTE_HOST=0.0.0.0
ENV QROUTE_PORT=8000
CMD ["qroute", "serve"]
