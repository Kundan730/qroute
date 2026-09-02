"""Loading benchmark instances from disk.

Two families are supported:

* **CVRPLIB / TSPLIB** ``.vrp`` files (sets A, B, P, X) - capacitated VRP.
* **Solomon** ``.txt`` files - VRP with time windows.

Both come with ``.sol`` files holding the best-known solution, which the
benchmark reports use as the reference point for gap computation.

A note on rounding. The two families use different, incompatible conventions,
and every published best-known value depends on getting them right:

* CVRPLIB ``EUC_2D``: Euclidean distance **rounded to the nearest integer**
  (TSPLIB ``nint``). Verified exactly against the reference solutions of
  A-n32-k5, A-n80-k10, B-n31-k5, P-n16-k8, X-n101-k25, X-n502-k39 and
  X-n1001-k43.
* Solomon VRPTW: Euclidean distance **truncated to one decimal place**.
  Verified exactly against C101, C201, R101, R112, R201 and RC101.

Note that ``vrplib`` returns an unrounded ``edge_weight`` matrix for coordinate
instances, which overstates the cost of the reference solution by roughly 0.5%.
We therefore always rebuild the matrix from the coordinates ourselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import vrplib

from qroute.problems.instance import Instance, ObjectiveWeights

DATA_ROOT = Path(os.environ.get("QROUTE_DATA", "data"))
CVRP_DIR = DATA_ROOT / "benchmarks" / "cvrplib"
VRPTW_DIR = DATA_ROOT / "benchmarks" / "solomon"


def euclidean_matrix(coords: np.ndarray, rounding: str = "round") -> np.ndarray:
    """Distance matrix for 2-D coordinates.

    Parameters
    ----------
    rounding:
        ``'nint'``  - round to nearest integer (TSPLIB ``EUC_2D``, all CVRPLIB sets).
        ``'trunc1'`` - truncate to one decimal (Solomon VRPTW).
        ``'none'``  - keep full floating point precision (road-network instances).
    """
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    if rounding in ("nint", "round"):
        # TSPLIB nint(): round half away from zero, matching the C reference code.
        d = np.floor(d + 0.5)
    elif rounding == "trunc1":
        # Solomon's convention: chop, do not round, at the first decimal.
        d = np.floor(d * 10.0 + 1e-9) / 10.0
    elif rounding != "none":
        raise ValueError(f"unknown rounding mode {rounding!r}")
    return np.ascontiguousarray(d, dtype=np.float64)


def _vehicles_from_name(name: str) -> Optional[int]:
    """CVRPLIB names encode the reference route count as ``-k<N>``."""
    tail = name.rsplit("-k", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    return None


def load_cvrplib(path: str | Path, fixed_fleet: bool = False) -> Instance:
    """Load a CVRPLIB ``.vrp`` file.

    ``fixed_fleet`` caps the fleet at the ``k`` value in the file name. CVRPLIB
    convention for sets A/B/P is that this is the *minimum* number of vehicles in
    the best-known solution, so we leave the fleet free by default and report the
    route count separately.
    """
    path = Path(path)
    raw = vrplib.read_instance(str(path))
    name = raw.get("name") or path.stem

    ewt = str(raw.get("edge_weight_type", "EUC_2D"))
    if "node_coord" in raw and np.asarray(raw["node_coord"]).size:
        coords = np.asarray(raw["node_coord"], dtype=np.float64)
        # Rebuild rather than trust vrplib's unrounded edge_weight (see module docstring).
        dist = euclidean_matrix(coords, "nint" if "EUC" in ewt else "none")
    else:
        dist = np.asarray(raw["edge_weight"], dtype=np.float64)
        coords = np.zeros((dist.shape[0], 2), dtype=np.float64)

    demand = np.asarray(raw["demand"], dtype=np.float64)
    depot = int(np.asarray(raw.get("depot", [0])).ravel()[0])
    if depot != 0:  # normalise so that the depot is index 0
        order = [depot] + [i for i in range(len(demand)) if i != depot]
        dist = dist[np.ix_(order, order)]
        demand = demand[order]
        coords = coords[order]

    inst = Instance(
        name=name,
        distance=dist,
        demand=demand,
        capacity=float(raw["capacity"]),
        n_vehicles=_vehicles_from_name(path.stem) if fixed_fleet else None,
        coords=coords,
        weights=ObjectiveWeights(distance=1.0),
        meta={"source": str(path), "family": "cvrplib", "edge_weight_type": str(ewt),
              "reference_k": _vehicles_from_name(path.stem)},
    )
    sol_path = path.with_suffix(".sol")
    if sol_path.exists():
        try:
            sol = vrplib.read_solution(str(sol_path))
            inst.meta["bks"] = float(sol["cost"])
            inst.meta["bks_routes"] = len(sol.get("routes", []))
        except Exception:  # a malformed reference file must not break loading
            pass
    return inst


def load_solomon(path: str | Path, n_customers: int | None = None) -> Instance:
    """Load a Solomon VRPTW ``.txt`` file.

    Solomon distances are Euclidean truncated to one decimal, and travel time
    equals distance. ``n_customers`` truncates the instance (Solomon's 25- and
    50-customer subsets are simply the first rows of the 100-customer file).
    """
    path = Path(path)
    raw = vrplib.read_instance(str(path), instance_format="solomon")
    coords = np.asarray(raw["node_coord"], dtype=np.float64)
    demand = np.asarray(raw["demand"], dtype=np.float64)
    tw = np.asarray(raw["time_window"], dtype=np.float64)
    service = np.asarray(raw["service_time"], dtype=np.float64)

    if n_customers is not None:
        k = n_customers + 1
        coords, demand, tw, service = coords[:k], demand[:k], tw[:k], service[:k]

    dist = euclidean_matrix(coords, "trunc1")
    name = raw.get("name") or path.stem
    if n_customers is not None:
        name = f"{name}-{n_customers}"

    inst = Instance(
        name=name,
        distance=dist,
        duration=dist,  # Solomon: travel time == Euclidean distance
        demand=demand,
        capacity=float(raw["capacity"]),
        n_vehicles=int(raw["vehicles"]) if raw.get("vehicles") else None,
        time_windows=tw,
        service_time=service,
        coords=coords,
        weights=ObjectiveWeights(distance=1.0),
        meta={"source": str(path), "family": "solomon"},
    )
    sol_path = path.with_suffix(".sol")
    if sol_path.exists() and n_customers is None:
        try:
            sol = vrplib.read_solution(str(sol_path))
            inst.meta["bks"] = float(sol["cost"])
            inst.meta["bks_routes"] = len(sol.get("routes", []))
        except Exception:
            pass
    return inst


def load(name_or_path: str | Path, **kwargs) -> Instance:
    """Load an instance by name (``'A-n32-k5'``, ``'C101'``) or by file path."""
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        if p.suffix == ".txt":
            return load_solomon(p, **kwargs)
        return load_cvrplib(p, **kwargs)

    stem = str(name_or_path)
    cand = CVRP_DIR / f"{stem}.vrp"
    if cand.exists():
        return load_cvrplib(cand, **kwargs)
    cand = VRPTW_DIR / f"{stem}.txt"
    if cand.exists():
        return load_solomon(cand, **kwargs)
    raise FileNotFoundError(
        f"instance {stem!r} not found under {CVRP_DIR} or {VRPTW_DIR}. "
        "Run `qroute data fetch` to download the benchmark sets."
    )


def list_instances() -> dict[str, list[str]]:
    """Names of every benchmark instance available locally."""
    cvrp = sorted(p.stem for p in CVRP_DIR.glob("*.vrp")) if CVRP_DIR.exists() else []
    tw = sorted(p.stem for p in VRPTW_DIR.glob("*.txt")) if VRPTW_DIR.exists() else []
    return {"cvrp": cvrp, "vrptw": tw}


def read_reference_solution(inst: Instance) -> Optional[list[list[int]]]:
    """Routes of the best-known solution, if the ``.sol`` file is present."""
    src = inst.meta.get("source")
    if not src:
        return None
    sol_path = Path(src).with_suffix(".sol")
    if not sol_path.exists():
        return None
    try:
        sol = vrplib.read_solution(str(sol_path))
    except Exception:
        return None
    return [[int(c) for c in r] for r in sol.get("routes", [])]
