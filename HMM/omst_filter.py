"""omst_filter — reusable Python wrapper around the MATLAB OMST routine.

Apply Orthogonal-MST (OMST) global-cost-efficiency filtering to weighted
connectivity matrices via a MATLAB subprocess. Wraps
``threshold_omst_gce_wu_very_fast`` from the topological_filtering_networks
package.

Three sign strategies, applied independently to each input matrix:

  * ``A_pos``  — zero negatives, OMST on positive subgraph (output ≥ 0)
  * ``A_neg``  — zero positives, abs, OMST, negate output (output ≤ 0)
  * ``B``      — abs, OMST, restore signs from the original (signed output)

Diagonal is zeroed before filtering and left at 0 in the output.

Typical use::

    from omst_filter import omst_filter_batch
    filtered, metrics = omst_filter_batch(matrices_3d,
                                          strategies=("A_pos","A_neg","B"))

A 2D input is also accepted and is internally treated as a single-matrix batch.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Defaults — change these once if your system layout differs.
# ---------------------------------------------------------------------------
DEFAULT_MATLAB = "/home/aazarg/bin/matlab"
DEFAULT_OMST_PKG = (
    "/home/aazarg/data/topological_filtering_networks/"
    "threshold_schemes/threshold_schemes"
)
DEFAULT_HELPER_M = Path(__file__).resolve().parent / "run_omst_single_matrix.m"

ALL_STRATEGIES = ("A_pos", "A_neg", "B")
METRIC_FIELDS = ("n_msts", "mdeg", "gce", "costmax", "E")


# ---------------------------------------------------------------------------
# Core call
# ---------------------------------------------------------------------------
def omst_filter_batch(
    matrices: np.ndarray,
    strategies: Sequence[str] = ALL_STRATEGIES,
    *,
    matlab_exe: str = DEFAULT_MATLAB,
    omst_pkg_dir: str = DEFAULT_OMST_PKG,
    helper_m_path: os.PathLike = DEFAULT_HELPER_M,
    work_dir: os.PathLike | None = None,
    timeout_sec: int = 1800,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    """Filter a stack of matrices through MATLAB OMST.

    Parameters
    ----------
    matrices
        Either a (K, N, N) stack or a single 2D (N, N) matrix. Must be
        symmetric (no symmetry check is performed; the OMST routine itself
        treats the upper triangle).
    strategies
        Iterable subset of {"A_pos", "A_neg", "B"}.
    matlab_exe
        Path to the MATLAB executable.
    omst_pkg_dir
        Path to the folder containing ``threshold_omst_gce_wu_very_fast.m``.
    helper_m_path
        Path to ``run_omst_single_matrix.m`` (the helper function this
        wrapper calls).
    work_dir
        Where to put the temporary input/output .mat files. Defaults to a
        fresh tempdir, which is cleaned up on success.
    timeout_sec
        Subprocess timeout for the MATLAB call.
    verbose
        Echo MATLAB stdout to the console.

    Returns
    -------
    filtered : dict
        ``{strategy: ndarray of shape (K, N, N) float64}``. If the input was
        2D, each value is also 2D.
    metrics : dict
        ``{strategy: list of K dicts}``. Each dict has keys
        ``n_msts, mdeg, gce, costmax, E`` plus ``index`` (0..K-1).
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    squeeze_output = False
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
        squeeze_output = True
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            f"matrices must have shape (K, N, N); got {matrices.shape}"
        )

    strategies = tuple(strategies)
    bad = set(strategies) - set(ALL_STRATEGIES)
    if bad:
        raise ValueError(f"Unknown strategies: {sorted(bad)}")

    helper_m_path = Path(helper_m_path).resolve()
    if not helper_m_path.exists():
        raise FileNotFoundError(f"Helper not found: {helper_m_path}")

    # Run MATLAB in a tempdir to keep intermediate .mat files isolated.
    cleanup = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="omst_")) if cleanup else Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    in_mat = work / "omst_input.mat"
    out_mat = work / "omst_output.mat"
    sio.savemat(
        in_mat,
        {"matrices": matrices, "strategies": np.array(strategies, dtype=object)},
        do_compression=False,
    )

    # Build the MATLAB command. We run helper_m_path's containing folder on
    # the path so the function is callable by name.
    helper_dir = helper_m_path.parent
    helper_name = helper_m_path.stem
    matlab_cmd = (
        f"addpath('{helper_dir}'); "
        f"{helper_name}('{in_mat}', '{out_mat}', '{omst_pkg_dir}'); "
        "exit;"
    )

    if verbose:
        print(f"[omst] {matrices.shape[0]} matrices x {len(strategies)} strategies")

    proc = subprocess.run(
        [matlab_exe, "-nodisplay", "-nosplash", "-nodesktop", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "MATLAB OMST failed.\n"
            f"  stdout:\n{proc.stdout}\n  stderr:\n{proc.stderr}"
        )
    if verbose and proc.stdout.strip():
        print(proc.stdout.rstrip())

    if not out_mat.exists():
        raise RuntimeError(f"MATLAB did not write expected output: {out_mat}")

    raw = sio.loadmat(out_mat, squeeze_me=False, struct_as_record=False)

    filtered: dict[str, np.ndarray] = {}
    metrics: dict[str, list[dict]] = {}
    for strat in strategies:
        mat_key = f"mat_{strat}"
        met_key = f"metrics_{strat}"
        if mat_key not in raw or met_key not in raw:
            raise KeyError(
                f"Missing OMST output for strategy {strat!r} "
                f"(expected '{mat_key}' and '{met_key}')."
            )
        arr = np.asarray(raw[mat_key], dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if squeeze_output:
            filtered[strat] = arr[0]
        else:
            filtered[strat] = arr

        metrics[strat] = _struct_array_to_dicts(raw[met_key])

    if cleanup:
        for p in (in_mat, out_mat):
            if p.exists():
                p.unlink()
        try:
            work.rmdir()
        except OSError:
            pass  # leave non-empty work dir if user passed one

    return filtered, metrics


def omst_filter_one(
    matrix: np.ndarray,
    strategies: Sequence[str] = ALL_STRATEGIES,
    **kwargs,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Convenience wrapper for a single 2D matrix.

    Returns ``(filtered, metrics)`` where each entry is keyed by strategy and
    the metrics value is a single dict (not a list).
    """
    filtered, metrics = omst_filter_batch(matrix, strategies=strategies, **kwargs)
    metrics_one = {s: m[0] for s, m in metrics.items()}
    return filtered, metrics_one


# ---------------------------------------------------------------------------
# Internal: parse MATLAB struct array (loaded via struct_as_record=False)
# ---------------------------------------------------------------------------
def _struct_array_to_dicts(arr) -> list[dict]:
    """Flatten a MATLAB (Kx1) struct array to a list of plain Python dicts."""
    arr = np.asarray(arr).ravel()  # (K,) of mat_struct objects
    out: list[dict] = []
    for k, item in enumerate(arr):
        d = {"index": k}
        for f in METRIC_FIELDS:
            v = getattr(item, f, np.nan)
            if isinstance(v, np.ndarray):
                v = float(v.flatten()[0]) if v.size else np.nan
            else:
                v = float(v) if v is not None else np.nan
            d[f] = v
        out.append(d)
    return out