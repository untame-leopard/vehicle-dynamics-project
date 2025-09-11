from __future__ import annotations
import re
import numpy as np
import pandas as pd
from vdyn.telemetry.aim import load_aim_csv

# ---------- header parsing ----------

def _parse_mmss(token: str) -> float:
    s = token.strip().strip('"').strip()
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)

def read_markers_and_segments_robust(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (markers_s, seg_times_s) from an AiM CSV header.
    Robust to commas/semicolons/quotes/spacing.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        header = [next(f) for _ in range(60)]
    norm = "\n".join(header).replace(";", ",")

    def grab_after(key: str) -> str | None:
        m = re.search(rf'^\s*"?{re.escape(key)}"?\s*,(.*)$', norm, flags=re.MULTILINE)
        return m.group(1) if m else None

    markers_line = grab_after("Beacon Markers")
    segtimes_line = grab_after("Segment Times")
    if markers_line is None or segtimes_line is None:
        raise ValueError("Could not find 'Beacon Markers' / 'Segment Times' in CSV header.")

    markers = [float(x.strip().strip('"')) for x in markers_line.split(",") if x.strip()]
    seg_times = [_parse_mmss(x) for x in segtimes_line.split(",") if x.strip()]

    return np.asarray(markers, float), np.asarray(seg_times, float)

# ---------- lap extraction ----------

def extract_fastest_lap(
    df: pd.DataFrame,
    markers_s: np.ndarray,
    seg_times_s: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """
    Use AiM header 'Beacon Markers' and 'Segment Times' to cut the fastest lap.
    Robust to NaNs, non-monotonic header edges, and boundary rounding.
    Returns (lap_df, info).
    """
    # 1) pick fastest *real* lap: ignore micro segments (<10 s) commonly found at file start
    valid = seg_times_s >= 10.0
    if not np.any(valid):
        raise ValueError("No valid laps in header (all segments < 10 s).")
    k_best = int(np.flatnonzero(valid)[np.argmin(seg_times_s[valid])])

    t_start = float(markers_s[k_best])
    t_end   = float(t_start + seg_times_s[k_best])
    lap_time = float(seg_times_s[k_best])

    # 2) clean time vector
    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
    ok = np.isfinite(t)
    if not ok.any():
        raise ValueError("No finite 'time_s' values in dataframe.")
    t_clean = t[ok]
    df_clean = df.loc[ok].reset_index(drop=True)

    # 3) find indices with tolerance and clamping
    eps = 1e-6
    i0 = int(np.searchsorted(t_clean, t_start - eps, side="left"))
    i1 = int(np.searchsorted(t_clean, t_end   + eps, side="right"))

    i0 = max(0, min(i0, len(t_clean) - 1))
    i1 = max(i0 + 1, min(i1, len(t_clean)))  # ensure at least two samples

    lap = df_clean.iloc[i0:i1].copy()
    if lap.empty:
        # extreme fallback: choose nearest indices by absolute difference
        i0 = int(np.argmin(np.abs(t_clean - t_start)))
        i1 = int(np.argmin(np.abs(t_clean - t_end)))
        if i1 <= i0:
            i1 = min(i0 + 1, len(t_clean) - 1)
        lap = df_clean.iloc[i0:i1+1].copy()
        if lap.empty:
            raise RuntimeError("Lap slice empty after fallbacks (check header markers).")

    # 4) distance rebase
    s = pd.to_numeric(lap["distance_m"], errors="coerce")
    if s.notna().sum() < 2:
        raise RuntimeError("distance_m has <2 finite samples in selected lap window.")
    s = s.to_numpy()
    lap["s_rel"] = s - s[0]

    info = {
        "t_start": t_start,
        "t_end": t_end,
        "lap_time": lap_time,
        "lap_length_m": float(lap["s_rel"].iloc[-1]),
        "k_best": k_best,
    }
    return lap, info

# ---------- small helpers ----------

def sector_times_equal_thirds(lap_df: pd.DataFrame, n: int = 3) -> dict[tuple[float, float], float]:
    """Compute n equal sectors by distance within the lap."""
    s = pd.to_numeric(lap_df["s_rel"], errors="coerce").to_numpy()
    t = pd.to_numeric(lap_df["time_s"], errors="coerce").to_numpy()
    edges = np.linspace(0.0, s[-1], n + 1)
    out: dict[tuple[float, float], float] = {}
    for a, b in zip(edges[:-1], edges[1:]):
        ia = int(np.searchsorted(s, a, side="left"))
        ib = int(np.searchsorted(s, b, side="right"))
        ia = max(0, min(ia, len(s) - 2))
        ib = max(ia + 1, min(ib, len(s) - 1))
        out[(float(a), float(b))] = float(t[ib] - t[ia])
    return out, edges

def get_fastest_lap_from_csv(csv_path: str):
    """
    Convenience wrapper: read AiM CSV, parse markers, and return (lap_df, info)
    so notebooks can do it in one call.
    """
    df = load_aim_csv(csv_path)
    markers_s, seg_times_s = read_markers_and_segments_robust(csv_path)
    lap, info = extract_fastest_lap(df, markers_s, seg_times_s)
    return lap, info
