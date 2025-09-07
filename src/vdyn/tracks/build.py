import numpy as np
import pandas as pd
from typing import Tuple

R_EARTH = 6371000.0  # meters

def latlon_to_local_xy(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple equirectangular projection around the first sample as origin.
    Good enough for small areas like kart tracks.
    """
    lat = np.radians(lat_deg.astype(float))
    lon = np.radians(lon_deg.astype(float))
    lat0 = lat[0]
    lon0 = lon[0]
    x = R_EARTH * (lon - lon0) * np.cos(lat0)
    y = R_EARTH * (lat - lat0)
    return x, y

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    # reflect padding to avoid edge shrinkage
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    kern = np.ones(w) / w
    return np.convolve(xpad, kern, mode="valid")

def cumulative_s(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    ds = np.hypot(dx, dy)
    s = np.cumsum(ds)
    s[0] = 0.0
    return s

def resample_by_s(s: np.ndarray, arrs: Tuple[np.ndarray, ...], ds: float) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    s = s.astype(float)
    s_uniform = np.arange(0.0, float(s[-1]) + ds/2, ds)
    out = []
    for a in arrs:
        out.append(np.interp(s_uniform, s, a.astype(float)))
    return s_uniform, tuple(out)

def heading_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.arctan2(dy, dx)

def curvature_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_s = np.gradient(x)
    y_s = np.gradient(y)
    x_ss = np.gradient(x_s)
    y_ss = np.gradient(y_s)
    denom = (x_s**2 + y_s**2)**1.5
    # avoid div by zero
    denom = np.where(denom < 1e-6, 1e-6, denom)
    kappa = (x_s * y_ss - y_s * x_ss) / denom
    return kappa

def build_racing_line_from_aim(
    df: pd.DataFrame,
    lat_col: str = "GPS Latitude",
    lon_col: str = "GPS Longitude",
    smooth_window: int = 9,
    ds: float = 0.5,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: s_m, x_m, y_m, psi_rad, kappa_1pm
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Expected '{lat_col}' and '{lon_col}' in AiM CSV columns.")
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()

    # drop NaNs conservatively
    mask = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[mask]
    lon = lon[mask]

    x, y = latlon_to_local_xy(lat, lon)

    # light smoothing to tame GPS jitter 
    if smooth_window and smooth_window > 1:
        x = moving_average(x, smooth_window)
        y = moving_average(y, smooth_window)

    s = cumulative_s(x, y)
    s_u, (x_u, y_u) = resample_by_s(s, (x, y), ds=ds)

    psi = heading_from_xy(x_u, y_u)
    kappa = curvature_from_xy(x_u, y_u)

    out = pd.DataFrame({
        "s_m": s_u,
        "x_m": x_u,
        "y_m": y_u,
        "psi_rad": psi,
        "kappa_1pm": kappa,
    })
    return out
