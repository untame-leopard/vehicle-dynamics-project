from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

class CenterlineTrack:
    """
    Track defined by arc-length s and curvature kappa(s).
    Provides interpolation of kappa(s), heading psi(s) and XY coordinates.
    """
    
    def __init__(self, s: np.ndarray, kappa: np.ndarray):
        assert len(s) == len(kappa) and len(s) >= 2
        # ensure strictly increasing s
        order = np.argsort(s)
        self.s = np.asarray(s[order], dtype=float)
        self.kappa = np.asarray(kappa[order], dtype=float)
        self.L = float(self.s[-1])
        # precompute heading by integrating kappa over s (cumulative trapezoid)
        dk = 0.5 * (self.kappa[1:] + self.kappa[:-1]) * np.diff(self.s)
        psi = np.concatenate([[0.0], np.cumsum(dk)])  # psi(0)=0
        self.psi = psi
        # precompute XY by integrating v=[cos psi, sin psi] w.r.t. s
        x = [0.0]
        y = [0.0]
        for i in range(len(self.s) - 1):
            ds = self.s[i+1] - self.s[i]
            cx = 0.5 * (np.cos(psi[i]) + np.cos(psi[i+1]))
            cy = 0.5 * (np.sin(psi[i]) + np.sin(psi[i+1]))
            x.append(x[-1] + cx * ds)
            y.append(y[-1] + cy * ds)
        self.x = np.array(x)
        self.y = np.array(y)

        self.L = float(self.s[-1])

        # Build all periodic splines once
        self._build_periodic_splines()

        self._s_samples = np.linspace(0.0, self.L, 5000, endpoint=False)
        Xs = self._x_spline(self._s_samples)
        Ys = self._y_spline(self._s_samples)
        self._xy_samples = np.column_stack([Xs, Ys])
        self._kdtree = cKDTree(self._xy_samples)


    def _build_periodic_splines(self):
        # Curvature κ(s)
        self._kappa_spline = CubicSpline(self.s, self.kappa, bc_type='periodic')
        # Centerline x(s), y(s) as periodic too (so you can tile laps cleanly)
        self._x_spline = CubicSpline(self.s, self.x)
        self._y_spline = CubicSpline(self.s, self.y)

    def sample_kappa(self, s_query: np.ndarray) -> np.ndarray:
        s = np.array(s_query, dtype=float)
        out = self._kappa_spline(np.mod(s, self.L))
        return float(out) if np.ndim(out) == 0 else out

    @classmethod
    def from_csv(cls, path: str) -> "CenterlineTrack":
        """
    Creates a new CenterlineTrack instance by loading data from a CSV file.

    Args:
        path (str): The file path to the CSV containing 's' and 'kappa' data.
                    The file should have two columns with s [m] and kappa [1/m],
                    and can include commented lines starting with '#'.

    Returns:
        CenterlineTrack: A new instance of the class initialised with the loaded data.
    """
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(",")
                rows.append((float(a), float(b)))
        s, k = (np.array([r[i] for r in rows], dtype=float) for i in (0, 1))
        return cls(s, k)

    def sample_psi(self, s_query: np.ndarray) -> np.ndarray:
        """
        Samples the track's heading (psi) at a given arc-length.

        Args:
            s_query (np.ndarray): The arc-length(s) to query.
                Can be a single value or a NumPy array.

        Returns:
            np.ndarray: The heading angle in radians. For a closed-loop track,
                        the result will loop back to the start of the track.
        """
        s = np.asarray(s_query, dtype=float)
        sm = np.mod(s, self.L)
        # differentiate x(s), y(s) splines for tangent heading
        dxds = CubicSpline(self.s, self.x).derivative()(sm) if not hasattr(self, "_x_spline") else self._x_spline.derivative()(sm)
        dyds = CubicSpline(self.s, self.y).derivative()(sm) if not hasattr(self, "_y_spline") else self._y_spline.derivative()(sm)
        psi = np.arctan2(dyds, dxds)
        return float(psi) if np.ndim(psi) == 0 else psi

    def sample_xy(self, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples the track's global X and Y coordinates at a given arc-length.

        Args:
            s_query (np.ndarray): The arc-length(s) to query.
                                 Can be a single value or a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the X and Y coordinates.
        """
        s = np.asarray(s_query, dtype=float)
        sm = np.mod(s, self.L)
        x = self._x_spline(sm)
        y = self._y_spline(sm)
        if np.ndim(x) == 0:
            return float(x), float(y) 
        return x, y
    
    def project_to_centerline(self, x, y):
        d, idx = self._kdtree.query([x, y], k=1)
        idx = int(idx)
        s_star = float(self._s_samples[idx])
        x_star = float(self._xy_samples[idx, 0])
        y_star = float(self._xy_samples[idx, 1])
        return s_star, x_star, y_star, idx, float(d)

