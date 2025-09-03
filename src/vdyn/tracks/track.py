from __future__ import annotations
import numpy as np

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

    def sample_kappa(self, s_query: np.ndarray) -> np.ndarray:
        """
    Samples the track's curvature (kappa) at a given arc-length.

    This function uses linear interpolation on the pre-computed track data
    to find the curvature at any point along the track's centerline.

    Args:
        s_query (np.ndarray): The arc-length(s) to query. This can be a single
                              value or a NumPy array.

    Returns:
        np.ndarray: The curvature in inverse meters (1/m). For a closed-loop
                    track, the result seamlessly wraps around to the start.
    """
        s = np.mod(s_query, self.L)
        return np.interp(s, self.s, self.kappa)

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
        s = np.mod(s_query, self.L) #Modulo for closed-loop
        return np.interp(s, self.s, self.psi)

    def sample_xy(self, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples the track's global X and Y coordinates at a given arc-length.

        Args:
            s_query (np.ndarray): The arc-length(s) to query.
                                 Can be a single value or a NumPy array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the X and Y coordinates.
        """
        s = np.mod(s_query, self.L)
         # Use linear interpolation to find the X and Y coordinates at the queried s
        x = np.interp(s, self.s, self.x) 
        y = np.interp(s, self.s, self.y)
        return x, y
