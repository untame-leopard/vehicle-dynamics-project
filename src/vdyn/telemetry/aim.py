import csv
from typing import List
import pandas as pd
import numpy as np

DELIMS = [",", ";", "\t", "|"]

def _sniff_delim(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        # fallback: pick the delimiter with most occurrences on the longest lines
        lines = [l for l in sample.splitlines() if l.strip()]
        best = (0, ",")
        for d in DELIMS:
            score = max((ln.count(d) for ln in lines), default=0)
            if score > best[0]:
                best = (score, d)
        return best[1]

def load_aim_csv(path: str) -> pd.DataFrame:
    """
    Robust AiM/RaceStudio CSV loader:
      - Detects delimiter (comma/semicolon/tab/pipe).
      - Finds the TRUE header row (starts with 'Time' AND contains 'GPS Speed' with >=6 cols).
      - Skips the following units row.
      - Returns normalized convenience columns: time_s, speed_mps, distance_m, rpm (if present).
    """
    # detect delimiter from a small sample
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    delim = _sniff_delim(sample)

    header: List[str] = []
    rows: List[List[str]] = []

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.reader(f, delimiter=delim)
        header_found = False

        for line in r:
            if not line:
                continue
            cells = [c.strip().strip('"') for c in line]
            # choose the *long* telemetry header, not the metadata "Time,10:38 AM"
            if (not header_found
                and len(cells) >= 6
                and cells[0] == "Time"
                and ("GPS Speed" in cells or "GPS speed" in cells)):
                header = cells
                header_found = True
                # skip the units row that immediately follows
                _ = next(r, None)
                break

        if not header_found:
            raise ValueError("Could not find telemetry header row (look for 'Time,...GPS Speed...').")

        # collect data rows
        for line in r:
            if not line:
                continue
            cells = [c.strip().strip('"') for c in line]
            # pad/trim to header length to avoid shape mismatches
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[:len(header)]
            rows.append(cells)

    if not rows:
        raise ValueError("No data rows found after header.")

    df = pd.DataFrame(rows, columns=header)

    # convert obvious numeric columns (pandas will leave strings alone)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # normalized convenience fields
    if "Time" in df.columns:
        df["time_s"] = pd.to_numeric(df["Time"], errors="coerce")

    if "GPS Speed" in df.columns:
        s = pd.to_numeric(df["GPS Speed"], errors="coerce")
        # AiM GPS Speed is km/h -> convert to m/s
        df["speed_mps"] = s / 3.6

    if "Distance on GPS Speed" in df.columns:
        d = pd.to_numeric(df["Distance on GPS Speed"], errors="coerce").to_numpy()
        if np.isfinite(d).any():
            start = d[np.isfinite(d)][0]
            d = np.nan_to_num(d, nan=start)
            df["distance_m"] = np.maximum.accumulate(d)

    if "RPM" in df.columns:
        df["rpm"] = pd.to_numeric(df["RPM"], errors="coerce")

    return df
