import argparse
import pandas as pd
from vdyn.telemetry import load_aim_csv
from vdyn.tracks.build import build_racing_line_from_aim

def main():
    ap = argparse.ArgumentParser(description="Export kart track centerline CSV from AiM telemetry.")
    ap.add_argument("--in_csv", default="data/2.csv", help="AiM CSV path")
    ap.add_argument("--out_csv", default="data/tracks/kart_centerline.csv", help="Output centerline CSV")
    ap.add_argument("--lat_col", default="GPS Latitude")
    ap.add_argument("--lon_col", default="GPS Longitude")
    ap.add_argument("--ds", type=float, default=0.5, help="Resample spacing in meters")
    ap.add_argument("--smooth", type=int, default=9, help="Moving-average window (odd integer)")
    args = ap.parse_args()

    df = load_aim_csv(args.in_csv)
    cl = build_racing_line_from_aim(
        df,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        smooth_window=args.smooth,
        ds=args.ds,
    )
    cl.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}  ({len(cl)} points)")

if __name__ == "__main__":
    main()
