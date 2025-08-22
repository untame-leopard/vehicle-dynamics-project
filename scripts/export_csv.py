import pandas as pd
from vdyn.models.longitudinal_1d import Vehicle, accel_brake_run

base = dict(CdA=0.90)
no_df = Vehicle(ClA=0.0, dCdA_per_ClA=0.1, **base)
hi_df  = Vehicle(ClA=2.5, dCdA_per_ClA=0.1, **base)

for name, car in [("no_df", no_df), ("hi_df", hi_df)]:
    T,V,S,A = accel_brake_run(car, dt=0.01)
    pd.DataFrame({"t":T, "v":V, "s":S, "a":A}).to_csv(f"docs/{name}.csv", index=False)
