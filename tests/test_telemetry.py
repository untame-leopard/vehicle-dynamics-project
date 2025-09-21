from vdyn.telemetry.aim import load_aim_csv

def test_load_basic():
    df = load_aim_csv("data/2.csv")  # your file
    assert "time_s" in df.columns and df["time_s"].notna().sum() > 10
    assert "speed_mps" in df.columns and df["speed_mps"].notna().sum() > 10
    assert "distance_m" in df.columns and df["distance_m"].notna().sum() > 10
