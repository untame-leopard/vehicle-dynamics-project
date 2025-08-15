import numpy as np


def accel_brake_test(
    m=800,        # mass [kg]
    power=300000, # constant engine power [W]
    CdA=0.9,      # drag area [m^2]
    rho=1.225,    # air density [kg/m^3]
    Crr=0.015,    # rolling resistance coeff
    mu_brake=1.5, # braking friction coeff
    v_target=200/3.6,  # target speed [m/s]
    dt=0.01       # timestep [s]
):
    """
    Simulate 0–v_target–0 run.

    Returns:
        t (array): time [s]
        v (array): velocity [m/s]
    """

    # Acceleration phase
    v = 0.0
    t = 0.0
    times = []
    speeds = []

    while v < v_target:
        F_drag = 0.5 * rho * CdA * v**2
        F_rr = Crr * m * 9.81
        F_engine = power / max(v, 1e-3)  # P = F*v
        a = (F_engine - F_drag - F_rr) / m
        v += a * dt
        t += dt
        times.append(t)
        speeds.append(v)

    # Braking phase
    while v > 0:
        F_drag = 0.5 * rho * CdA * v**2
        F_rr = Crr * m * 9.81
        F_brake = mu_brake * m * 9.81
        a = -(F_brake + F_drag + F_rr) / m
        v += a * dt
        v = max(v, 0)
        t += dt
        times.append(t)
        speeds.append(v)

    return np.array(times), np.array(speeds)
