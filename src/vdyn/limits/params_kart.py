from dataclasses import dataclass

@dataclass
class KartParams:
    m: float = 163.0           # kg (kart + driver)
    L: float = 1.05            # wheelbase [m]
    mu_lat: float = 1.8        # baseline lateral friction (dry slicks ballpark)
    mu_long_drive: float = 1.6 # longitudinal accel friction
    mu_long_brake: float = 1.7 # longitudinal brake friction
    ClA: float = 0.0           # downforce area
    CdA: float = 0.70          # drag area [m^2] (will adjust after telemetry fit)
    rho: float = 1.225         # air density
    Crr: float = 0.012         # rolling resistance
    Rw: float = 0.105          # wheel radius [m]
    ratio: float = 11.714        #Fixed overall gear ratio (single-gear kart)
    eta: float = 0.95          # Drivetrain efficiency  
    P_max: float = 24_000.0    # ~ 32 hp