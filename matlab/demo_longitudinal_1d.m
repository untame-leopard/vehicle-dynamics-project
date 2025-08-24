clear; clc;

% Match Python defaults
car.m=800; car.power=300e3; car.CdA=0.90; car.rho=1.225;
car.Crr=0.015; car.mu_drive=1.2; car.mu_brake=1.2;
car.ita_drive=0.90; car.v_target=200/3.6;
car.ClA=2.5; car.dCdA_per_ClA=0.1;   % L/D ≈ 10

dt = 0.01;

sim = simulate_longitudinal_1d(car, dt);
k   = compute_kpis(sim.T, sim.V, sim.S)

% Overlay with Python CSV
try
    py = readtable('../docs/hi_df.csv');   % change to 'no_df.csv' to compare no-DF case
    figure; hold on;
    plot(sim.T, sim.V*3.6, 'DisplayName','MATLAB');
    plot(py.t, py.v*3.6, '--', 'DisplayName','Python');
    grid on; xlabel('Time [s]'); ylabel('Speed [km/h]');
    title('0–200–0 MATLAB vs Python'); legend;
catch ME
    warning('Overlay skipped: %s', ME.message);
end
