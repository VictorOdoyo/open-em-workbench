import numpy as np


def _pulse_capture(scale, device, reference, start, stop):
    from .core import C0
    from .fdtd1d import evolve, sheet_current
    from .monitors import fourier_integral, spectral_ratio
    from .sources import gaussian
    from .timedomain import numerical_wavenumber

    if device.dt != reference.dt or device.nodes != reference.nodes:
        raise ValueError("paired device/reference grids and timesteps must match")
    steps = int(np.ceil(12e-9 / device.dt))
    source_node = 100 * scale
    probes = (200 * scale, 450 * scale)
    frequency = np.linspace(1e9, 3e9, 201)
    outputs = []
    for state in (reference, device):

        def current(time, state=state):
            return sheet_current(state, source_node, gaussian(time, 1.5e-9, 0.25e-9, 2e9))

        outputs.append(
            evolve(
                state,
                steps,
                current,
                probes,
                snapshot_stride=max(1, int(1 / (8 * float(frequency.max()) * state.dt))),
            )
        )
    baseline, observed = outputs
    incident = fourier_integral(baseline["electric"], reference.dt, frequency, reference.dt)
    total = fourier_integral(observed["electric"], device.dt, frequency, device.dt)
    reflected, valid_r = spectral_ratio(total[:, 0] - incident[:, 0], incident[:, 0])
    transmitted, valid_t = spectral_ratio(total[:, 1], incident[:, 1])
    valid = valid_r & valid_t
    k = numerical_wavenumber(frequency, device.dx, device.dt, C0)
    reflected *= np.exp(2j * k * (start - probes[0]) * device.dx)
    transmitted *= np.exp(-1j * k * (stop - start) * device.dx)
    arrays = {
        "frequency_hz": frequency,
        "frequency_ghz": frequency / 1e9,
        "valid": valid,
        "r": reflected,
        "t": transmitted,
        "R": abs(reflected) ** 2,
        "T": abs(transmitted) ** 2,
        "A": 1 - abs(reflected) ** 2 - abs(transmitted) ** 2,
        "time_ns": observed["electric_time_s"] * 1e9,
        "reference_left_v_m": baseline["electric"][:, 0],
        "device_left_v_m": observed["electric"][:, 0],
        "reflected_left_v_m": observed["electric"][:, 0] - baseline["electric"][:, 0],
        "reference_electric": baseline["electric"],
        "device_electric": observed["electric"],
        "reference_magnetic": baseline["magnetic"],
        "device_magnetic": observed["magnetic"],
        "snapshots": observed["snapshots"],
        "snapshot_times_ns": observed["snapshot_steps"] * device.dt * 1e9,
        "final_e": observed["final_e"],
        "final_h": observed["final_h"],
        "incident_spectrum": incident,
        "device_spectrum": total,
    }
    parameters = {
        "nodes": device.nodes,
        "spacing_m": device.dx,
        "dt_s": device.dt,
        "steps": steps,
        "pml_cells_each_side": device.pml_cells,
        "slab_start_m": start * device.dx,
        "slab_end_m": stop * device.dx,
        "source_position_m": source_node * device.dx,
        "probe_positions_m": [p * device.dx for p in probes],
        "sheet_current_peak_a_m": 1.0,
        "pulse_center_s": 1.5e-9,
        "pulse_width_s": 0.25e-9,
        "pulse_carrier_hz": 2e9,
        "spectrum_points": len(frequency),
        "spectrum_floor_relative": 1e-3,
        "snapshot_samples_per_fastest_analyzed_period": 8,
    }
    return arrays, parameters


def _slab_result(name, arrays, parameters, reference, material, seed):
    from .core import relative_error
    from .results import curve, make

    valid = arrays["valid"]
    arrays.update(
        {
            "reference_r": reference["r"],
            "reference_t": reference["t"],
            "reference_R": reference["R"],
            "reference_T": reference["T"],
            "reference_A": reference["A"],
        }
    )
    summary = {
        "reflection_complex_relative_l2": relative_error(reference["r"][valid], arrays["r"][valid]),
        "transmission_complex_relative_l2": relative_error(reference["t"][valid], arrays["t"][valid]),
        "maximum_absorption_error": np.max(abs(arrays["A"][valid] - reference["A"][valid])),
        "valid_frequency_points": int(valid.sum()),
        "final_device_peak_e_v_m": np.max(abs(arrays["final_e"])),
    }
    parameters = {**parameters, **material, "seed": seed}
    snapshot_plot = {
        "kind": "image",
        "title": "Recorded device field evolution",
        "array": "snapshots",
        "x": (np.arange(parameters["nodes"]) * parameters["spacing_m"]).tolist(),
        "y": arrays["snapshot_times_ns"].tolist(),
        "xlabel": "Position (m)",
        "ylabel": "Time (ns)",
        "colorbar": "Ey (V/m)",
        "component": "real",
        "mask": None,
        "aspect": "auto",
    }
    plots = [
        curve(
            "Reflected power at the slab reference plane",
            "frequency_ghz",
            [("R", "Yee time domain"), ("reference_R", "analytic multilayer")],
            "Frequency (GHz)",
            "Power reflectance",
            valid="valid",
        ),
        curve(
            "Transmitted power at the slab reference plane",
            "frequency_ghz",
            [("T", "Yee time domain"), ("reference_T", "analytic multilayer")],
            "Frequency (GHz)",
            "Power transmittance",
            valid="valid",
        ),
        curve(
            "Incident normalization and measured reflection",
            "time_ns",
            [("reference_left_v_m", "reference incident"), ("reflected_left_v_m", "device minus reference")],
            "Electric sample time (ns)",
            "Ey (V/m)",
        ),
        snapshot_plot,
    ]
    rows = [
        {
            "frequency_hz": f,
            "valid": bool(v),
            "R": r,
            "T": t,
            "A": a,
            "reference_R": rr,
            "reference_T": rt,
            "reference_A": ra,
        }
        for f, v, r, t, a, rr, rt, ra in zip(
            arrays["frequency_hz"],
            valid,
            arrays["R"],
            arrays["T"],
            arrays["A"],
            reference["R"],
            reference["T"],
            reference["A"],
            strict=True,
        )
    ]
    return make(
        name,
        parameters,
        summary,
        arrays,
        plots,
        {"spectrum": rows},
        [
            "Device and incident-reference simulations use the same physical source and timestep.",
            "Scattering uses electric-field subtraction; no true channel or frequency-domain field enters Yee updates.",
            "Complex amplitudes are deembedded with the vacuum numerical wavenumber; both exterior ports are vacuum.",
            "Interface electric nodes use half-cell arithmetic material averages. Capture duration is finite.",
            "Fourier ratios below the incident-spectrum threshold are explicitly masked.",
            "Magnetic probe samples retain their previous half-step time; these scattering ratios use E only.",
        ],
    )


def slab(scale=1, seed=42):
    from .fdtd1d import CPML1D
    from .layers import stack

    nodes, spacing, start, stop, pml = 600 * scale + 1, 0.001 / scale, 250 * scale, 350 * scale, 40 * scale
    epsilon = np.ones(nodes)
    epsilon[start + 1 : stop] = 4
    epsilon[[start, stop]] = 2.5
    device = CPML1D(nodes, spacing, pml_cells=pml, epsilon_r=epsilon)
    baseline = CPML1D(nodes, spacing, pml_cells=pml)
    arrays, parameters = _pulse_capture(scale, device, baseline, start, stop)
    reference = stack(arrays["frequency_hz"], [4], [(stop - start) * spacing])
    return _slab_result("fdtd-slab", arrays, parameters, reference, {"slab_epsilon_r": 4}, seed)


def debye_slab(scale=1, seed=42):
    from .fdtd1d import CPML1D, Debye1D
    from .layers import stack
    from .materials import debye
    from .timedomain import debye_permittivity

    nodes, spacing, start, stop, pml = 600 * scale + 1, 0.001 / scale, 250 * scale, 350 * scale, 40 * scale
    high, static = np.ones(nodes), np.ones(nodes)
    high[start + 1 : stop], static[start + 1 : stop] = 2, 6
    high[[start, stop]], static[[start, stop]] = 1.5, 3.5
    tau = 0.12e-9
    device = Debye1D(nodes, spacing, static, high, tau, pml_cells=pml)
    baseline = CPML1D(nodes, spacing, pml_cells=pml)
    arrays, parameters = _pulse_capture(scale, device, baseline, start, stop)
    arrays["final_polarization_c_m2"] = device.polarization.copy()
    epsilon = debye(arrays["frequency_hz"], 6, 2, tau)
    reference = stack(arrays["frequency_hz"], [epsilon], [(stop - start) * spacing])
    discrete = debye_permittivity(arrays["frequency_hz"], 6, 2, tau, device.dt)
    discrete_reference = stack(arrays["frequency_hz"], [discrete], [(stop - start) * spacing])
    arrays.update(
        {
            "epsilon_continuum": epsilon,
            "epsilon_discrete": discrete,
            "discrete_reference_R": discrete_reference["R"],
            "discrete_reference_T": discrete_reference["T"],
        }
    )
    result = _slab_result(
        "fdtd-debye",
        arrays,
        parameters,
        reference,
        {"epsilon_static": 6, "epsilon_infinity": 2, "relaxation_s": tau},
        seed,
    )
    result["summary"]["maximum_discrete_continuum_material_difference"] = float(
        np.max(abs(discrete - epsilon))
    )
    result["summary"]["minimum_measured_absorption"] = float(np.min(arrays["A"][arrays["valid"]]))
    result["assumptions"].append(
        "Debye polarization is evolved by an auxiliary differential equation, not a fitted constant loss."
    )
    return result


def _cavity_capture(scale, epsilon, conductivity):
    from .core import C0, EPS0
    from .fdtd2d import TMz2D, cavity_mode, conductive_cavity_pole, stored_energy
    from .grid import from_bounds
    from .modes import rectangular_cutoff
    from .monitors import ringdown_pole

    grid = from_bounds((0, 0.2), (0, 0.1), 40 * scale + 1, 30 * scale + 1)
    state = TMz2D(grid, epsilon_r=epsilon, conductivity_s_m=conductivity)
    cavity_mode(state, 1, 1)
    initial = state.ez.copy()
    initial_energy = stored_energy(state)
    steps = 1400 * scale
    signal, energy, invariant = [], [], []
    for _ in range(steps):
        state.step()
        signal.append(state.ez[grid.ny // 3, grid.nx // 2])
        energy.append(stored_energy(state) / initial_energy)
        if conductivity == 0:
            invariant.append(stored_energy(state, modified=True) / initial_energy)
    signal = np.array(signal)
    time = np.arange(1, steps + 1) * state.dt
    fit = ringdown_pole(signal, state.dt)
    roots = conductive_cavity_pole(grid, 1, 1, state.dt, epsilon, conductivity)
    pole = roots[roots.imag > 0][0]
    discrete_frequency = np.angle(pole) / (2 * np.pi * state.dt)
    discrete_decay = max(0.0, -np.log(abs(pole)) / state.dt)
    omega0 = 2 * np.pi * rectangular_cutoff(0.2, 0.1, 1, 1, "tm", epsilon)
    alpha = conductivity / (2 * EPS0 * epsilon)
    continuum_frequency = np.sqrt(omega0**2 - alpha**2) / (2 * np.pi)
    fit_time = time - time[0]
    design = np.exp(-fit["decay_rate_s"] * fit_time[:, None]) * np.column_stack(
        (
            np.cos(2 * np.pi * fit["frequency_hz"] * fit_time),
            np.sin(2 * np.pi * fit["frequency_hz"] * fit_time),
        )
    )
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    arrays = {
        "time_ns": time * 1e9,
        "electric_v_m": signal,
        "fitted_v_m": design @ coefficients,
        "energy_snapshot": energy,
        "energy_reference": invariant if conductivity == 0 else np.exp(-2 * alpha * time),
        "initial_e": initial,
        "final_e": state.ez.copy(),
        "final_hx": state.hx.copy(),
        "final_hy": state.hy.copy(),
    }
    summary = {
        "measured_frequency_hz": fit["frequency_hz"],
        "continuum_frequency_hz": continuum_frequency,
        "discrete_frequency_hz": discrete_frequency,
        "relative_discrete_frequency_error": abs(fit["frequency_hz"] / discrete_frequency - 1),
        "continuum_frequency_relative_error": abs(fit["frequency_hz"] / continuum_frequency - 1),
        "quality_factor_measured": fit["quality_factor"],
        "pole_fit_residual": fit["fit_residual"],
    }
    if conductivity == 0:
        summary["maximum_relative_discrete_energy_drift"] = np.max(abs(np.array(invariant) - 1))
    else:
        summary["continuum_quality_factor"] = np.pi * continuum_frequency / alpha
        summary["discrete_quality_factor"] = np.pi * discrete_frequency / discrete_decay
        summary["relative_discrete_quality_error"] = abs(
            fit["quality_factor"] / summary["discrete_quality_factor"] - 1
        )
    parameters = {
        "width_m": 0.2,
        "height_m": 0.1,
        "grid_nodes": [grid.nx, grid.ny],
        "epsilon_r": epsilon,
        "mu_r": 1,
        "conductivity_s_m": conductivity,
        "mode_indices": [1, 1],
        "initial_peak_e_v_m": 1,
        "dt_s": state.dt,
        "steps": steps,
        "duration_s": steps * state.dt,
        "boundary": "PEC",
        "vacuum_speed_m_s": C0,
    }
    return grid, arrays, parameters, summary


def _cavity_result(name, capture, seed):
    from .results import curve, image, make

    grid, arrays, parameters, summary = capture
    lossless = parameters["conductivity_s_m"] == 0
    label = "exact discrete invariant" if lossless else "continuous energy-decay envelope"
    return make(
        name,
        {**parameters, "seed": seed},
        summary,
        arrays,
        [
            curve(
                "Isolated cavity ringdown",
                "time_ns",
                [("electric_v_m", "Yee trace"), ("fitted_v_m", "isolated-pole fit")],
                "Electric sample time (ns)",
                "Ez (V/m)",
            ),
            curve(
                "Stored E/H samples and independent energy reference",
                "time_ns",
                [("energy_snapshot", "staggered snapshot"), ("energy_reference", label)],
                "Electric sample time (ns)",
                "Energy / initial energy",
            ),
            image("Initial electric eigenfield", "initial_e", grid, "Ez (V/m)"),
            image("Final electric field", "final_e", grid, "Ez (V/m)"),
        ],
        {
            "trace": [
                {
                    "time_s": t * 1e-9,
                    "electric_v_m": e,
                    "energy_snapshot_normalized": u,
                    "energy_reference_normalized": r,
                }
                for t, e, u, r in zip(
                    arrays["time_ns"],
                    arrays["electric_v_m"],
                    arrays["energy_snapshot"],
                    arrays["energy_reference"],
                    strict=True,
                )
            ]
        },
        [
            "Uniform rectangular PEC cavity initialized in one TM mode; no impressed current during ringdown.",
            "E and H snapshots are staggered in time. Only the lossless modified leapfrog energy is an exact invariant.",
            "The isolated-pole fit estimates one damped mode; it is not a generic multimode spectrum estimator.",
            "Continuum, discrete-recurrence and measured resonance predictions are retained separately.",
        ],
    )


def cavity(scale=1, seed=42):
    return _cavity_result("fdtd-cavity", _cavity_capture(scale, 1.0, 0.0), seed)
