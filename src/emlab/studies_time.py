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

        outputs.append(evolve(state, steps, current, probes, snapshot_stride=max(1, steps // 24)))
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
