#!/usr/bin/env python3
"""Graphene-hBN-WSe2 multilayer dispersion using a 4x4 anisotropic TMM.

Default ``report`` parameters reproduce the configuration discussed in
"Academic Internship.pdf": 1 nm hBN spacers, EF=350 meV, five WSe2 layers,
Er=115 meV, linewidth=5 meV and normalized resonance strength A=10.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12
E_CHARGE = 1.602_176_634e-19
HBAR = 1.054_571_817e-34
KB = 1.380_649e-23
M_E = 9.109_383_7015e-31
V_F = 1.0e6


# ---------------------------------------------------------------------------
# Compact embedded SiO2 dataset
# ---------------------------------------------------------------------------
_SIO2_L_UM = np.array([
    12.4082397003745,12.1055997076825,11.8173711432138,11.5425485584879,
    11.2802179094314,11.0295464003329,10.7897736524996,10.5602040003188,
    10.3401997503121,10.1291752656119,9.92659176029963,9.7319527061761,
    9.54479976951887,9.36470920782983,9.1912886669441,9.02417432754512,
    8.86302835741038,8.70753663184178,8.55740668991347,8.412365898559,
    8.27215980024969,8.13655062319642,8.00531593572551,7.87824742880923,
    7.75514981273408,7.6358398156151,7.52014527295426,7.40790429873106,
    7.29896452963208,7.19318243499973,7.0904226859283,6.99055757767579,
    6.89346650020807,6.79903545226002,6.70715659479704,6.61772784019975,
    6.53065247388133,6.44583880538937,6.36319984634592,6.28265301284786,
    6.20411985018727,6.12752577796273,6.05279985384124,5.97987455439737,
    5.90868557160692,5.83917162370566,5.77127427924397,5.70493779327565,
    5.6401089547157,5.57673694398855,5.51477320016646,5.45417129686793,
    5.3948868262498,5.33687729048367,5.28010200015938,5.22452197910507,
    5.17009987515606,5.11679987644311,5.06458763280593,5.01343018196951,
    4.96329588014981,4.91415433678199,4.86597635308805,4.81873386422312,
    4.77239988475943,4.72694845728554,4.68235460391492,4.63859428051384,
    4.59564433347205,4.55348245885304,4.51208716377256,4.4714377298647,
    4.43151417870519,4.39229723907063,4.35376831592089,4.31590946099984,
    4.27870334495674,4.24213323089728,4.2061829492795,4.17083687407547,
    4.13607990012484
])
_SIO2_ER = np.array([
    3.25522592045878,3.08272026564609,3.00372153773939,3.04332436223669,
    3.21037978574888,3.50993313163949,3.96618489799276,4.65063461442254,
    5.66438547748933,6.91822291142148,7.64493088882815,6.36638494090201,
    2.55479517068528,-1.61365015920097,-3.3829707424631,-2.77176169152009,
    -1.55962923306089,-0.758827628371846,-0.391181748903566,-0.226728606295871,
    -0.102432929139175,0.0509482242148542,0.238106134314969,0.434701153332476,
    0.616057667007743,0.76994931543292,0.895703239125033,0.998215970008435,
    1.08319737768052,1.15518624523424,1.21733433956151,1.27177422791124,
    1.31999191343641,1.36307091513326,1.40183350696907,1.43692262639009,
    1.46885216079959,1.49803985506317,1.52482996902649,1.54950946831348,
    1.57231992067797,1.59346643085315,1.6131244732093,1.63144519710769,
    1.64855960094577,1.66458185427042,1.67961196908203,1.69373796765733,
    1.70703765646515,1.71958008877385,1.73142677896772,1.74263271717431,
    1.75324722205741,1.76331466152611,1.77287506493498,1.7819646455999,
    1.7906162487677,1.79885973729305,1.80672232500304,1.81422886592454,
    1.82140210610755,1.82826290361836,1.83483042133835,1.8411222964425,
    1.84715478980822,1.85294291809356,1.85850057080203,1.86384061430084,
    1.86897498446898,1.87391476940771,1.87867028344244,1.88325113347332,
    1.8876662785865,1.89192408371561,1.89603236803814,1.89999844870259,
    1.90382918040583,1.90753099127511,1.91110991545241,1.91457162273103,
    1.91792144555177
])
_SIO2_EI = np.array([
    1.46724370167272,1.34616783721392,1.15296065471604,0.928273799873014,
    0.711684658091069,0.533097649327091,0.417658637099922,0.425158602484518,
    0.760722565749989,1.90460089468079,4.37596369144577,7.68481567134704,
    9.6083569425428,8.27863106652267,4.95960421656432,2.32787646832115,
    1.23502489518456,0.987571936347749,0.923202396461516,0.805999407526822,
    0.621440851720058,0.420403622835509,0.251636374529702,0.136546720831889,
    0.0708680368217977,0.038660127082164,0.0245173010712778,0.01847109141969094,
    0.0155648834565629,0.0137943839492764,0.012469708188975,0.0113727398860479,
    0.0104289583075993,0.00960423618717895,0.00887710767190789,0.00823168338088486,
    0.007655492649574,0.00713851021969094,0.00667255840984731,0.00625089036312628,
    0.0058678856396693,0.00551882234916122,0.00519970351197411,0.00490712278693213,
    0.00463815932196446,0.00439029449352285,0.00416134532440456,0.00394941076397403,
    0.00375282799510439,0.00357013663344828,0.00340004919397568,0.00324142657462426,
    0.00309325758621127,0.00295464176811332,0.00282477488923338,0.0027029366566024,
    0.00258848024905727,0.00248082336761624,0.00237944055248394,0.00228385656274876,
    0.00219364065157479,0.00210840159912155,0.00202778338913732,0.00195146143437748,
    0.0018791392716351,0.00181054565996055,0.00174543202615673,0.00168357021030999,
    0.00162475047130511,0.00156877971825264,0.00151547993875158,0.00146468679909611,
    0.00141624839505556,0.00137002413482761,0.00132588373827914,0.00128370633872554,
    0.00124337967531288,0.00120479936562975,0.00116786824949279,0.00113249579600139,
    0.00109859756693467
])


@dataclass(frozen=True)
class SimulationConfig:
    graphene_model: str = "nonlocal"
    fermi_energy_ev: float = 0.350
    temperature_k: float = 300.0
    relaxation_time_s: float = 0.7e-12
    graphene_thickness_m: float = 0.30e-9
    hbn_top_m: float = 1.0e-9
    hbn_bottom_m: float = 1.0e-9
    tmd_layers: int = 5
    tmd_monolayer_m: float = 0.65e-9
    substrate_thickness_m: float = 300e-9
    tmd_resonance_mev: float = 115.0
    tmd_linewidth_mev: float = 5.0
    tmd_strength: float = 10.0
    tmd_model: str = "normalized"
    energy_min_mev: float = 105.0
    energy_max_mev: float = 140.0
    n_energy: int = 101
    q_min_rad_m: float = 2.5e7
    q_max_rad_m: float = 9.0e7
    n_q: int = 180
    linecut_energy_mev: float = 115.0


def report_config() -> SimulationConfig:
    return SimulationConfig()


def matlab_config() -> SimulationConfig:
    """Parameters in the uploaded BNgrapheneBN_TMD.m."""
    ns = 9.000290360793872e16
    ef_ev = HBAR * V_F * np.sqrt(np.pi * ns) / E_CHARGE
    return SimulationConfig(
        fermi_energy_ev=float(ef_ev), hbn_top_m=2e-9, hbn_bottom_m=2e-9,
        tmd_linewidth_mev=10.0, tmd_strength=1.0, tmd_model="legacy",
        energy_min_mev=111.6, energy_max_mev=148.8, q_min_rad_m=10.0,
        q_max_rad_m=1.5e8, n_energy=101, n_q=200,
    )


class SiO2Data:
    """Complex SiO2 permittivity with optional full-resolution MAT loading."""

    def __init__(self, mat_path: str | None = None):
        if mat_path:
            from scipy.io import loadmat
            data = loadmat(mat_path)
            wavelength_um = np.asarray(data["L"]).ravel()
            er = np.asarray(data["RepshilonSiO2"]).ravel()
            ei = np.asarray(data["IepshilonSiO2"]).ravel()
        else:
            wavelength_um, er, ei = _SIO2_L_UM, _SIO2_ER, _SIO2_EI
        order = np.argsort(wavelength_um)
        self.wavelength_um = wavelength_um[order]
        self.er = er[order]
        self.ei = ei[order]

    def epsilon(self, wavelength_m: float | np.ndarray) -> complex | np.ndarray:
        x = np.asarray(wavelength_m) * 1e6
        er = np.interp(x, self.wavelength_um, self.er)
        ei = np.interp(x, self.wavelength_um, self.ei)
        result = er + 1j * ei
        return result.item() if result.ndim == 0 else result


# ---------------------------------------------------------------------------
# Material models
# ---------------------------------------------------------------------------
def hbn_parallel(wavenumber_cm: float | np.ndarray) -> complex | np.ndarray:
    """hBN in-plane epsilon (upper Reststrahlen band)."""
    w = np.asarray(wavenumber_cm, dtype=complex)
    return 4.90 * (1.0 + (1614.0**2 - 1360.0**2) /
                   (1360.0**2 - w**2 - 1j * 7.0 * w))


def hbn_perpendicular(wavenumber_cm: float | np.ndarray) -> complex | np.ndarray:
    """hBN out-of-plane epsilon (lower Reststrahlen band)."""
    w = np.asarray(wavenumber_cm, dtype=complex)
    return 2.95 * (1.0 + (825.0**2 - 760.0**2) /
                   (760.0**2 - w**2 - 1j * 2.0 * w))


def wse2_permittivity(omega: float, resonance_mev: float, linewidth_mev: float,
                      strength: float, model: str = "normalized") -> complex:
    """Out-of-plane WSe2 Lorentz oscillator.

    normalized: Eq. (7) of the report, where Im(epsilon) at resonance is A.
    legacy: uploaded FunctionWSe2.m, where oscillator numerator is omega_r^2.
    """
    wr = resonance_mev * 1e-3 * E_CHARGE / HBAR
    gamma = linewidth_mev * 1e-3 * E_CHARGE / HBAR
    denominator = wr**2 - omega**2 - 1j * omega * gamma
    if model == "legacy":
        oscillator = wr**2 / denominator
    else:
        normalization = wr / gamma
        oscillator = (strength / normalization) * wr**2 / denominator
    return 4.2 + oscillator


def graphene_kubo_permittivity(omega: float, ef_ev: float, tau_s: float,
                               temperature_k: float, thickness_m: float) -> Tuple[complex, complex]:
    """Local finite-temperature Kubo model translated from the supplied MATLAB."""
    mu = ef_ev * E_CHARGE
    z = omega - 1j / tau_s
    sigma_inter = (-1j * E_CHARGE**2 / (4 * np.pi * HBAR)) * np.log(
        (2 * abs(mu) - HBAR * z) / (2 * abs(mu) + HBAR * z)
    )
    thermal = mu / (KB * temperature_k) + 2 * np.log1p(np.exp(-mu / (KB * temperature_k)))
    sigma_intra = (-1j * E_CHARGE**2 * KB * temperature_k /
                   (np.pi * HBAR**2 * (omega - 1j / tau_s))) * thermal
    # The conjugation follows the e^{-i omega t} convention used by the TMM.
    sigma = np.conj(sigma_intra + sigma_inter)
    eps_parallel = 1.0 + 1j * sigma / (omega * EPS0 * thickness_m)
    return complex(eps_parallel), 1.0 + 0j


def _graphene_g(z: complex) -> complex:
    return z * np.sqrt(z - 1 + 0j) * np.sqrt(z + 1 + 0j) - np.log(
        z + np.sqrt(z + 1 + 0j) * np.sqrt(z - 1 + 0j)
    )


def graphene_nonlocal_conductivity(omega: float, q_rad_m: float, ef_ev: float,
                                   tau_s: float, v_f: float = V_F) -> complex:
    """Nonlocal RPA conductivity translated from func_nonlocal_RPA.m."""
    q = max(float(abs(q_rad_m)), 1e-12)
    kf = ef_ev * E_CHARGE / (HBAR * v_f)
    wt = omega + 1j / tau_s
    dp = (wt / v_f + 2 * kf) / q
    dn = (wt / v_f - 2 * kf) / q
    x = 2 * kf / q
    conductance_quantum = E_CHARGE**2 / (2 * np.pi * HBAR)
    a = 8 * kf / (v_f * q**2)
    if np.real(dn) < -1:
        dn_term = _graphene_g(-dn)
    elif np.real(dn) > 1:
        dn_term = _graphene_g(dn) + 1j * np.pi
    else:
        dn_term = 0.0j
    b = (dn_term - _graphene_g(dp)) / np.sqrt(wt**2 - (v_f * q)**2 + 0j)
    chi = 0.5 * conductance_quantum * (a + b)
    static_tail = 0.0
    if x < 1:
        static_tail = x * np.sqrt(1 - x**2) - np.arccos(x)
    chi0 = conductance_quantum / (v_f * q) * (4 * kf / q - static_tail)
    corrected = (1 + 1j / (wt * tau_s)) * chi / (1 + 1j * chi / (wt * tau_s * chi0))
    return complex(-1j * wt * corrected)


def graphene_nonlocal_permittivity(omega: float, q_rad_m: float, ef_ev: float,
                                   tau_s: float, thickness_m: float) -> Tuple[complex, complex]:
    sigma = graphene_nonlocal_conductivity(omega, q_rad_m, ef_ev, tau_s)
    return 1.0 + 1j * sigma / (omega * EPS0 * thickness_m), 1.0 + 0j


def gold_drude_permittivity(omega: float) -> complex:
    """Gold Drude model using Eq. (8) and parameters stated in the report."""
    density = 5.9e28
    effective_mass = 0.99 * M_E
    tau = 9.3e-15
    sigma = density * E_CHARGE**2 / (effective_mass * (1 / tau - 1j * omega))
    return complex(1.0 + 1j * sigma / (EPS0 * omega))


def silicon_index(wavelength_m: float) -> complex:
    """Exit-medium Sellmeier expression used throughout the MATLAB scripts."""
    wavelength_um = wavelength_m * 1e6
    return complex(np.sqrt(11.67316 + 1 / wavelength_um**2 +
                           0.004482633 / (wavelength_um**2 - 1.108205**2) + 0j))


def diagonal_tensor(eps_x: complex, eps_y: complex, eps_z: complex) -> np.ndarray:
    return np.diag(np.asarray([eps_x, eps_y, eps_z], dtype=complex))


# ---------------------------------------------------------------------------
# 4x4 anisotropic transfer-matrix method
# ---------------------------------------------------------------------------
def incident_matrix_inverse(n_incident: complex, phi: complex) -> np.ndarray:
    cphi = np.cos(phi)
    m = np.zeros((4, 4), dtype=complex)
    m[0, 1], m[0, 2] = 1, -1 / (n_incident * cphi)
    m[1, 1], m[1, 2] = 1, +1 / (n_incident * cphi)
    m[2, 0], m[2, 3] = 1 / cphi, 1 / n_incident
    m[3, 0], m[3, 3] = -1 / cphi, 1 / n_incident
    return 0.5 * m


def exit_matrix(n_incident: complex, n_exit: complex, phi: complex) -> np.ndarray:
    cos_exit = np.sqrt(1 - ((n_incident / n_exit) * np.sin(phi))**2 + 0j)
    m = np.zeros((4, 4), dtype=complex)
    m[0, 2] = cos_exit
    m[1, 0] = 1
    m[2, 0] = -n_exit * cos_exit
    m[3, 2] = n_exit
    return m


def layer_matrix(epsilon: np.ndarray, n_incident: complex, wavelength_m: float,
                 thickness_m: float, phi: complex) -> np.ndarray:
    kx = n_incident * np.sin(phi)  # dimensionless kx/k0
    e = epsilon
    delta = np.zeros((4, 4), dtype=complex)
    delta[0, 0] = -kx * e[2, 0] / e[2, 2]
    delta[0, 1] = -kx * e[2, 1] / e[2, 2]
    delta[0, 3] = 1 - kx**2 / e[2, 2]
    delta[1, 2] = -1
    delta[2, 0] = e[1, 2] * e[2, 0] / e[2, 2] - e[1, 0]
    delta[2, 1] = kx**2 - e[1, 1] + e[1, 2] * e[2, 1] / e[2, 2]
    delta[2, 3] = kx * e[1, 2] / e[2, 2]
    delta[3, 0] = e[0, 0] - e[0, 2] * e[2, 0] / e[2, 2]
    delta[3, 1] = e[0, 1] - e[0, 2] * e[2, 1] / e[2, 2]
    delta[3, 3] = -kx * e[0, 2] / e[2, 2]
    k0 = 2 * np.pi / wavelength_m
    # Negative thickness preserves the propagation convention of the MATLAB code.
    return expm(-1j * k0 * delta * thickness_m)


def reflection_transmission(n_incident: complex, wavelength_m: float,
                            tensors: Iterable[np.ndarray], thicknesses_m: Iterable[float],
                            n_exit: complex, phi: complex) -> Tuple[complex, complex, complex, complex]:
    total = incident_matrix_inverse(n_incident, phi)
    for tensor, thickness in zip(tensors, thicknesses_m):
        total = total @ layer_matrix(tensor, n_incident, wavelength_m, thickness, phi)
    total = total @ exit_matrix(n_incident, n_exit, phi)
    denominator = total[0, 0] * total[2, 2] - total[0, 2] * total[2, 0]
    rs = (total[1, 0] * total[2, 2] - total[1, 2] * total[2, 0]) / denominator
    rp = (total[0, 0] * total[3, 2] - total[3, 0] * total[0, 2]) / denominator
    ts = total[2, 2] / denominator
    tp = total[0, 0] / denominator
    return complex(rs), complex(rp), complex(ts), complex(tp)


def build_stack(cfg: SimulationConfig, substrate: str, wavelength_m: float,
                omega: float, q_rad_m: float, sio2: SiO2Data) -> Tuple[list[np.ndarray], list[float]]:
    wn_cm = 1.0 / (wavelength_m * 100.0)
    hbn = diagonal_tensor(hbn_parallel(wn_cm), hbn_parallel(wn_cm), hbn_perpendicular(wn_cm))
    if cfg.graphene_model == "kubo":
        eg_t, eg_z = graphene_kubo_permittivity(
            omega, cfg.fermi_energy_ev, cfg.relaxation_time_s,
            cfg.temperature_k, cfg.graphene_thickness_m)
    else:
        eg_t, eg_z = graphene_nonlocal_permittivity(
            omega, q_rad_m, cfg.fermi_energy_ev,
            cfg.relaxation_time_s, cfg.graphene_thickness_m)
    graphene = diagonal_tensor(eg_t, eg_t, eg_z)
    tmd_z = wse2_permittivity(
        omega, cfg.tmd_resonance_mev, cfg.tmd_linewidth_mev,
        cfg.tmd_strength, cfg.tmd_model)
    tmd = diagonal_tensor(12.7, 12.7, tmd_z)
    eps_sub = sio2.epsilon(wavelength_m) if substrate == "sio2" else gold_drude_permittivity(omega)
    sub = diagonal_tensor(eps_sub, eps_sub, eps_sub)
    tensors = [hbn, graphene, hbn.copy(), tmd, sub]
    thicknesses = [cfg.hbn_top_m, cfg.graphene_thickness_m, cfg.hbn_bottom_m,
                   cfg.tmd_layers * cfg.tmd_monolayer_m, cfg.substrate_thickness_m]
    return tensors, thicknesses


def simulate(cfg: SimulationConfig, substrate: str, sio2: SiO2Data,
             progress: bool = True) -> Dict[str, np.ndarray]:
    energies = np.linspace(cfg.energy_min_mev, cfg.energy_max_mev, cfg.n_energy)
    q_values = np.linspace(cfg.q_min_rad_m, cfg.q_max_rad_m, cfg.n_q)
    rp = np.empty((cfg.n_energy, cfg.n_q), dtype=complex)
    started = time.perf_counter()
    for i, energy_mev in enumerate(energies):
        omega = energy_mev * 1e-3 * E_CHARGE / HBAR
        wavelength = 2 * np.pi * C0 / omega
        n_exit = silicon_index(wavelength)
        for j, q in enumerate(q_values):
            # Complex phi is intentional for evanescent high-momentum modes.
            phi = np.arcsin(q * wavelength / (2 * np.pi) + 0j)
            tensors, thicknesses = build_stack(cfg, substrate, wavelength, omega, q, sio2)
            _, rp[i, j], _, _ = reflection_transmission(
                1.0 + 0j, wavelength, tensors, thicknesses, n_exit, phi)
        if progress and (i == 0 or (i + 1) % max(1, cfg.n_energy // 10) == 0):
            elapsed = time.perf_counter() - started
            print(f"  {substrate:>4}: {i + 1:>3}/{cfg.n_energy} energies, {elapsed:6.1f} s")
    return {"energy_mev": energies, "q_rad_m": q_values, "rp": rp,
            "loss": np.abs(np.imag(rp))}


def estimate_splitting(energy_mev: np.ndarray, trace: np.ndarray) -> Tuple[float, np.ndarray]:
    peaks, properties = find_peaks(trace, prominence=max(1e-12, 0.04 * np.ptp(trace)))
    if peaks.size < 2:
        return float("nan"), peaks
    selected = peaks[np.argsort(trace[peaks])[-2:]]
    selected.sort()
    return float(energy_mev[selected[1]] - energy_mev[selected[0]]), selected


def plot_results(results: Dict[str, Dict[str, np.ndarray]],
                 cfg: SimulationConfig) -> None:
    names = list(results)
    fig = plt.figure(figsize=(6.8 * len(names), 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, len(names), height_ratios=[2.2, 1.0])
    cmap = "inferno"
    for col, substrate in enumerate(names):
        data = results[substrate]
        e, q, loss = data["energy_mev"], data["q_rad_m"], data["loss"]
        ax = fig.add_subplot(grid[0, col])
        mesh = ax.pcolormesh(q / 1e7, e, loss, shading="auto", cmap=cmap,
                             vmin=np.nanpercentile(loss, 2), vmax=np.nanpercentile(loss, 99.5))
        ax.axhline(cfg.tmd_resonance_mev, color="cyan", lw=1.0, ls="--", alpha=0.85)
        ax.set_title(f"{substrate.upper()} substrate")
        ax.set_xlabel(r"In-plane momentum $q$ ($10^7$ rad m$^{-1}$)")
        if col == 0:
            ax.set_ylabel("Energy (meV)")
        fig.colorbar(mesh, ax=ax, label=r"$|\mathrm{Im}(r_p)|$")

        # PDF Fig. 5/6 line cuts: 3.9e7 rad/m for SiO2 and 7.2e7 rad/m for Au.
        # If a custom q window excludes that value, the nearest available edge is used.
        target_q = 3.9e7 if substrate == "sio2" else 7.2e7
        q_index = int(np.argmin(np.abs(q - target_q)))
        trace = loss[:, q_index]
        splitting, peaks = estimate_splitting(e, trace)
        bx = fig.add_subplot(grid[1, col])
        bx.plot(e, trace, color="#b21f35", lw=1.8)
        if peaks.size:
            bx.plot(e[peaks], trace[peaks], "o", color="#164c96", ms=4)
        split_text = "not resolved" if np.isnan(splitting) else f"{splitting:.2f} meV"
        bx.set_title(f"Line cut: q={q[q_index]/1e7:.2f}e7 rad/m; splitting {split_text}")
        bx.set_xlabel("Energy (meV)")
        if col == 0:
            bx.set_ylabel(r"$|\mathrm{Im}(r_p)|$")
        bx.grid(alpha=0.25)

    fig.suptitle(
        f"hBN({cfg.hbn_top_m*1e9:g} nm)/graphene({cfg.fermi_energy_ev*1e3:.0f} meV)/"
        f"hBN({cfg.hbn_bottom_m*1e9:g} nm)/WSe2({cfg.tmd_layers} layers)\n"
        f"{cfg.graphene_model.upper()} graphene; WSe2 Er={cfg.tmd_resonance_mev:g} meV, "
        f"gamma={cfg.tmd_linewidth_mev:g} meV, A={cfg.tmd_strength:g}", fontsize=14)
    # Display the interactive figure. No image file is written.
    plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preset", choices=["report", "matlab"], default="report")
    p.add_argument("--graphene", choices=["nonlocal", "kubo"], default=None)
    p.add_argument("--substrates", choices=["both", "sio2", "gold"], default="both")
    p.add_argument("--sio2-mat", help="Optional full-resolution WaveLength_SiO2.mat")
    p.add_argument("--ef-mev", type=float, help="Override graphene Fermi energy")
    p.add_argument("--hbn-top-nm", type=float)
    p.add_argument("--hbn-bottom-nm", type=float)
    p.add_argument("--tmd-layers", type=int)
    p.add_argument("--resonance-mev", type=float)
    p.add_argument("--linewidth-mev", type=float)
    p.add_argument("--strength", type=float)
    p.add_argument("--energy-min", type=float)
    p.add_argument("--energy-max", type=float)
    p.add_argument("--n-energy", type=int)
    p.add_argument("--q-min", type=float, help="Minimum q in rad/m")
    p.add_argument("--q-max", type=float, help="Maximum q in rad/m")
    p.add_argument("--n-q", type=int)
    p.add_argument("--save-data", action="store_true", help="Save numerical arrays as NPZ")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def configuration_from_args(args: argparse.Namespace) -> SimulationConfig:
    cfg = report_config() if args.preset == "report" else matlab_config()
    mapping = {
        "graphene_model": args.graphene,
        "fermi_energy_ev": None if args.ef_mev is None else args.ef_mev / 1000,
        "hbn_top_m": None if args.hbn_top_nm is None else args.hbn_top_nm * 1e-9,
        "hbn_bottom_m": None if args.hbn_bottom_nm is None else args.hbn_bottom_nm * 1e-9,
        "tmd_layers": args.tmd_layers,
        "tmd_resonance_mev": args.resonance_mev,
        "tmd_linewidth_mev": args.linewidth_mev,
        "tmd_strength": args.strength,
        "energy_min_mev": args.energy_min,
        "energy_max_mev": args.energy_max,
        "n_energy": args.n_energy,
        "q_min_rad_m": args.q_min,
        "q_max_rad_m": args.q_max,
        "n_q": args.n_q,
    }
    return replace(cfg, **{key: value for key, value in mapping.items() if value is not None})


def main() -> None:
    args = parse_args()
    cfg = configuration_from_args(args)
    if cfg.n_energy < 2 or cfg.n_q < 2:
        raise ValueError("n-energy and n-q must both be at least 2")
    sio2 = SiO2Data(args.sio2_mat)
    substrates = ["sio2", "gold"] if args.substrates == "both" else [args.substrates]
    print("Running anisotropic 4x4 TMM")
    print(cfg)
    results = {name: simulate(cfg, name, sio2, progress=not args.quiet) for name in substrates}
    plot_results(results, cfg)
    if args.save_data:
        payload = {f"{name}_{key}": value for name, result in results.items()
                   for key, value in result.items()}
        data_path = Path.home() / "Documents" / "Graphene_hBN_WSe2_Dispersion.npz"
        np.savez_compressed(data_path, **payload)
        print(f"Saved data: {data_path}")


if __name__ == "__main__":
    main()