import numpy as np
import pytest

from vna_app.processing.filters import average_sweeps, smooth_complex
from vna_app.processing.impedance import C0_M_S, gamma_to_y, gamma_to_z, shift_reference_plane
from vna_app.processing.permittivity import (
    debye,
    eps_capacitive,
    eps_three_reference,
    to_eps_columns,
    water_25c,
)

F = np.linspace(400e6, 3e9, 101)
W = 2 * np.pi * F
C0 = 0.08e-12
Z0 = 50.0


def probe_gamma(eps):
    """Γ idéal au plan de la sonde, Y = jωC0 ε*."""
    y = 1j * W * C0 * eps
    return (1 - Z0 * y) / (1 + Z0 * y)


def error_box(g):
    e00 = 0.07 * np.exp(1j * F / 1e9)
    e11 = 0.1 * np.exp(-1j * F / 2e9)
    e0110 = 0.9 * np.exp(-1j * F / 4e8)
    return e00 + e0110 * g / (1 - e11 * g)


def measure(eps):
    return error_box(probe_gamma(eps))


def test_z_and_y():
    g = np.array([0.0, 1 / 3, -1 / 3])
    np.testing.assert_allclose(gamma_to_z(g), [50, 100, 25])
    np.testing.assert_allclose(gamma_to_y(g), 1 / gamma_to_z(g))
    assert np.isinf(gamma_to_z(np.array([1.0]))[0])


def test_shift_reference_plane_roundtrip():
    length, er = 0.05, 2.1
    g_dut = 0.4 * np.exp(1j * np.linspace(0, 2, len(F)))
    beta = 2 * np.pi * F * np.sqrt(er) / C0_M_S
    g_meas = g_dut * np.exp(-2j * beta * length)  # propagation aller-retour
    np.testing.assert_allclose(shift_reference_plane(F, g_meas, length, er), g_dut, atol=1e-12)


def test_three_reference_exact_with_error_box():
    eps_true = 25.0 - 8.0j * np.linspace(1, 0.3, len(F))
    air = measure(np.ones(len(F)))
    short = error_box(-np.ones(len(F), dtype=complex))
    ref_eps = water_25c(F)
    ref = measure(ref_eps)
    out = eps_three_reference(measure(eps_true), air, short, ref, ref_eps)
    np.testing.assert_allclose(out, eps_true, rtol=1e-9)


def test_three_reference_works_on_calibrated_data_too():
    eps_true = 4.0 - 0.2j
    out = eps_three_reference(
        probe_gamma(eps_true),
        probe_gamma(1.0),
        -np.ones(len(F), dtype=complex),
        probe_gamma(water_25c(F)),
        water_25c(F),
    )
    np.testing.assert_allclose(out, eps_true, rtol=1e-9)


def test_three_reference_grid_mismatch():
    a = np.ones(5, complex)
    with pytest.raises(ValueError):
        eps_three_reference(a, a, a, np.ones(4, complex), 1)


def test_capacitive_roundtrip():
    eps_true = 10.0 - 3.0j
    out = eps_capacitive(F, probe_gamma(eps_true), C0)
    np.testing.assert_allclose(out, eps_true, rtol=1e-9)
    with pytest.raises(ValueError):
        eps_capacitive(F, probe_gamma(1), 0)


def test_water_and_sign_convention():
    w = water_25c(np.array([1e6, 2.45e9]))
    assert w[0].real == pytest.approx(78.4, abs=0.1)
    e1, e2, tan = to_eps_columns(w)
    assert np.all(e2 > 0) and np.all(tan > 0)
    # relaxation de l'eau ~ 18-19 GHz : pertes déjà notables à 2.45 GHz
    assert e1[1] < e1[0]


def test_debye_limits():
    d = debye(np.array([0.0, 1e15]), 10, 2, 1e-11)
    assert d[0] == pytest.approx(10)
    assert d[1].real == pytest.approx(2, abs=0.01)


def test_smooth_reduces_noise_and_clamps_window():
    rng = np.random.default_rng(0)
    clean = np.exp(1j * np.linspace(0, 3, 200))
    noisy = clean + 0.05 * (rng.standard_normal(200) + 1j * rng.standard_normal(200))
    sm = smooth_complex(noisy, 15)
    assert np.abs(sm - clean).mean() < np.abs(noisy - clean).mean() / 2
    assert smooth_complex(clean[:4], 15).shape == (4,)
    np.testing.assert_array_equal(smooth_complex(clean, 1), clean)


def test_average_sweeps():
    a = np.array([[1 + 1j, 2], [3 + 1j, 4]], dtype=complex)
    np.testing.assert_allclose(average_sweeps(a), [2 + 1j, 3])
    with pytest.raises(ValueError):
        average_sweeps(np.ones(3))
