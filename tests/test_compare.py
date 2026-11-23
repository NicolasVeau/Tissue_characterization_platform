import numpy as np

from vna_app.processing.compare import compare, snapshot, unwrap_deg

F = np.linspace(400e6, 3e9, 31)


def gam(z):
    return (z - 50) / (z + 50)


def test_diff_known_ratio():
    z1 = 100 * np.exp(1j * np.linspace(0, 1, F.size))
    k = 2.0 * np.exp(1j * np.deg2rad(30))
    a = snapshot(F, gam(z1), None)
    b = snapshot(F, gam(z1 * k), None)
    d = compare(a, b)
    assert np.allclose(d["d_mod_db"], 20 * np.log10(2.0))
    assert np.allclose(d["d_arg_deg"], 30.0)
    assert d["d_eps_re"] is None


def test_unwrap_no_jump_beyond_360():
    z = np.exp(1j * np.deg2rad(np.linspace(0, 720, F.size)))
    out = unwrap_deg(z)
    assert abs(out[-1] - out[0] - 720) < 1e-6


def test_eps_diff_and_other_grid():
    eps1 = (10 - 2j) * np.ones(F.size)
    eps2 = (12 - 5j) * np.ones(F.size)
    z = 80 * np.ones(F.size, dtype=complex)
    a = snapshot(F, gam(z), eps1)
    f2 = np.linspace(500e6, 2e9, 11)  # grille et plage différentes
    b = snapshot(f2, gam(z[: f2.size]), eps2[: f2.size])
    d = compare(a, b)
    inside = (F >= 500e6) & (F <= 2e9)
    assert np.allclose(d["d_eps_re"][inside], 2.0)
    assert np.allclose(d["d_eps_im"][inside], 3.0)  # ε'' = -Im
    assert np.all(np.isnan(d["d_eps_re"][~inside]))
