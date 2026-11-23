import numpy as np

from vna_app.processing import pressure as P

F = np.linspace(400e6, 3e9, 101)


def tissue(f=F):
    w = 2 * np.pi * f
    return 8 + 40 / (1 + 1j * w * 1.5e-10) - 1j * 0.8 / (w * 8.85e-12)


def contact(eps, a):
    return a * eps + (1 - a)


def test_invariants_to_contact_fraction():
    e = tissue()
    for a in (0.3, 0.6, 1.0):
        m = contact(e, a)
        np.testing.assert_allclose(P.normalize_shape(F, m, 1e9), P.normalize_shape(F, e, 1e9))
        np.testing.assert_allclose(P.loss_ratio(m), P.loss_ratio(e))
    assert P.contact_level(contact(e, 0.3)) < P.contact_level(contact(e, 1.0))
    np.testing.assert_allclose(
        P.contact_level(contact(e, 0.5)) / P.contact_level(e), 0.5, rtol=1e-12
    )


def test_features_pressure_invariant_and_fisher():
    rng = np.random.default_rng(0)
    zones = ("a", "b", "c")
    eps_z = {z: tissue() * (1 + 0.15 * i) for i, z in enumerate(zones)}
    X_raw, X_inv, y = [], [], []
    for z in zones:
        for _ in range(20):
            m = contact(eps_z[z], rng.uniform(0.3, 1.0))
            m = m * (1 + 0.002 * rng.standard_normal(len(F)))
            X_raw.append(np.r_[m.real[::10], -m.imag[::10]])
            X_inv.append(P.band_features(F, m, 1e9))
            y.append(z)
    # même séparation intrinsèque : les descripteurs invariants séparent bien mieux
    # que ε', ε'' bruts quand la pression varie
    f_raw = np.nanmedian(P.fisher_ratio(np.array(X_raw), np.array(y)))
    f_inv = np.nanmax(P.fisher_ratio(np.array(X_inv), np.array(y)))
    assert f_inv > 10 * f_raw


def test_normalize_raises_without_contact():
    import pytest

    with pytest.raises(ValueError):
        P.normalize_shape(F, np.ones(len(F), dtype=complex), 1e9)
