import numpy as np
import pytest

from vna_app.processing import classifier as C
from vna_app.processing import pressure as P

F = np.linspace(400e6, 3e9, 101)
W = 2 * np.pi * F


def tissue(tau, de, einf=8.0, sigma=0.8):
    return einf + de / (1 + 1j * W * tau) - 1j * sigma / (W * 8.85e-12)


ZONES = {
    "nez": tissue(1.2e-10, 30),
    "joue": tissue(1.5e-10, 42, sigma=1.0),
    "doigt": tissue(2.0e-10, 35, einf=6, sigma=0.5),
    "cuisse": tissue(1.0e-10, 20, einf=5, sigma=0.3),
}


def make_dataset(n_sessions=8, n_sweeps=5, seed=0):
    rng = np.random.default_rng(seed)
    ds = C.Dataset({"f_ref": 1e9, "grid": C.grid_signature(F)})
    for z, e in ZONES.items():
        for _ in range(n_sessions):
            sess = ds.new_session(z)
            a = rng.uniform(0.3, 1.0)  # pression propre à la pose
            for _ in range(n_sweeps):
                m = (a * e + (1 - a)) * (1 + 0.003 * rng.standard_normal(len(F)))
                ds.add(P.band_features(F, m, 1e9), z, sess)
    return ds


def test_cross_validation_by_session_is_accurate_despite_pressure():
    X, z, s = make_dataset().arrays()
    res = C.cross_validate(X, z, s)
    assert res["n_sessions"] == 32
    assert res["acc_sessions"] > 0.95
    assert res["d2_max"] is not None
    assert "Exactitude par session" in C.format_report(res, 0.2)


def test_unknown_rejected_and_known_accepted():
    X, z, s = make_dataset().arrays()
    res = C.cross_validate(X, z, s)
    m = C.ZoneClassifier(0.2).fit(X, z)
    m.d2_max = res["d2_max"]
    known = P.band_features(F, 0.6 * ZONES["joue"] + 0.4, 1e9)
    best, proba, _ = m.predict_one(known)
    assert best == "joue" and abs(sum(proba.values()) - 1) < 1e-9
    other = P.band_features(F, tissue(5e-10, 60, einf=30, sigma=3.0), 1e9)
    assert m.predict_one(other)[0] is None


def test_dataset_roundtrip(tmp_path):
    ds = make_dataset(2, 2)
    ds.save(tmp_path / "d.npz")
    d2 = C.Dataset.load(tmp_path / "d.npz")
    assert len(d2) == len(ds) and d2.meta["f_ref"] == 1e9
    assert d2.sessions == ds.sessions and d2.n_sessions("nez") == 2
    np.testing.assert_allclose(d2.arrays()[0], ds.arrays()[0])


def test_fit_validation():
    with pytest.raises(ValueError):
        C.ZoneClassifier().fit(np.zeros((4, 3)), np.array(["a"] * 4))
    with pytest.raises(ValueError):
        C.ZoneClassifier(2.0)


def test_features_for_none_without_contact():
    assert C.features_for(F, None, 1e9) is None
    assert C.features_for(F, np.ones(len(F), dtype=complex), 1e9) is None
