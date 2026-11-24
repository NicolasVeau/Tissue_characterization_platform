import numpy as np
import pytest

from vna_app.calibration import CalibrationError
from vna_app.processing.permittivity import water_25c
from vna_app.processing.pipeline import Pipeline

F = np.linspace(400e6, 3e9, 41)
W = 2 * np.pi * F
C0 = 0.1e-12


def probe(eps):
    y = 1j * W * C0 * eps
    return (1 - 50 * y) / (1 + 50 * y)


def box(g):
    e00 = 0.06 * np.exp(1j * F / 1e9)
    e11 = 0.1 * np.exp(-1j * F / 2e9)
    e0110 = 0.9 * np.exp(-1j * F / 4e8)
    return e00 + e0110 * g / (1 - e11 * g)


def sraw(g):
    return np.stack([box(g), np.full(len(F), 0.01 + 0j)], axis=1)


def one(v):
    return np.full(len(F), v, dtype=complex)


def cal_pipeline():
    p = Pipeline()
    p.capture("open", F, sraw(one(1)))
    p.capture("short", F, sraw(one(-1)))
    p.capture("load", F, sraw(one(0)))
    return p


def test_build_cal_and_process():
    p = cal_pipeline()
    p.build_cal()
    g = probe(one(5 - 1j))
    r = p.process(F, sraw(g))
    assert r.calibrated and not r.notes
    np.testing.assert_allclose(r.s[:, 0], g, atol=1e-10)


def test_apply_cal_off_gives_raw():
    p = cal_pipeline()
    p.build_cal()
    p.apply_cal = False
    r = p.process(F, sraw(one(0.3)))
    assert not r.calibrated
    np.testing.assert_allclose(r.s, sraw(one(0.3)))


def test_missing_standard():
    p = Pipeline()
    p.capture("open", F, sraw(one(1)))
    with pytest.raises(CalibrationError, match="short"):
        p.build_cal()


def test_cal_grid_mismatch_falls_back_to_raw():
    p = cal_pipeline()
    p.build_cal()
    r = p.process(F[:-1], sraw(one(0.1))[:-1])
    assert not r.calibrated and r.notes and "calibration non appliquée" in r.notes[0]


def test_eps_three_ref_on_raw():
    p = Pipeline()
    p.eps_mode = "three_ref"
    p.capture("air", F, sraw(probe(one(1))))
    p.capture("probe_short", F, sraw(one(-1)))
    p.capture("ref", F, sraw(probe(water_25c(F))))
    eps_true = 15 - 4j
    r = p.process(F, sraw(probe(one(eps_true))))
    assert not r.notes
    np.testing.assert_allclose(r.eps, eps_true, rtol=1e-9)


def test_eps_three_ref_missing_ref():
    p = Pipeline()
    p.eps_mode = "three_ref"
    r = p.process(F, sraw(one(0.1)))
    assert r.eps is None and "référence manquante" in r.notes[0]


def test_eps_capacitive_calibrated():
    p = cal_pipeline()
    p.build_cal()
    p.eps_mode, p.c0 = "capacitive", C0
    r = p.process(F, sraw(probe(one(9 - 2j))))
    np.testing.assert_allclose(r.eps, 9 - 2j, rtol=1e-8)
    assert not r.notes


def test_eps_capacitive_uncalibrated_warns():
    p = Pipeline()
    p.eps_mode, p.c0 = "capacitive", C0
    r = p.process(F, sraw(one(0.1)))
    assert r.eps is not None and "non calibré" in r.notes[0]


def test_unknown_standard():
    with pytest.raises(ValueError):
        Pipeline().capture("nope", F, sraw(one(0)))


def test_cal_save_load(tmp_path):
    p = cal_pipeline()
    p.build_cal()
    p.save_cal(tmp_path / "a.cal")
    q = Pipeline()
    q.load_cal(tmp_path / "a.cal")
    g = probe(one(3))
    np.testing.assert_allclose(q.process(F, sraw(g)).s[:, 0], g, atol=1e-10)
    with pytest.raises(CalibrationError):
        Pipeline().save_cal(tmp_path / "b.cal")


def test_refs_save_load_roundtrip(tmp_path):
    """Les références ε* sauvées évitent de les recapturer : ε* identique après reload."""
    p = Pipeline()
    p.eps_mode = "three_ref"
    p.ref_liquid = "water"
    p.capture("air", F, sraw(probe(one(1))))
    p.capture("probe_short", F, sraw(one(-1)))
    p.capture("ref", F, sraw(probe(water_25c(F))))
    path = tmp_path / "refs.npz"
    p.save_refs(path)

    q = Pipeline()
    q.eps_mode = "three_ref"
    names = q.load_refs(path)
    assert set(names) == {"air", "probe_short", "ref"}
    assert q.ref_liquid == "water"
    eps_true = 15 - 4j
    r = q.process(F, sraw(probe(one(eps_true))))
    assert not r.notes
    np.testing.assert_allclose(r.eps, eps_true, rtol=1e-9)


def test_refs_save_empty_fails(tmp_path):
    with pytest.raises(CalibrationError, match="aucune référence"):
        Pipeline().save_refs(tmp_path / "x.npz")


def test_refs_load_bad_file(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"pas un npz")
    with pytest.raises(CalibrationError, match="illisibles"):
        Pipeline().load_refs(bad)
