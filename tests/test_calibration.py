import numpy as np
import pytest

from vna_app.calibration import Calibration, CalibrationError, IdealKit

F = np.linspace(400e6, 3e9, 51)


def error_box(f):
    e00 = 0.08 * np.exp(1j * f / 1e9)  # directivité
    e11 = 0.12 * np.exp(-1j * f / 2e9)  # adaptation source
    e01e10 = 0.9 * np.exp(-1j * f / 3e8 * 0.2)  # suivi de réflexion
    return e00, e11, e01e10


def measure(gamma, f=F):
    e00, e11, e0110 = error_box(f)
    return e00 + e0110 * gamma / (1 - e11 * gamma)


def std():
    one = np.ones_like(F, dtype=complex)
    return measure(one), measure(-one), measure(0 * one)


def test_recovers_dut_exactly():
    o, s, ld = std()
    cal = Calibration.from_measurements(F, o, s, ld)
    gamma = 0.3 * np.exp(1j * np.linspace(0, 6, len(F)))
    raw = np.stack([measure(gamma), np.ones(len(F), complex)], axis=1)
    out = cal.apply(F, raw)
    np.testing.assert_allclose(out[:, 0], gamma, atol=1e-10)


def test_standards_map_to_ideal():
    o, s, ld = std()
    cal = Calibration.from_measurements(F, o, s, ld)
    z = np.zeros(len(F), complex)
    out = cal.apply(F, np.stack([o, z], 1))[:, 0]
    np.testing.assert_allclose(out, 1, atol=1e-10)
    out = cal.apply(F, np.stack([ld, z], 1))[:, 0]
    np.testing.assert_allclose(out, 0, atol=1e-10)


def test_matches_skrf_oneport():
    import skrf
    from skrf.calibration import OnePort

    o, s, ld = std()
    freq = skrf.Frequency.from_f(F, unit="hz")

    def net(v):
        return skrf.Network(frequency=freq, s=v.reshape(-1, 1, 1), z0=50)

    ideal = [
        net(np.ones(len(F), complex)),
        net(-np.ones(len(F), complex)),
        net(np.zeros(len(F), complex)),
    ]
    ref = OnePort(measured=[net(o), net(s), net(ld)], ideals=ideal)
    gamma = 0.4 * np.exp(1j * np.linspace(0, 3, len(F)))
    expected = ref.apply_cal(net(measure(gamma))).s[:, 0, 0]
    cal = Calibration.from_measurements(F, o, s, ld)
    out = cal.apply(F, np.stack([measure(gamma), np.ones(len(F), complex)], 1))
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-8)


def test_thru_normalises_s21():
    o, s, ld = std()
    thru = 0.01 * np.exp(1j * F / 1e9)
    cal = Calibration.from_measurements(F, o, s, ld, thru_s21=thru)
    dut = 0.5 * np.exp(-1j * F / 5e8)
    out = cal.apply(F, np.stack([measure(0 * dut), dut * thru], 1))
    np.testing.assert_allclose(out[:, 1], dut, atol=1e-12)


def test_s21_untouched_without_thru():
    o, s, ld = std()
    cal = Calibration.from_measurements(F, o, s, ld)
    s21 = np.full(len(F), 0.2 + 0.1j)
    out = cal.apply(F, np.stack([ld, s21], 1))
    np.testing.assert_allclose(out[:, 1], s21)


def test_save_load_roundtrip(tmp_path):
    o, s, ld = std()
    cal = Calibration.from_measurements(F, o, s, ld, thru_s21=np.full(len(F), 0.02 + 0j))
    p = cal.save(tmp_path / "x.cal")
    assert p.name == "x.cal"
    back = Calibration.load(p)
    np.testing.assert_array_equal(back.a, cal.a)
    np.testing.assert_array_equal(back.thru, cal.thru)
    assert back.meta["kit"] == "ideal"


def test_grid_mismatch_refused():
    o, s, ld = std()
    cal = Calibration.from_measurements(F, o, s, ld)
    with pytest.raises(CalibrationError, match="grille"):
        cal.apply(F + 1e6, np.ones((len(F), 2), complex))
    with pytest.raises(CalibrationError):
        cal.apply(F[:-1], np.ones((len(F) - 1, 2), complex))


def test_degenerate_standards_refused():
    o, _, _ = std()
    with pytest.raises(CalibrationError):
        Calibration.from_measurements(F, o, o, o)


def test_zero_thru_refused():
    o, s, ld = std()
    with pytest.raises(CalibrationError):
        Calibration.from_measurements(F, o, s, ld, thru_s21=np.zeros(len(F), complex))


def test_bad_file(tmp_path):
    p = tmp_path / "bad.cal"
    p.write_bytes(b"not an npz")
    with pytest.raises(CalibrationError):
        Calibration.load(p)


def test_kit_is_pluggable():
    """Avec un open caractérisé, l'open mesuré doit être ramené à la valeur du kit."""
    f = F

    class Kit2(IdealKit):
        name = "kit2"

        def open(self, freqs):
            return 0.98 * np.exp(-1j * freqs / 1e10)

    g_open = Kit2().open(f)
    o = measure(g_open)
    s = measure(-np.ones(len(f), complex))
    ld = measure(np.zeros(len(f), complex))
    cal = Calibration.from_measurements(f, o, s, ld, kit=Kit2())
    assert cal.meta["kit"] == "kit2"
    out = cal.apply(f, np.stack([o, np.zeros(len(f), complex)], 1))[:, 0]
    np.testing.assert_allclose(out, g_open, atol=1e-10)
