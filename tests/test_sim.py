"""Simulateur : la chaîne réelle du driver (sweep, streaming) sur le faux appareil."""

import numpy as np
import pytest

from vna_app.calibration import Calibration
from vna_app.driver.litevna import LiteVNA, SweepResult
from vna_app.driver.sim import (
    ResonatorDUT,
    SimLiteVNA,
    SimSerial,
    open_device,
    parse_sim_port,
)


def test_open_device_dispatch():
    assert isinstance(open_device("sim"), SimLiteVNA)
    assert isinstance(open_device("sim:noise=1e-3"), SimLiteVNA)
    real = open_device("/dev/litevna")
    assert type(real) is LiteVNA


def test_parse_sim_port():
    assert parse_sim_port("sim")["rate"] == 300.0
    o = parse_sim_port("sim:rate=300, noise=1e-3,seed=4")
    assert (o["rate"], o["noise"], o["seed"]) == (300.0, 1e-3, 4.0)
    with pytest.raises(ValueError):
        parse_sim_port("sim:bogus=1")


def test_info_and_sweep_match_dut():
    dut = ResonatorDUT(f0=1.5e9, q=30)
    with SimLiteVNA(SimSerial(dut=dut)) as vna:
        assert vna.device_info()["firmware"] == "2.2"
        freqs = vna.configure_sweep(400e6, 3e9, 201)
        f, s = vna.sweep()
    assert np.array_equal(f, freqs) and s.shape == (201, 2)
    s11, s21 = dut(freqs, 0.0)
    assert np.max(np.abs(s[:, 0] - s11)) < 1e-5  # quantification entière seulement
    assert np.max(np.abs(s[:, 1] - s21)) < 1e-5
    assert abs(f[np.argmin(np.abs(s[:, 0]))] - 1.5e9) <= 13e6  # creux à f0, au pas près


def test_averaging_reduces_noise():
    def run(avg):
        with SimLiteVNA(SimSerial(noise=1e-2, seed=1)) as vna:
            vna.configure_sweep(1e9, 2e9, 101, avg)
            _, s = vna.sweep()
        with SimLiteVNA(SimSerial()) as ref:
            ref.configure_sweep(1e9, 2e9, 101)
            _, s0 = ref.sweep()
        return np.std(s[:, 0] - s0[:, 0])

    assert run(16) < run(1) / 2.5


def test_fixed_frequency_is_a_time_series():
    """start = stop : pas nul, N échantillons à la même fréquence (mode CW)."""
    dut = ResonatorDUT(f0=1.0e9, drift_hz_s=20e6)
    with SimLiteVNA(SimSerial(dut=dut)) as vna:
        freqs = vna.configure_sweep(1.0e9, 1.0e9, 300)
        assert vna.step_hz == 0 and np.all(freqs == 1.0e9)
        _, s = vna.sweep()
    assert s.shape == (300, 2)
    # le DUT dérive pendant l'acquisition (1 s simulée) : |S11| remonte en sortant du creux
    assert abs(s[-1, 0]) > abs(s[0, 0]) + 0.05


def test_streaming_delivers_consecutive_sweeps():
    # débit limité : sans limite, le producteur dépasse le consommateur et écarte des sweeps
    with SimLiteVNA(SimSerial(dut=ResonatorDUT(drift_hz_s=50e6), rate=20000)) as vna:
        vna.configure_sweep(1e9, 2e9, 51)
        q = vna.start_streaming(maxsize=64)
        got = [q.get(timeout=5) for _ in range(4)]
        vna.stop_streaming()
    assert all(isinstance(r, SweepResult) for r in got)
    assert [r.seq for r in got] == [0, 1, 2, 3]
    assert not np.allclose(got[0].s_raw, got[3].s_raw)  # le DUT dérive entre les sweeps


def test_error_terms_are_removed_by_calibration():
    """Les erreurs systématiques du simulateur sont corrigées par la calibration OSL."""
    errors = (0.05 + 0.02j, 0.9 - 0.1j, 0.1 + 0.05j)

    class Const:
        def __init__(self, g):
            self.g = g

        def __call__(self, freqs, t):
            return np.full(len(freqs), self.g, dtype=complex), np.ones(len(freqs), dtype=complex)

    def measure(dut):
        with SimLiteVNA(SimSerial(dut=dut, errors=errors)) as vna:
            f = vna.configure_sweep(1e9, 2e9, 21)
            return f, vna.sweep()[1]

    f, o = measure(Const(1.0))
    _, s = measure(Const(-1.0))
    _, ld = measure(Const(0.0))
    cal = Calibration.from_measurements(f, o[:, 0], s[:, 0], ld[:, 0])
    dut = ResonatorDUT(f0=1.5e9)
    _, raw = measure(dut)
    assert np.max(np.abs(raw[:, 0] - dut(f, 0.0)[0])) > 0.05  # brut : faux
    assert np.max(np.abs(cal.apply(f, raw)[:, 0] - dut(f, 0.0)[0])) < 1e-4  # calibré : juste


def test_rate_limits_throughput():
    import time

    with SimLiteVNA(SimSerial(rate=2000)) as vna:
        vna.configure_sweep(1e9, 2e9, 200)
        t0 = time.monotonic()
        vna.sweep()
        dt = time.monotonic() - t0
    assert 0.07 < dt < 1.0  # 200 entrées à 2000/s ≈ 0,1 s
