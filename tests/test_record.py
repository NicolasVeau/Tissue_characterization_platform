"""Enregistrement / relecture de séries de sweeps, et sous-commande CLI `record`."""

import numpy as np
import pytest

from vna_app import cli
from vna_app.driver.litevna import SweepResult
from vna_app.driver.sim import SimLiteVNA, SimSerial
from vna_app.processing.record import RecordError, SweepRecorder, load_recording


def _results(n=5, points=11):
    freqs = np.linspace(1e9, 2e9, points)
    rng = np.random.default_rng(0)
    for i in range(n):
        s = rng.standard_normal((points, 2)) + 1j * rng.standard_normal((points, 2))
        yield SweepResult(i, 1000.0 + i, freqs, s)


def test_roundtrip_is_bit_exact(tmp_path):
    src = list(_results())
    path = tmp_path / "serie.npz"
    with SweepRecorder(path, {"averaging": 1}) as rec:
        for r in src:
            rec.append(r)
    back = load_recording(path)
    assert len(back) == 5 and back.points == 11 and back.lost() == 0
    assert back.meta["averaging"] == 1 and back.meta["sweeps"] == 5
    for a, b in zip(src, back.results()):
        assert a.seq == b.seq and a.timestamp == b.timestamp
        assert np.array_equal(a.freqs, b.freqs) and np.array_equal(a.s_raw, b.s_raw)


def test_lost_counts_seq_gaps(tmp_path):
    path = tmp_path / "trous.npz"
    with SweepRecorder(path) as rec:
        for r in _results(6):
            if r.seq not in (2, 3):
                rec.append(r)
    assert load_recording(path).lost() == 2


def test_grid_change_refused(tmp_path):
    rec = SweepRecorder(tmp_path / "x.npz")
    rec.add(np.linspace(1e9, 2e9, 11), np.zeros((11, 2)))
    with pytest.raises(RecordError):
        rec.add(np.linspace(1e9, 3e9, 11), np.zeros((11, 2)))
    with pytest.raises(RecordError):
        rec.add(np.linspace(1e9, 2e9, 11), np.zeros((11, 3)))


def test_empty_recording_writes_nothing(tmp_path):
    path = tmp_path / "vide.npz"
    assert SweepRecorder(path).close() is None
    assert not path.exists()


def test_unreadable_file(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"pas un npz")
    with pytest.raises(RecordError):
        load_recording(bad)


def test_record_from_streaming(tmp_path):
    path = tmp_path / "stream.npz"
    with SimLiteVNA(SimSerial(noise=1e-3, rate=20000)) as vna, SweepRecorder(path) as rec:
        vna.configure_sweep(1e9, 2e9, 21)
        q = vna.start_streaming(maxsize=64)
        live = [q.get(timeout=5) for _ in range(3)]
        vna.stop_streaming()
        for r in live:
            rec.append(r)
    back = load_recording(path)
    assert np.array_equal(back.s, np.stack([r.s_raw for r in live]))


def test_cli_record_with_simulator(tmp_path, capsys):
    out = tmp_path / "cli.npz"
    rc = cli.main(
        ["--port", "sim:rate=0", "record", "--start", "4e8", "--stop", "3e9", "--points", "51"]
        + ["--count", "4", "--out", str(out)]
    )
    assert rc == 0 and "4 sweep(s)" in capsys.readouterr().out
    rec = load_recording(out)
    assert len(rec) == 4 and rec.points == 51
    assert rec.meta["calibrated"] is False and rec.meta["firmware"] == "2.2"
    assert rec.freqs[0] == 4e8


def test_cli_record_requires_a_limit(tmp_path):
    rc = cli.main(
        ["--port", "sim", "record", "--start", "1e9", "--stop", "2e9", "--points", "11"]
        + ["--out", str(tmp_path / "x.npz")]
    )
    assert rc == 2
