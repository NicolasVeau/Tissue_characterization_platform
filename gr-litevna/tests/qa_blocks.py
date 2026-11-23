"""Tests des blocs gr-litevna sur le simulateur (aucun matériel).

Lancer : pytest gr-litevna/tests/qa_blocks.py   (avec le Python de GNU Radio)
"""

import time

import gr_litevna
import numpy as np
import pmt
import pytest
from gnuradio import blocks, gr
from gr_litevna import common

from vna_app.calibration import Calibration, CalibrationError
from vna_app.driver.sim import ResonatorDUT
from vna_app.processing.record import SweepRecorder, load_recording

SIM = "sim:rate=20000"  # assez rapide pour les tests, assez lent pour ne pas écarter de sweeps
N = 51
F0, F1 = 1e9, 2e9
FREQS = F0 + np.arange(N) * ((F1 - F0) // (N - 1))


def tag_values(sink, key):
    return [pmt.to_python(t.value) for t in sink.tags() if pmt.symbol_to_string(t.key) == key]


def run_source(src, n_items, vlen=N):
    tb = gr.top_block()
    head = blocks.head(gr.sizeof_gr_complex * vlen, n_items)
    s11 = blocks.vector_sink_c(vlen)
    s21 = blocks.vector_sink_c(vlen)
    tb.connect((src, 0), head, s11)
    tb.connect((src, 1), s21)
    tb.run()
    a = np.array(s11.data()).reshape(-1, vlen)
    b = np.array(s21.data()).reshape(-1, vlen)[: len(a)]
    return a, b, s11


def test_sweep_source_data_and_tags():
    src = gr_litevna.sweep_source(SIM, F0, F1, N, 1, "")
    a, b, sink = run_source(src, 5)
    assert a.shape == (5, N)
    s11, s21 = ResonatorDUT()(FREQS, 0.0)
    assert np.max(np.abs(a[0] - s11)) < 1e-4 and np.max(np.abs(b[0] - s21)) < 1e-4
    idx = tag_values(sink, "sweep_idx")[:5]
    assert idx == list(range(idx[0], idx[0] + 5))  # consécutifs, sans trou
    assert tag_values(sink, "f_start")[0] == F0
    assert tag_values(sink, "f_step")[0] == (F1 - F0) // (N - 1)
    assert tag_values(sink, "calibrated")[0] is False
    sec, frac = tag_values(sink, "rx_time")[0]
    assert abs(sec + frac - time.time()) < 60


def test_sweep_source_releases_device_on_stop():
    src = gr_litevna.sweep_source(SIM, F0, F1, N, 1, "")
    run_source(src, 2)
    assert src._vna is None  # port libéré, thread de streaming arrêté


def test_sweep_source_applies_calibration(tmp_path):
    one = np.ones(N, dtype=complex)
    cal = Calibration(FREQS, 0 * one, one, 0 * one, thru=2 * one)  # S11 inchangé, S21 / 2
    path = cal.save(tmp_path / "t.cal")
    a, b, sink = run_source(gr_litevna.sweep_source(SIM, F0, F1, N, 1, str(path)), 2)
    s11, s21 = ResonatorDUT()(FREQS, 0.0)
    assert np.max(np.abs(a[0] - s11)) < 1e-4
    assert np.max(np.abs(b[0] - s21 / 2)) < 1e-4
    assert tag_values(sink, "calibrated")[0] is True


def test_sweep_source_refuses_mismatched_calibration(tmp_path):
    one = np.ones(N, dtype=complex)
    path = Calibration(FREQS + 5e6, 0 * one, one, 0 * one).save(tmp_path / "autre.cal")
    with pytest.raises(CalibrationError):  # refus à la construction, pas dans start()
        gr_litevna.sweep_source(SIM, F0, F1, N, 1, str(path))


def test_sweep_source_missing_device_ends_flowgraph():
    src = gr_litevna.sweep_source("/dev/absent-litevna", F0, F1, N, 1, "")
    a, _, _ = run_source(src, 3)  # ne doit ni lever ni bloquer : le flowgraph se termine
    assert a.shape[0] == 0 and "impossible d'ouvrir" in src.error


def test_sweep_source_retune_while_running():
    src = gr_litevna.sweep_source(SIM, F0, F1, N, 1, "")
    tb = gr.top_block()
    sink = blocks.vector_sink_c(N)
    tb.connect((src, 0), sink)
    tb.connect((src, 1), blocks.null_sink(gr.sizeof_gr_complex * N))
    tb.start()
    time.sleep(0.3)
    src.set_f_start(1.2e9)
    time.sleep(0.5)
    tb.stop()
    tb.wait()
    starts = tag_values(sink, "f_start")
    assert starts[0] == F0 and starts[-1] == 1.2e9
    assert src.f_start == 1.2e9


def test_sweep_source_publishes_pdu():
    src = gr_litevna.sweep_source(SIM, F0, F1, N, 1, "")
    tb = gr.top_block()
    dbg = blocks.message_debug()
    head = blocks.head(gr.sizeof_gr_complex * N, 3)
    tb.connect((src, 0), head, blocks.null_sink(gr.sizeof_gr_complex * N))
    tb.connect((src, 1), blocks.null_sink(gr.sizeof_gr_complex * N))
    tb.msg_connect(src, "sweep", dbg, "store")
    tb.start()
    time.sleep(0.5)
    tb.stop()
    tb.wait()
    assert dbg.num_messages() >= 3
    msg = dbg.get_message(0)
    meta, data = pmt.to_python(pmt.car(msg)), pmt.c32vector_elements(pmt.cdr(msg))
    assert meta["points"] == N and len(data) == 2 * N and meta["f_start"] == F0


def test_cw_source_fixed_frequency():
    src = gr_litevna.cw_source(SIM, 1.5e9, 64, 1)
    tb = gr.top_block()
    head = blocks.head(gr.sizeof_gr_complex, 200)
    s11, s21 = blocks.vector_sink_c(), blocks.vector_sink_c()
    tb.connect((src, 0), head, s11)
    tb.connect((src, 1), s21)
    tb.run()
    x = np.array(s11.data())
    assert len(x) == 200
    exp = ResonatorDUT()(np.array([1.5e9]), 0.0)[0][0]
    assert np.max(np.abs(x - exp)) < 1e-4  # DUT statique : valeur constante à 1,5 GHz
    assert tag_values(s11, "freq")[0] == 1.5e9
    offs = [t.offset for t in s11.tags() if pmt.symbol_to_string(t.key) == "sweep_idx"]
    assert offs[:3] == [0, 64, 128]  # un tag par acquisition de block_len échantillons


def test_sink_then_replay_roundtrip(tmp_path):
    path = tmp_path / "gr.npz"
    src = gr_litevna.sweep_source("sim:rate=20000,noise=1e-3", F0, F1, N, 1, "")
    sink = gr_litevna.sweep_sink(str(path), N, 0.0, 0.0)
    tb = gr.top_block()
    h0 = blocks.head(gr.sizeof_gr_complex * N, 6)
    h1 = blocks.head(gr.sizeof_gr_complex * N, 6)
    live = blocks.vector_sink_c(N)
    tb.connect((src, 0), h0, (sink, 0))
    tb.connect((src, 1), h1, (sink, 1))
    tb.connect(h0, live)
    tb.run()
    rec = load_recording(path)
    assert len(rec) == 6 and rec.points == N and rec.lost() == 0
    assert rec.freqs[0] == F0 and rec.freqs[1] - rec.freqs[0] == (F1 - F0) // (N - 1)
    assert rec.meta["calibrated"] is False and rec.meta["precision"] == "complex64"
    assert np.array_equal(rec.s[:, :, 0].astype(np.complex64).ravel(), np.array(live.data()))

    rp = gr_litevna.replay_source(str(path), N, "", False, False, 1.0)
    a, b, out = run_source(rp, 6)
    assert np.array_equal(a, rec.s[:, :, 0].astype(np.complex64))  # relecture bit à bit
    assert np.array_equal(b, rec.s[:, :, 1].astype(np.complex64))
    assert tag_values(out, "sweep_idx") == [int(x) for x in rec.seq]
    assert tag_values(out, "f_start")[0] == F0


def test_replay_stops_at_end_and_repeats(tmp_path):
    path = tmp_path / "r.npz"
    with SweepRecorder(path) as rec:
        for i in range(3):
            rec.add(FREQS, np.full((N, 2), i + 1j), seq=i, timestamp=100.0 + i)
    a, _, _ = run_source(gr_litevna.replay_source(str(path), N, "", False, False, 1.0), 10)
    assert a.shape[0] == 3  # fin de fichier : le flowgraph s'arrête
    a, _, sink = run_source(gr_litevna.replay_source(str(path), N, "", True, False, 1.0), 7)
    assert [int(x.real) for x in a[:, 0]] == [0, 1, 2, 0, 1, 2, 0]
    assert tag_values(sink, "sweep_idx")[:7] == list(range(7))


def test_replay_realtime_follows_timestamps(tmp_path):
    path = tmp_path / "rt.npz"
    with SweepRecorder(path) as rec:
        for i in range(4):
            rec.add(FREQS, np.zeros((N, 2)), seq=i, timestamp=10.0 + 0.2 * i)
    t0 = time.monotonic()
    run_source(gr_litevna.replay_source(str(path), N, "", False, True, 2.0), 4)
    assert 0.25 < time.monotonic() - t0 < 1.5  # 0,6 s enregistrées, rejouées à vitesse 2


def test_replay_refuses_wrong_point_count(tmp_path):
    path = tmp_path / "p.npz"
    with SweepRecorder(path) as rec:
        rec.add(FREQS, np.zeros((N, 2)))
    with pytest.raises(ValueError):
        gr_litevna.replay_source(str(path), N + 1, "", False, False, 1.0)


def test_rx_time_roundtrip():
    assert abs(common.rx_time_float(common.rx_time_pmt(1795283132.25)) - 1795283132.25) < 1e-6
