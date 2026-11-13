"""Test de la fenêtre hors écran (QT_QPA_PLATFORM=offscreen), sans VNA."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from vna_app.gui.main_window import MainWindow  # noqa: E402

F = np.linspace(400e6, 3e9, 31)
W = 2 * np.pi * F
C0 = 80e-15
PARAMS = {"start": 400e6, "stop": 3e9, "points": 31, "avg": 1, "step_hz": 86666666}


def probe(eps):
    y = 1j * W * C0 * eps
    return (1 - 50 * y) / (1 + 50 * y)


def box(g):
    return 0.06 * np.exp(1j * F / 1e9) + 0.9 * np.exp(-1j * F / 4e8) * g / (
        1 - 0.1 * np.exp(-1j * F / 2e9) * g
    )


def sraw(g):
    return np.stack([box(g), np.full(len(F), 0.01 + 0j)], axis=1)


def one(v):
    return np.full(len(F), v, dtype=complex)


@pytest.fixture
def win(monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = MainWindow("/dev/inexistant")
    # pas de matériel : on neutralise l'envoi de commandes au worker
    monkeypatch.setattr(w, "start", lambda continuous: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    yield w
    w.close()
    app.processEvents()


def feed(w, g):
    w.on_sweep(F, sraw(g), PARAMS)


def test_raw_display(win):
    feed(win, one(0.2))
    assert "brut" in win.plot11.getPlotItem().titleLabel.text
    assert len(win.c11.getData()[0]) == len(F)
    assert not win.plot_eps.isVisibleTo(win)


def test_capture_standards_calibrate_and_display(win, tmp_path, monkeypatch):
    for name, g in (("open", 1), ("short", -1), ("load", 0)):
        win.pending = name
        feed(win, one(g))
    assert "Open" in win.cal_captured.text() and "Load" in win.cal_captured.text()
    cal = tmp_path / "b.cal"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(cal), ""))
    win.build_cal()
    assert cal.exists() and win.chk_cal.isChecked()
    g = probe(one(6 - 1j))
    feed(win, g)
    assert "calibré" in win.plot11.getPlotItem().titleLabel.text
    np.testing.assert_allclose(win.result.s[:, 0], g, atol=1e-9)
    # décocher => brut
    win.chk_cal.setChecked(False)
    assert "brut" in win.plot11.getPlotItem().titleLabel.text
    win.chk_cal.setChecked(True)
    # sauvegarde .s2p : métadonnées de calibration
    out = tmp_path / "m.s2p"
    win.info = {"firmware": "2.2", "hardware_revision": 2}
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.save()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["calibrated"] is True and meta["calibration_file"] == str(cal)


def test_build_cal_missing_standard(win):
    win.pending = "open"
    feed(win, one(1))
    win.build_cal()
    assert "short" in win.status.text()


def test_eps_capacitive_curves_and_csv(win, tmp_path, monkeypatch):
    for name, g in (("open", 1), ("short", -1), ("load", 0)):
        win.pending = name
        feed(win, one(g))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    win.build_cal()
    win.eps_mode.setCurrentIndex(1)  # capacitif
    win.c0_ff.setValue(80.0)
    feed(win, probe(one(9 - 2j)))
    assert win.plot_eps.isVisibleTo(win)
    x, y = win.ceps1.getData()
    np.testing.assert_allclose(y, 9.0, rtol=1e-6)
    np.testing.assert_allclose(win.ceps2.getData()[1], 2.0, rtol=1e-6)
    out = tmp_path / "e.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.save_eps()
    d = np.loadtxt(out, delimiter=",", skiprows=1)
    np.testing.assert_allclose(d[:, 1], 9.0, rtol=1e-6)


def test_eps_three_ref_without_refs_reports(win):
    win.eps_mode.setCurrentIndex(2)
    feed(win, one(0.3))
    assert "référence manquante" in win.status.text()
    assert win.ceps1.getData()[0] is None or len(win.ceps1.getData()[0]) == 0
