"""Test de la fenêtre hors écran (QT_QPA_PLATFORM=offscreen), sans VNA."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["VNA_APP_NO_SETTINGS"] = "1"  # pas de QSettings réels pendant les tests

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from vna_app.gui import main_window as mw  # noqa: E402
from vna_app.gui.main_window import MainWindow  # noqa: E402
from vna_app.processing.permittivity import to_eps_columns  # noqa: E402

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
def win(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(mw, "LAST_CAL", tmp_path / "last_cal")
    monkeypatch.setattr(mw, "LAST_REFS", tmp_path / "last_refs")
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
    win.on_tab_changed(win.eps_tab_index)  # le tracé ε* vit dans la page ε*
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


def test_smith_follows_s11(win):
    feed(win, one(0.2))
    x, y = win.smith.trace.getData()
    assert len(x) == len(F)
    np.testing.assert_allclose(x + 1j * y, win.result.s[:, 0])


def test_comparison_tab(win):
    # rien à mémoriser avant un sweep
    win.store_memory(1)
    assert win.mem[1] is None
    feed(win, one(0.2))
    win.store_memory(1)
    feed(win, one(0.4))
    win.store_memory(2)
    assert win.mem[1] is not None and win.mem[2] is not None
    c = win.cmp_curves
    assert len(c["dmod"].getData()[0]) == len(F)
    assert len(c["darg"].getData()[0]) == len(F)
    assert np.all(np.isfinite(c["dmod"].getData()[1]))
    assert c["dere"].getData()[0] is None or len(c["dere"].getData()[0]) == 0  # ε* inactif
    win.on_tab_changed(win.cmp_tab_index)
    assert win.stack.currentIndex() == 1
    win.on_tab_changed(0)
    assert win.stack.currentIndex() == 0
    win.clear_memory(2)
    assert win.mem[2] is None
    assert c["dmod"].getData()[0] is None or len(c["dmod"].getData()[0]) == 0


def _eps_setup(win):
    """Méthode capacitive sur S11 : ε* défini sans références."""
    i = win.eps_mode.findData("capacitive")
    win.eps_mode.setCurrentIndex(i)


def test_eps_page_layout_and_tab_switch(win):
    win.on_tab_changed(win.eps_tab_index)
    assert win.stack.currentIndex() == 2
    page = win.stack.currentWidget()
    for w in (win.smith_eps, win.wf_iso, win.wf_tan, win.plot_eps):
        assert w.parentWidget() is page or page.isAncestorOf(w)
    assert not win.smith.isVisibleTo(win)  # le Smith de la page ε* remplace celui de droite
    win.on_tab_changed(0)
    assert win.stack.currentIndex() == 0 and win.smith.isVisibleTo(win)
    assert win.plot11.isVisibleTo(win)


def test_waterfalls_follow_sweeps(win):
    _eps_setup(win)
    for k in range(4):
        feed(win, one(0.2 + 0.05 * k))
    e1, e2, tan = to_eps_columns(win.result.eps)
    for wf in win.waterfalls:
        assert len(wf.rows) == 4
    np.testing.assert_allclose(win.wf_iso.rows[-1], e1)  # couleur = ε'
    np.testing.assert_allclose(win.wf_iso.aux_rows[-1], e2)  # isocontours = ε''
    np.testing.assert_allclose(win.wf_tan.rows[-1], tan)
    # le Smith de la page ε* suit S11 comme l'autre
    x, y = win.smith_eps.trace.getData()
    np.testing.assert_allclose(x + 1j * y, win.result.s[:, 0])


def test_hidden_waterfall_defers_render_but_keeps_levels(win):
    """Page ε* non affichée : pas d'isocontours construits, mais données et niveaux à jour."""
    _eps_setup(win)
    win.on_tab_changed(0)
    for k in range(6):
        feed(win, one(0.2 + 0.05 * k))
    wf = win.wf_iso
    assert len(wf.rows) == 6 and wf.levels[1] > wf.levels[0]
    assert wf._dirty and not wf._iso
    win.on_tab_changed(win.eps_tab_index)
    win.show()
    QApplication.processEvents()
    assert not wf._dirty and len(wf._iso) == 6  # rendu rattrapé à l'affichage


def test_iso_waterfall_contours(win):
    _eps_setup(win)
    win.on_tab_changed(win.eps_tab_index)
    for k in range(6):
        feed(win, one(0.2 + 0.05 * k))
    wf = win.wf_iso
    assert len(wf._iso) == 6 and wf._iso_txt.isVisible()
    assert "isocontours ε''" in wf._iso_txt.toPlainText()
    win.wf_niso.setValue(3)
    assert len(wf._iso) == 3
    win.wf_niso.setValue(0)
    assert wf._iso == [] and not wf._iso_txt.isVisible()
    win.wf_niso.setValue(4)
    feed(win, one(0.5))
    assert len(wf._iso) == 4  # reconstruits à chaque sweep, sans accumulation
    win.clear_waterfalls()
    assert wf._iso == [] and wf.aux_rows == []


def test_waterfall_colour_scale_manual_and_auto(win):
    _eps_setup(win)
    for k in range(3):
        feed(win, one(0.2 + 0.1 * k))
    lo, hi = win.wf_iso.levels
    assert lo < hi  # auto : bornes issues des données
    win.wf_auto.setChecked(False)
    win.wf_lim["re", "min"].setValue(5.0)
    win.wf_lim["re", "max"].setValue(60.0)
    assert win.wf_iso.levels == (5.0, 60.0)
    assert win.wf_iso.levels == (5.0, 60.0)  # le waterfall à isocontours suit l'échelle de ε'
    win.wf_lim["tan", "min"].setValue(0.1)
    win.wf_lim["tan", "max"].setValue(0.9)
    assert win.wf_tan.levels == (0.1, 0.9)
    win.wf_auto.setChecked(True)
    assert win.wf_iso.levels != (5.0, 60.0)


def test_waterfall_depth_grid_change_and_clear(win):
    _eps_setup(win)
    win.wf_depth.setValue(5)
    for _ in range(8):
        feed(win, one(0.2))
    assert len(win.wf_iso.rows) == 5
    n_before = win.wf_iso.seq
    win.on_sweep(F[:20], sraw(one(0.2))[:20], PARAMS)  # autre grille : on repart de zéro
    assert len(win.wf_iso.rows) == 1 and win.wf_iso.seq == n_before + 1
    win.c0_ff.setValue(120.0)  # changement de réglage ε* : efface
    assert win.wf_iso.rows == []


def test_standard_capture_not_in_waterfall(win):
    _eps_setup(win)
    win.pending = "open"
    feed(win, one(1.0))
    assert win.wf_iso.rows == []
    feed(win, one(0.2))
    assert len(win.wf_iso.rows) == 1


def test_waterfall_selector_shows_one_at_a_time(win):
    _eps_setup(win)
    win.on_tab_changed(win.eps_tab_index)
    assert win.wf_iso.isVisibleTo(win) and not win.wf_tan.isVisibleTo(win)  # défaut : opt. 1
    feed(win, one(0.2))
    win.wf_select.setCurrentIndex(1)
    assert win.wf_tan.isVisibleTo(win) and not win.wf_iso.isVisibleTo(win)
    feed(win, one(0.3))
    win.wf_select.setCurrentIndex(0)
    assert win.wf_iso.isVisibleTo(win) and not win.wf_tan.isVisibleTo(win)
    # les deux ont reçu les deux sweeps : l'historique n'est pas perdu en basculant
    assert len(win.wf_iso.rows) == len(win.wf_tan.rows) == 2
    assert win.plot_eps.isVisibleTo(win)


def feed_g(w, g):
    """Sweep dont S11 mesuré = g (sans distorsion de boîtier), S21 constant."""
    w.on_sweep(F, np.stack([g, np.full(len(F), 0.01 + 0j)], axis=1), PARAMS)


def _build_cal(w):
    for name, g in (("open", one(1)), ("short", one(-1)), ("load", one(0))):
        w.pending = name
        feed(w, g)
    w.pipeline.build_cal()
    w.pipeline.save_cal(str(w.pipeline.cal_path or mw.LAST_CAL.parent / "t.cal"))


def test_filter_rejects_inconsistent_sweeps(win):
    win.chk_filt.setChecked(True)
    win.filt_sp["d_min"].setValue(0.1)
    win.filt_sp["d_max"].setValue(1.0)
    win.filt_sp["radius"].setValue(0.5)
    g_ok = 0.2 + 0.3 * np.linspace(0, 1, len(F)) ** 2  # extrémités à 0,3, tout dans le voisinage
    feed_g(win, g_ok.astype(complex))
    assert win.filt.accepted == 1 and win.last is not None
    before = win.last[1].copy()
    feed_g(win, one(0.2))  # extrémités confondues
    assert win.filt.rejected == 1 and "Sweep rejeté" in win.status.text()
    g_bump = g_ok.astype(complex)
    g_bump[len(F) // 2] += 0.9  # point intermédiaire loin des extrémités
    feed_g(win, g_bump)
    assert win.filt.rejected == 2
    np.testing.assert_array_equal(win.last[1], before)  # sweeps rejetés non affichés
    win.chk_filt.setChecked(False)
    feed_g(win, one(0.2))
    assert win.last is not None and not np.array_equal(win.last[1], before)


def test_standard_capture_bypasses_filter(win):
    win.chk_filt.setChecked(True)
    win.pending = "open"
    feed(win, one(0.2))  # extrémités confondues mais capture de standard
    assert "open" in win.pipeline.captured()
    assert win.filt.rejected == 0


def test_grid_aligned_on_calibration(win):
    assert win.start_mhz.isEnabled()
    _build_cal(win)
    win._set_chk(True)
    win.align_to_cal()
    assert win.points.value() == len(F)
    assert abs(win.start_mhz.value() - F[0] / 1e6) < 1e-6
    assert abs(win.stop_mhz.value() - F[-1] / 1e6) < 1e-6
    assert not win.points.isEnabled()
    win.chk_cal.setChecked(False)  # libère la grille
    assert win.points.isEnabled()


def test_last_saved_cal_is_default(win, tmp_path):
    _build_cal(win)
    path = tmp_path / "bench.cal"
    win.pipeline.save_cal(str(path))
    win.remember_cal(path)
    win2 = MainWindow("/dev/inexistant")
    try:
        assert win2.pipeline.cal is not None
        assert win2.pipeline.cal_path == str(path)
        assert win2.points.value() == len(F) and not win2.points.isEnabled()
    finally:
        win2.close()


def _tissue():
    return 8 + 40 / (1 + 1j * W * 1.5e-10) - 1j * 0.8 / (W * 8.85e-12)


def test_normalized_eps_invariant_to_contact(win):
    _eps_setup(win)
    win.chk_norm.setChecked(True)
    out = []
    for a in (0.4, 1.0):
        feed_g(win, probe(a * _tissue() + (1 - a)))
        out.append(np.array(win.ceps1.getData()[1]))
    np.testing.assert_allclose(out[0], out[1], rtol=1e-6, atol=1e-9)
    win.chk_norm.setChecked(False)
    feed_g(win, probe(0.4 * _tissue() + 0.6))
    assert not np.allclose(np.array(win.ceps1.getData()[1]), out[1])


def test_contact_gate_rejects_weak_contact(win):
    _eps_setup(win)
    feed_g(win, probe(_tissue()))
    full = win.contact_lbl.text()
    assert "Contact |ε*−1|" in full
    lvl = float(full.split("= ")[1].split()[0])
    win.contact_min.setValue(lvl * 0.9)
    before = win.last[1].copy()
    feed_g(win, probe(0.1 * _tissue() + 0.9))  # presque l'air
    assert win.n_contact_rej == 1 and "contact insuffisant" in win.status.text()
    np.testing.assert_array_equal(win.last[1], before)
    feed_g(win, probe(_tissue()))  # contact normal : accepté
    assert win.n_contact_rej == 1


def test_eps_page_columns(win):
    page = win.stack.widget(2)
    left, right = page.layout().itemAt(0).layout(), page.layout().itemAt(1).layout()
    assert left.indexOf(win.smith_eps) >= 0 and left.indexOf(win.plot_eps) >= 0
    assert right.indexOf(win.wf_iso) >= 0 and right.indexOf(win.wf_tan) >= 0
    assert right.indexOf(win.plot_eps) < 0
    win.on_tab_changed(win.cls_tab_index)
    assert win.stack.currentIndex() == 2


def test_zone_recording_training_and_live_prediction(win):
    _eps_setup(win)
    zones = {"nez": _tissue(), "doigt": 6 + 25 / (1 + 1j * W * 3e-10) - 1j * 0.3 / (W * 8.85e-12)}
    rng = np.random.default_rng(1)
    win.rec_n.setValue(3)
    win.start_recording()  # aucun sweep reçu : refusé
    assert win.rec is None
    feed_g(win, probe(_tissue()))
    for z, e in zones.items():
        win.zone_box.setEditText(z)
        for _ in range(4):
            win.start_recording()
            a = rng.uniform(0.4, 1.0)
            for _ in range(3):
                noise = 1 + 0.002 * rng.standard_normal(len(F))
                feed_g(win, probe((a * e + 1 - a) * noise))
            assert win.rec is None
    assert len(win.ds) == 2 * 4 * 3 and win.ds.n_sessions("nez") == 4
    win.train_model()
    assert win.model is not None and "Exactitude par session" in win.cls_report.toPlainText()
    win.chk_live.setChecked(True)
    feed_g(win, probe(0.7 * zones["doigt"] + 0.3))
    assert "Zone : doigt" in win.cls_label.text()
    feed_g(win, probe(0.7 * zones["nez"] + 0.3))
    assert "Zone : nez" in win.cls_label.text()


def test_recording_refused_without_eps(win):
    feed_g(win, probe(_tissue()))
    win.start_recording()
    assert win.rec is None


def test_power_controls_defaults_and_signal(win):
    assert (win.pwr_high.currentData(), win.pwr_low.currentData()) == (3, 1)
    sent = []
    win._power.connect(lambda a, b: sent.append((a, b)))
    win.on_power()  # déconnecté : rien n'est envoyé
    assert sent == []
    win.connected = True
    win.pwr_high.setCurrentIndex(1)
    assert sent[-1] == (1, 1)
    win.pwr_low.setCurrentIndex(0)
    assert sent[-1] == (1, 0)
    win.connected = False


def test_worker_backpressure(monkeypatch):
    """Sans ack du thread Qt, le worker n'empile pas plus de MAX_INFLIGHT sweeps."""
    import time

    from vna_app.gui.worker import Worker

    app = QApplication.instance() or QApplication([])

    class StubVNA:
        def sweep(self):
            return F, sraw(one(0.2))

        def close(self):
            pass

    w = Worker("/dev/inexistant")
    w._vna = StubVNA()
    w._params = dict(PARAMS)
    got = []
    w.sweep_done.connect(lambda *a: got.append(a))

    def spin(sec):
        t = time.monotonic() + sec
        while time.monotonic() < t:
            app.processEvents()
            time.sleep(0.005)

    w.run(True)
    spin(0.3)
    assert len(got) == w.MAX_INFLIGHT  # bloqué : l'affichage n'a rien acquitté
    w.ack()
    spin(0.3)
    assert len(got) == w.MAX_INFLIGHT + 1  # un acquittement = un sweep de plus
    w.stop()
    w.ack()
    w.ack()
    spin(0.2)
    assert len(got) == w.MAX_INFLIGHT + 1  # stop effectif : plus rien n'est émis


# -- cartographie caméra (onglet Carto) ----------------------------------------------


def _cal_and_eps(win, monkeypatch):
    for name, g in (("open", 1), ("short", -1), ("load", 0)):
        win.pending = name
        feed(win, one(g))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    win.build_cal()
    win.eps_mode.setCurrentIndex(1)  # capacitif


def test_map_point_added_on_contact(win, monkeypatch):
    import time as _time

    _cal_and_eps(win, monkeypatch)
    t = _time.monotonic()
    win.scan_positions.push(t, 100.0, 50.0)
    win.tip_dx.setValue(3)
    win.tip_dy.setValue(-5)
    win.on_sweep(F, sraw(probe(one(40 - 10j))), dict(PARAMS, t_mono=t))
    assert len(win.scan_map) == 1
    _, x, y, e1, e2 = win.scan_map.arrays()
    assert x[0] == 103.0 and y[0] == 45.0
    np.testing.assert_allclose(e1[0], 40.0, rtol=1e-6)
    np.testing.assert_allclose(e2[0], 10.0, rtol=1e-6)
    assert "contact" in win.map_live.text()
    assert len(win.scan_view.path.getData()[0]) == 1  # point affiché


def test_map_requires_contact_and_fresh_position(win, monkeypatch):
    import time as _time

    _cal_and_eps(win, monkeypatch)
    t = _time.monotonic()
    # contact mais aucune position caméra : pas de point inventé
    win.on_sweep(F, sraw(probe(one(40 - 10j))), dict(PARAMS, t_mono=t))
    assert len(win.scan_map) == 0
    # position fraîche mais pas de contact (air, ε ≈ 1)
    win.scan_positions.push(t, 10.0, 10.0)
    win.on_sweep(F, sraw(probe(one(1 + 0j))), dict(PARAMS, t_mono=t))
    assert len(win.scan_map) == 0
    assert "pas de contact" in win.map_live.text()


def test_map_tab_switch_export_and_clear(win, tmp_path, monkeypatch):
    win.on_tab_changed(win.map_tab_index)
    assert win.stack.currentIndex() == 3
    assert not win.smith.isVisibleTo(win)  # toute la largeur pour l'image
    win.on_tab_changed(0)
    assert win.smith.isVisibleTo(win)
    win.scan_map.add(1.0, 1.0, 2.0, 3.0, 4.0)
    out = tmp_path / "c.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_map()
    d = np.loadtxt(out, delimiter=",", skiprows=1)
    np.testing.assert_allclose(d, [1.0, 1.0, 2.0, 3.0, 4.0])
    win.clear_map()
    assert len(win.scan_map) == 0


def test_map_cmap_switch_recolors_path(win):
    from vna_app.gui.scanview import CMAPS

    win.scan_map.add(1.0, 10.0, 20.0, 5.0, 1.0)
    win.scan_map.add(2.0, 30.0, 40.0, 50.0, 2.0)
    win.update_map_path()
    win.map_cmap.setCurrentIndex(1)  # Turbo
    assert win.scan_view.cmap is CMAPS["turbo"]
    assert len(win.scan_view.path.getData()[0]) == 2  # chemin recoloré, pas perdu
    win.map_cmap.setCurrentIndex(0)
    assert win.scan_view.cmap is CMAPS["magenta"]


def test_scanview_normal_and_axis_overlays(win):
    from vna_app.vision.tracker import Tip

    sv = win.scan_view
    sv.set_tip(Tip(100.0, 80.0, 200.0, ax=1.0, ay=0.0))
    x, y = sv.axis_line.getData()
    assert len(x) == 2 and x[0] < 100.0 < x[1]  # segment le long de l'axe, à travers la pointe
    sv.set_normal(100.0, 80.0, np.array([0.34, 0.0, -0.94]))
    assert sv.normal_text.isVisible() and "°" in sv.normal_text.textItem.toPlainText()
    sv.set_normal(0.0, 0.0, None)
    assert not sv.normal_text.isVisible()
    sv.set_tip(Tip(100.0, 80.0, 200.0))  # sans gaine : pas d'axe
    ax_data = sv.axis_line.getData()[0]
    assert ax_data is None or len(ax_data) == 0


class _FakeCam:
    def __init__(self):
        self.frame = np.zeros((8, 8, 3), np.uint8)

    def latest(self):
        return (0.0, self.frame, None)

    def stop(self, timeout=0):
        pass


def test_export_map_saves_csv_and_images(win, tmp_path, monkeypatch):
    win.scan_map.add(1.0, 1.0, 2.0, 3.0, 4.0)
    win.cam_stream = _FakeCam()
    win.scan_view.set_frame(win.cam_stream.frame)
    out = tmp_path / "carto.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_map()
    assert out.exists()
    assert (tmp_path / "carto_image.png").exists()  # image caméra brute
    assert (tmp_path / "carto_vue.png").exists()  # vue annotée
    win.cam_stream = None


def test_refs_save_and_load_buttons(win, tmp_path, monkeypatch):
    for name, g in (("air", 0.9), ("probe_short", -0.9), ("ref", 0.2)):
        win.pending = name
        feed(win, one(g))
    out = tmp_path / "refs.npz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.save_refs()
    assert out.exists()
    win.pipeline.standards.clear()
    win.update_captured_labels()
    assert "Air" not in win.eps_captured.text()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(out), ""))
    win.load_refs()
    assert "Air" in win.eps_captured.text()
    assert "Court-circuit" in win.eps_captured.text()


def test_refs_reloaded_at_startup(win, tmp_path, monkeypatch):
    for name, g in (("air", 0.9), ("probe_short", -0.9), ("ref", 0.2)):
        win.pending = name
        feed(win, one(g))
    out = tmp_path / "refs.npz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.save_refs()
    w2 = MainWindow("/dev/inexistant")  # LAST_REFS pointe sur refs.npz : rechargées
    try:
        assert "air" in w2.pipeline.standards and "ref" in w2.pipeline.standards
        assert "Air" in w2.eps_captured.text()
    finally:
        w2.close()


def _frame_with_patch(h_val=84, s_val=200, v_val=220):
    """Trame RGB 64×64 avec un carré coloré HSV donné autour de (32, 32)."""
    import cv2

    hsv = np.zeros((64, 64, 3), np.uint8)
    hsv[:, :, 2] = 30  # fond sombre quasi gris
    hsv[24:40, 24:40] = (h_val, s_val, v_val)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


class _FakeCamFrame:
    def __init__(self, rgb):
        self.rgb = rgb

    def latest(self):
        return (0.0, self.rgb, None)

    def stop(self, timeout=0):
        pass


def test_pick_ring_measures_hsv(win):
    pytest.importorskip("cv2")
    win.cam_stream = _FakeCamFrame(_frame_with_patch(h_val=110, s_val=180, v_val=200))
    win.btn_pick_ring.setChecked(True)
    assert win._pick_mode == "ring"
    win.on_scan_click(32.0, 32.0)
    assert abs(win.trk_sp["h_center"].value() - 110) <= 2
    assert win.trk_sp["s_min"].value() == int(180 * 0.6)
    assert win.trk_sp["v_min"].value() == int(200 * 0.5)
    assert not win.btn_pick_ring.isChecked()  # mesure faite : mode quitté
    assert "Mesuré" in win.cam_status.text()
    win.cam_stream = None


def test_pick_sheath_sets_center_only(win):
    pytest.importorskip("cv2")
    win.cam_stream = _FakeCamFrame(_frame_with_patch(h_val=120, s_val=255, v_val=230))
    s_before = win.trk_sp["s_min"].value()
    win.btn_pick_sheath.setChecked(True)
    win.on_scan_click(32.0, 32.0)
    assert abs(win.trk_sp["sheath_h_center"].value() - 120) <= 2
    assert win.trk_sp["s_min"].value() == s_before  # seuils bague inchangés
    win.cam_stream = None


def test_pick_buttons_mutually_exclusive(win):
    win.btn_pick_ring.setChecked(True)
    win.btn_pick_sheath.setChecked(True)
    assert not win.btn_pick_ring.isChecked()
    assert win._pick_mode == "sheath"


def test_masked_view_tints_matches(win):
    pytest.importorskip("cv2")
    from vna_app.vision.tracker import RingTracker

    win.cam_tracker = RingTracker()
    rgb = _frame_with_patch(h_val=win.trk_sp["h_center"].value())
    out = win._masked_view(rgb)
    assert (out[32, 32] == (255, 0, 255)).all()  # pixels bague teintés magenta
    assert (out[5, 5] == rgb[5, 5]).all()  # fond inchangé
    win.cam_tracker = None
    assert win._masked_view(rgb) is rgb  # sans tracker : image brute


def test_carto_settings_roundtrip(win, tmp_path):
    from PyQt5.QtCore import QSettings

    win.trk_sp["h_center"].setValue(123)
    win.cam_expo.setValue(42)
    win.map_contact.setValue(7.5)
    win.map_cmap.setCurrentIndex(1)  # turbo
    win.chk_depth.setChecked(False)
    s = QSettings(str(tmp_path / "carto.ini"), QSettings.IniFormat)
    win._carto_save(s)
    s.sync()

    # valeurs écrasées puis restaurées
    win.trk_sp["h_center"].setValue(10)
    win.cam_expo.setValue(100)
    win.map_contact.setValue(1.0)
    win.map_cmap.setCurrentIndex(0)
    win.chk_depth.setChecked(True)
    win._carto_restore(QSettings(str(tmp_path / "carto.ini"), QSettings.IniFormat))
    assert win.trk_sp["h_center"].value() == 123
    assert win.cam_expo.value() == 42
    assert win.map_contact.value() == 7.5
    assert win.map_cmap.currentData() == "turbo"
    assert not win.chk_depth.isChecked()


# -- affichage caméra adaptatif (X11 déporté via SSH) -------------------------------------


class _FakeCamPoll:
    error = ""

    def __init__(self):
        self.frame = np.zeros((480, 848, 3), np.uint8)

    def latest(self):
        return (0.0, self.frame, None)

    def stop(self, timeout=0):
        pass


def test_poll_camera_rearms_single_shot_timer(win):
    """Timer mono-coup réarmé en fin de poll : la file d'événements ne s'accumule pas."""
    assert win.cam_timer.isSingleShot()
    win.cam_stream = _FakeCamPoll()
    win.poll_camera()
    assert win.cam_timer.isActive()
    win.cam_stream = None
    win.cam_timer.stop()


def test_display_decimation_policy():
    from vna_app.gui.main_window import _display_decimation

    assert _display_decimation(1, 0, 0.30, 0.066) == (2, 30)  # congestion : dégrade
    assert _display_decimation(2, 0, 0.30, 0.066) == (4, 30)
    assert _display_decimation(4, 0, 9.90, 0.066) == (4, 0)  # plafond 1/4
    assert _display_decimation(2, 5, 0.30, 0.066) == (2, 4)  # hold : pas de réévaluation
    assert _display_decimation(2, 0, 0.066, 0.066) == (1, 150)  # ça respire : remonte
    assert _display_decimation(1, 0, 0.066, 0.066) == (1, 0)  # plancher
    assert _display_decimation(2, 0, 0.10, 0.066) == (2, 0)  # hystérésis : stable


def test_poll_camera_downscale_keeps_full_res_coords(win):
    """Image affichée réduite mais repère pleine résolution : overlays et clics intacts."""
    import pytest as _pytest

    win.cam_stream = _FakeCamPoll()
    win._cam_scale, win._cam_hold = 2, 10**6  # décimation figée pour le test
    win.poll_camera()
    img = win.scan_view.img
    assert img.image.shape[:2] == (240, 424)  # moitié des pixels envoyés à l'affichage
    assert img.transform().m11() == _pytest.approx(2.0)  # étirés sur le repère 848×480
    assert img.transform().m22() == _pytest.approx(2.0)
    win._cam_scale = 1
    win.poll_camera()
    assert img.image.shape[:2] == (480, 848)
    assert img.transform().m11() == _pytest.approx(1.0)
    win.cam_stream = None
    win.cam_timer.stop()
