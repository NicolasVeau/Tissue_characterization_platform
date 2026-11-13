from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vna_app.calibration import CalibrationError
from vna_app.gui.worker import Worker
from vna_app.processing.export import write_s2p
from vna_app.processing.permittivity import REFERENCE_LIQUIDS, to_eps_columns
from vna_app.processing.pipeline import Pipeline

EPS_MODES = [("off", "Désactivé"), ("capacitive", "Capacitif (C0)"), ("three_ref", "3 références")]
CAL_LABELS = {"open": "Open", "short": "Short", "load": "Load", "thru": "Thru (S21)"}
EPS_LABELS = {"air": "Air", "probe_short": "Court-circuit", "ref": "Liquide de réf."}


def _db(x):
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.abs(x))


class MainWindow(QMainWindow):
    _connect = pyqtSignal()
    _disconnect = pyqtSignal()
    _configure = pyqtSignal(float, float, int, int)
    _run = pyqtSignal(bool)

    def __init__(self, port: str = "/dev/litevna"):
        super().__init__()
        self.setWindowTitle("vna_app — LiteVNA 64")
        self.resize(1250, 800)
        self.info = None
        self.last = None  # (freqs, s_raw, params)
        self.result = None  # processing.pipeline.Result
        self.connected = False
        self.pipeline = Pipeline()
        self.pending = None  # standard à capturer au prochain sweep

        tabs = QTabWidget()
        tabs.addTab(self._build_acq_tab(), "Acquisition")
        tabs.addTab(self._build_cal_tab(), "Calibration")
        tabs.addTab(self._build_eps_tab(), "ε*")
        tabs.setFixedWidth(300)
        self.status = QLabel("Déconnecté")
        self.status.setWordWrap(True)
        side = QVBoxLayout()
        side.addWidget(tabs, 1)
        side.addWidget(self.status)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(310)

        self.plot11 = pg.PlotWidget(title="|S11| (brut)")
        self.plot21 = pg.PlotWidget(title="|S21| (brut)")
        self.plot_eps = pg.PlotWidget(title="ε*")
        for p in (self.plot11, self.plot21, self.plot_eps):
            p.setLabel("bottom", "Fréquence", units="Hz")
            p.showGrid(x=True, y=True)
        self.plot11.setLabel("left", "dB")
        self.plot21.setLabel("left", "dB")
        self.plot_eps.setLabel("left", "ε', ε''")
        self.plot21.setXLink(self.plot11)
        self.plot_eps.setXLink(self.plot11)
        self.plot_eps.addLegend()
        self.c11 = self.plot11.plot(pen=pg.mkPen("y", width=2))
        self.c21 = self.plot21.plot(pen=pg.mkPen("c", width=2))
        self.ceps1 = self.plot_eps.plot(pen=pg.mkPen("g", width=2), name="ε'")
        self.ceps2 = self.plot_eps.plot(pen=pg.mkPen("m", width=2), name="ε''")
        self.plot_eps.setVisible(False)
        plots = QVBoxLayout()
        for p in (self.plot11, self.plot21, self.plot_eps):
            plots.addWidget(p)

        root = QHBoxLayout()
        root.addWidget(side_w)
        root.addLayout(plots, 1)
        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)

        # Thread d'acquisition
        self.thread = QThread(self)
        self.worker = Worker(port)
        self.worker.moveToThread(self.thread)
        self._connect.connect(self.worker.connect_device)
        self._disconnect.connect(self.worker.disconnect_device)
        self._configure.connect(self.worker.configure)
        self._run.connect(self.worker.run)
        self.worker.connected.connect(self.on_connected)
        self.worker.disconnected.connect(self.on_disconnected)
        self.worker.sweep_done.connect(self.on_sweep)
        self.worker.error.connect(self.on_error)
        self.thread.start()
        self.refresh_buttons()

    # -- construction des onglets ----------------------------------------------
    def _build_acq_tab(self):
        self.start_mhz = QDoubleSpinBox()
        self.start_mhz.setRange(0.05, 6300)
        self.start_mhz.setValue(400)
        self.stop_mhz = QDoubleSpinBox()
        self.stop_mhz.setRange(0.05, 6300)
        self.stop_mhz.setValue(3000)
        for w in (self.start_mhz, self.stop_mhz):
            w.setDecimals(3)
            w.setSuffix(" MHz")
        self.points = QSpinBox()
        self.points.setRange(2, 65535)
        self.points.setValue(201)
        self.avg = QSpinBox()
        self.avg.setRange(1, 255)
        self.avg.setValue(1)
        self.btn_conn = QPushButton("Connecter")
        self.btn_single = QPushButton("Sweep unique")
        self.btn_cont = QPushButton("Continu")
        self.btn_cont.setCheckable(True)
        self.btn_save = QPushButton("Enregistrer .s2p…")
        self.btn_conn.clicked.connect(self.toggle_connect)
        self.btn_single.clicked.connect(lambda: self.start(False))
        self.btn_cont.toggled.connect(self.toggle_continuous)
        self.btn_save.clicked.connect(self.save)
        form = QFormLayout()
        form.addRow("Début", self.start_mhz)
        form.addRow("Fin", self.stop_mhz)
        form.addRow("Points", self.points)
        form.addRow("Moyenne", self.avg)
        lay = QVBoxLayout()
        lay.addWidget(self.btn_conn)
        lay.addLayout(form)
        lay.addWidget(self.btn_single)
        lay.addWidget(self.btn_cont)
        lay.addWidget(self.btn_save)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_cal_tab(self):
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Un sweep par standard (mêmes paramètres que la mesure) :"))
        self.cal_buttons = {}
        for k, label in CAL_LABELS.items():
            b = QPushButton(f"Capturer {label}")
            b.clicked.connect(lambda _=False, k=k: self.capture(k))
            self.cal_buttons[k] = b
            lay.addWidget(b)
        self.cal_captured = QLabel("Capturés : —")
        self.cal_captured.setWordWrap(True)
        self.btn_build = QPushButton("Calculer la calibration…")
        self.btn_load_cal = QPushButton("Charger un .cal…")
        self.chk_cal = QCheckBox("Appliquer la calibration")
        self.cal_info = QLabel("Aucune calibration")
        self.cal_info.setWordWrap(True)
        self.btn_build.clicked.connect(self.build_cal)
        self.btn_load_cal.clicked.connect(self.load_cal)
        self.chk_cal.toggled.connect(self.on_apply_cal)
        for w in (
            self.cal_captured,
            self.btn_build,
            self.btn_load_cal,
            self.chk_cal,
            self.cal_info,
        ):
            lay.addWidget(w)
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_eps_tab(self):
        self.eps_mode = QComboBox()
        for key, label in EPS_MODES:
            self.eps_mode.addItem(label, key)
        self.c0_ff = QDoubleSpinBox()
        self.c0_ff.setRange(0.001, 100000)
        self.c0_ff.setDecimals(3)
        self.c0_ff.setValue(80.0)
        self.c0_ff.setSuffix(" fF")
        self.ref_liquid = QComboBox()
        self.ref_liquid.addItems(sorted(REFERENCE_LIQUIDS))
        self.smooth = QSpinBox()
        self.smooth.setRange(1, 51)
        self.smooth.setSingleStep(2)
        form = QFormLayout()
        form.addRow("Méthode", self.eps_mode)
        form.addRow("C0 (capacitif)", self.c0_ff)
        form.addRow("Liquide de réf.", self.ref_liquid)
        form.addRow("Lissage (impair)", self.smooth)
        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addWidget(QLabel("Références (méthode 3 références, S11 brut) :"))
        self.eps_buttons = {}
        for k, label in EPS_LABELS.items():
            b = QPushButton(f"Capturer {label}")
            b.clicked.connect(lambda _=False, k=k: self.capture(k))
            self.eps_buttons[k] = b
            lay.addWidget(b)
        self.eps_captured = QLabel("Capturées : —")
        self.eps_captured.setWordWrap(True)
        self.btn_save_eps = QPushButton("Enregistrer ε* (.csv)…")
        lay.addWidget(self.eps_captured)
        lay.addWidget(self.btn_save_eps)
        lay.addStretch(1)
        self.eps_mode.currentIndexChanged.connect(self.on_eps_params)
        self.c0_ff.valueChanged.connect(self.on_eps_params)
        self.ref_liquid.currentIndexChanged.connect(self.on_eps_params)
        self.smooth.valueChanged.connect(self.on_eps_params)
        self.btn_save_eps.clicked.connect(self.save_eps)
        w = QWidget()
        w.setLayout(lay)
        return w

    # -- état
    def refresh_buttons(self):
        self.btn_conn.setText("Déconnecter" if self.connected else "Connecter")
        for b in [
            self.btn_single,
            self.btn_cont,
            *self.cal_buttons.values(),
            *self.eps_buttons.values(),
        ]:
            b.setEnabled(self.connected)
        self.btn_save.setEnabled(self.last is not None)
        self.btn_save_eps.setEnabled(self.result is not None and self.result.eps is not None)
        self.chk_cal.setEnabled(self.pipeline.cal is not None)

    def toggle_connect(self):
        self.btn_conn.setEnabled(False)
        self.status.setText("Connexion…" if not self.connected else "Déconnexion…")
        (self._disconnect if self.connected else self._connect).emit()

    def on_connected(self, info):
        self.connected, self.info = True, info
        self.status.setText(f"Connecté — fw {info['firmware']}")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()

    def on_disconnected(self):
        self.connected = False
        self.pending = None
        self.btn_cont.blockSignals(True)
        self.btn_cont.setChecked(False)
        self.btn_cont.blockSignals(False)
        self.status.setText("Déconnecté")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()

    def on_error(self, msg):
        self.pending = None
        self.status.setText(f"Erreur : {msg}")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()
        QMessageBox.warning(self, "vna_app", msg)

    # -- acquisition
    def start(self, continuous):
        self._configure.emit(
            self.start_mhz.value() * 1e6,
            self.stop_mhz.value() * 1e6,
            self.points.value(),
            self.avg.value(),
        )
        self._run.emit(continuous)
        self.status.setText("Acquisition…")

    def toggle_continuous(self, on):
        if on:
            self.start(True)
        else:
            self.worker.stop()
            self.status.setText("Arrêté")

    def capture(self, name):
        """Capture brute de `name` au prochain sweep (sweep unique)."""
        self.btn_cont.blockSignals(True)
        self.btn_cont.setChecked(False)
        self.btn_cont.blockSignals(False)
        self.worker.stop()
        self.pending = name
        self.start(False)
        self.status.setText(f"Capture {name}…")

    def on_sweep(self, freqs, s_raw, params):
        self.last = (freqs, s_raw, params)
        if self.pending is not None:
            name, self.pending = self.pending, None
            self.pipeline.capture(name, freqs, s_raw)  # toujours le BRUT
            self.update_captured_labels()
            self.status.setText(f"{name} capturé ({len(freqs)} pts)")
        self.redisplay()

    def redisplay(self):
        if self.last is None:
            return
        freqs, s_raw, _ = self.last
        self.result = self.pipeline.process(freqs, s_raw)
        r = self.result
        tag = "calibré" if r.calibrated else "brut"
        self.plot11.setTitle(f"|S11| ({tag})")
        self.plot21.setTitle(f"|S21| ({tag})")
        self.c11.setData(freqs, _db(r.s[:, 0]))
        self.c21.setData(freqs, _db(r.s[:, 1]))
        show_eps = r.eps is not None
        self.plot_eps.setVisible(self.pipeline.eps_mode != "off")
        if show_eps:
            e1, e2, _ = to_eps_columns(r.eps)
            self.ceps1.setData(freqs, e1)
            self.ceps2.setData(freqs, e2)
        else:
            self.ceps1.setData([], [])
            self.ceps2.setData([], [])
        if r.notes:
            self.status.setText(" / ".join(r.notes))
        self.refresh_buttons()

    # -- calibration
    def update_captured_labels(self):
        got = self.pipeline.captured()
        cal = [CAL_LABELS[k] for k in CAL_LABELS if k in got]
        eps = [EPS_LABELS[k] for k in EPS_LABELS if k in got]
        self.cal_captured.setText("Capturés : " + (", ".join(cal) or "—"))
        self.eps_captured.setText("Capturées : " + (", ".join(eps) or "—"))

    def build_cal(self):
        try:
            cal = self.pipeline.build_cal()
        except CalibrationError as e:
            self.status.setText(f"Calibration : {e}")
            QMessageBox.warning(self, "Calibration", str(e))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la calibration", "bench.cal", "Calibration (*.cal)"
        )
        if path:
            self.pipeline.save_cal(path)
        self.cal_info.setText(
            f"{cal.meta['type']} — {len(cal.freqs)} pts, "
            f"{cal.freqs[0] / 1e6:.0f}–{cal.freqs[-1] / 1e6:.0f} MHz"
            + ("" if path else " (non enregistrée)")
        )
        self._set_chk(True)
        self.redisplay()

    def load_cal(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une calibration", "", "Calibration (*.cal)"
        )
        if not path:
            return
        try:
            cal = self.pipeline.load_cal(path)
        except CalibrationError as e:
            self.status.setText(f"Calibration : {e}")
            QMessageBox.warning(self, "Calibration", str(e))
            return
        self.cal_info.setText(
            f"{path} — {len(cal.freqs)} pts, {cal.freqs[0] / 1e6:.0f}–{cal.freqs[-1] / 1e6:.0f} MHz"
        )
        self._set_chk(True)
        self.redisplay()

    def _set_chk(self, on):
        self.chk_cal.blockSignals(True)
        self.chk_cal.setChecked(on)
        self.chk_cal.blockSignals(False)
        self.pipeline.apply_cal = on

    def on_apply_cal(self, on):
        self.pipeline.apply_cal = on
        self.redisplay()

    # -- ε*
    def on_eps_params(self, *_):
        p = self.pipeline
        p.eps_mode = self.eps_mode.currentData()
        p.c0 = self.c0_ff.value() * 1e-15
        p.ref_liquid = self.ref_liquid.currentText()
        p.smooth = self.smooth.value()
        self.redisplay()

    def save_eps(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer ε*", "eps.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        freqs = self.last[0]
        e1, e2, tan = to_eps_columns(self.result.eps)
        np.savetxt(
            path,
            np.column_stack([freqs, e1, e2, tan]),
            delimiter=",",
            header="f_Hz,eps_real,eps_loss,tan_delta",
            comments="",
            fmt="%.9g",
        )
        self.status.setText(f"Enregistré : {path}")

    # -- export s2p
    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "mesure.s2p", "Touchstone (*.s2p)"
        )
        if not path:
            return
        if not path.lower().endswith(".s2p"):
            path += ".s2p"
        freqs, s_raw, p = self.last
        r = self.result
        s = r.s if r is not None else s_raw
        cal_file = None
        if r is not None and r.calibrated:
            cal_file = self.pipeline.cal_path or "<en mémoire, non enregistrée>"
        write_s2p(
            path,
            freqs,
            s,
            step_hz=p["step_hz"],
            points=p["points"],
            averaging=p["avg"],
            info=self.info,
            stop_hz_requested=p["stop"],
            calibration_file=cal_file,
        )
        self.status.setText(f"Enregistré : {path}")

    def closeEvent(self, ev):
        self.worker.stop()
        self._disconnect.emit()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(ev)
