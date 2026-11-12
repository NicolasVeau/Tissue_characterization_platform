from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vna_app.gui.worker import Worker
from vna_app.processing.export import write_s2p


class MainWindow(QMainWindow):
    _connect = pyqtSignal()
    _disconnect = pyqtSignal()
    _configure = pyqtSignal(float, float, int, int)
    _run = pyqtSignal(bool)

    def __init__(self, port: str = "/dev/litevna"):
        super().__init__()
        self.setWindowTitle("vna_app — LiteVNA 64")
        self.resize(1100, 700)
        self.info = None
        self.last = None  # (freqs, s_raw, params)
        self.connected = False

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
        self.status = QLabel("Déconnecté")

        form = QFormLayout()
        form.addRow("Début", self.start_mhz)
        form.addRow("Fin", self.stop_mhz)
        form.addRow("Points", self.points)
        form.addRow("Moyenne", self.avg)
        side = QVBoxLayout()
        side.addWidget(self.btn_conn)
        side.addLayout(form)
        side.addWidget(self.btn_single)
        side.addWidget(self.btn_cont)
        side.addWidget(self.btn_save)
        side.addWidget(self.status)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(240)

        self.plot11 = pg.PlotWidget(title="|S11| (brut)")
        self.plot21 = pg.PlotWidget(title="|S21| (brut)")
        for p in (self.plot11, self.plot21):
            p.setLabel("left", "dB")
            p.setLabel("bottom", "Fréquence", units="Hz")
            p.showGrid(x=True, y=True)
        self.plot21.setXLink(self.plot11)
        self.c11 = self.plot11.plot(pen=pg.mkPen("y", width=2))
        self.c21 = self.plot21.plot(pen=pg.mkPen("c", width=2))
        plots = QVBoxLayout()
        plots.addWidget(self.plot11)
        plots.addWidget(self.plot21)

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

        self.btn_conn.clicked.connect(self.toggle_connect)
        self.btn_single.clicked.connect(lambda: self.start(False))
        self.btn_cont.toggled.connect(self.toggle_continuous)
        self.btn_save.clicked.connect(self.save)
        self.refresh_buttons()

    # -- état
    def refresh_buttons(self):
        self.btn_conn.setText("Déconnecter" if self.connected else "Connecter")
        for b in (self.btn_single, self.btn_cont):
            b.setEnabled(self.connected)
        self.btn_save.setEnabled(self.last is not None)

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
        self.btn_cont.blockSignals(True)
        self.btn_cont.setChecked(False)
        self.btn_cont.blockSignals(False)
        self.status.setText("Déconnecté")
        self.btn_conn.setEnabled(True)
        self.refresh_buttons()

    def on_error(self, msg):
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

    def on_sweep(self, freqs, s_raw, params):
        self.last = (freqs, s_raw, params)
        with np.errstate(divide="ignore"):
            self.c11.setData(freqs, 20 * np.log10(np.abs(s_raw[:, 0])))
            self.c21.setData(freqs, 20 * np.log10(np.abs(s_raw[:, 1])))
        self.refresh_buttons()

    def save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "mesure.s2p", "Touchstone (*.s2p)"
        )
        if not path:
            return
        if not path.lower().endswith(".s2p"):
            path += ".s2p"
        freqs, s_raw, p = self.last
        write_s2p(
            path,
            freqs,
            s_raw,
            step_hz=p["step_hz"],
            points=p["points"],
            averaging=p["avg"],
            info=self.info,
            stop_hz_requested=p["stop"],
        )
        self.status.setText(f"Enregistré : {path}")

    def closeEvent(self, ev):
        self.worker.stop()
        self._disconnect.emit()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(ev)
