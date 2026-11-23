"""Worker d'acquisition : toute l'I/O série vit dans ce thread, jamais dans le thread Qt."""

from __future__ import annotations

import threading
import time

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from vna_app.driver.litevna import LiteVNA, LiteVNAError


class Worker(QObject):
    connected = pyqtSignal(dict)  # device_info
    disconnected = pyqtSignal()
    sweep_done = pyqtSignal(object, object, dict)  # freqs, s_raw, params
    error = pyqtSignal(str)

    def __init__(self, port: str = "/dev/litevna"):
        super().__init__()
        self._port = port
        self._vna = None
        self._params = None
        self._running = False
        # Contre-pression : sweeps émis mais pas encore affichés par le thread Qt. Au-delà de
        # MAX_INFLIGHT on suspend l'acquisition, sinon les sweep_done s'empilent dans la file
        # d'événements Qt et stop / changement d'affichage / fermeture attendent tout le retard.
        self._inflight = 0
        self._lock = threading.Lock()

    MAX_INFLIGHT = 2

    @pyqtSlot()
    def connect_device(self):
        try:
            self._vna = LiteVNA(self._port)
            self._vna.connect()
            self.connected.emit(self._vna.device_info())
        except LiteVNAError as e:
            self._vna = None
            self.error.emit(str(e))

    @pyqtSlot()
    def disconnect_device(self):
        self._running = False
        if self._vna is not None:
            self._vna.close()
            self._vna = None
        self.disconnected.emit()

    @pyqtSlot(int, int)
    def set_power(self, adf4350, si5351):
        """Puissance source (0..3) ; sans appareil : ignoré (GUI renvoie avant chaque sweep)."""
        if self._vna is None:
            return
        try:
            self._vna.set_power(adf4350, si5351)
        except (LiteVNAError, ValueError) as e:
            self.error.emit(str(e))

    @pyqtSlot(float, float, int, int)
    def configure(self, start, stop, points, avg):
        if self._vna is None:
            self.error.emit("appareil non connecté")
            return
        try:
            self._vna.configure_sweep(start, stop, points, avg)
            self._params = {
                "start": start,
                "stop": stop,
                "points": points,
                "avg": avg,
                "step_hz": self._vna.step_hz,
            }
        except (LiteVNAError, ValueError) as e:
            self._params = None
            self.error.emit(str(e))

    @pyqtSlot(bool)
    def run(self, continuous: bool):
        """Un sweep, ou en boucle jusqu'à stop()."""
        if self._vna is None or self._params is None:
            self.error.emit("appareil non connecté ou non configuré")
            return
        self._running = continuous
        with self._lock:
            self._inflight = 0
        self._step()

    def ack(self):
        """Appelable depuis le thread Qt : un sweep émis a été traité (affiché)."""
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def _step(self):
        t0 = time.monotonic()
        try:
            freqs, s = self._vna.sweep()
        except LiteVNAError as e:
            self._running = False
            self.error.emit(str(e))
            if self._vna is not None:
                self.disconnect_device()
            return
        with self._lock:
            self._inflight += 1
        params = dict(self._params)
        params["t_mono"] = 0.5 * (t0 + time.monotonic())  # milieu du sweep (fusion caméra)
        self.sweep_done.emit(freqs, s, params)
        if self._running:
            QTimer.singleShot(0, self._next)  # laisse passer les slots (stop, disconnect)

    def _next(self):
        if not self._running or self._vna is None:
            return
        if self._inflight >= self.MAX_INFLIGHT:  # l'affichage est en retard : on attend
            QTimer.singleShot(10, self._next)
            return
        self._step()

    def stop(self):
        """Appelable depuis le thread Qt : simple drapeau."""
        self._running = False
