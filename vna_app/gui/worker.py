"""Worker d'acquisition : toute l'I/O série vit dans ce thread, jamais dans le thread Qt."""

from __future__ import annotations

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
        self._step()

    def _step(self):
        try:
            freqs, s = self._vna.sweep()
        except LiteVNAError as e:
            self._running = False
            self.error.emit(str(e))
            if self._vna is not None:
                self.disconnect_device()
            return
        self.sweep_done.emit(freqs, s, dict(self._params))
        if self._running:
            QTimer.singleShot(0, self._step)  # laisse passer les slots (stop, disconnect)

    def stop(self):
        """Appelable depuis le thread Qt : simple drapeau."""
        self._running = False
