"""Capture RGB de la RealSense (D455/D435) via OpenCV-V4L2, dans un thread dédié.

La caméra couleur des D4xx est un périphérique UVC standard (/dev/video4 sur ce banc) :
pas besoin de librealsense pour le flux RGB. La profondeur n'est pas utilisée ici : le
contact canule/tissu est détecté par le signal VNA (niveau |ε*−1|), pas par la géométrie.

Modèle « boîte aux lettres » : le thread garde seulement la DERNIÈRE trame (pas de file,
pas de retard accumulé) ; le thread Qt vient la chercher avec latest() sur un QTimer.
Le tracker tourne dans ce thread (jamais dans le thread Qt) et pousse les positions
horodatées (time.monotonic, l'horloge commune avec le worker VNA) dans un PositionBuffer.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from vna_app.vision.fusion import PositionBuffer
from vna_app.vision.tracker import RingTracker, Tip


class CameraError(RuntimeError):
    pass


class CameraStream(threading.Thread):
    def __init__(
        self,
        tracker: RingTracker,
        positions: PositionBuffer,
        device: str = "/dev/video4",
        width: int = 848,
        height: int = 480,
        fps: int = 30,
    ):
        super().__init__(daemon=True, name="camera")
        self.tracker = tracker
        self.positions = positions
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.error: Optional[str] = None  # consulté par le thread Qt (écriture atomique)
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Tuple[float, np.ndarray, Optional[Tip]]] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # -- appelé par le thread Qt AVANT start() (échec immédiat si caméra absente) ----
    def open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CameraError(f"caméra introuvable : {self.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise CameraError(f"caméra illisible : {self.device}")
        self._cap = cap

    def set_controls(self, auto: bool, exposure: int = 156, wb_temp: int = 4600):
        """Expo/WB auto ou figées. À figer pour que le seuil HSV du tracker tienne.

        V4L2 : AUTO_EXPOSURE 3 = auto, 1 = manuel ; EXPOSURE en unités de 100 µs.
        """
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto else 1)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto else 0)
        if not auto:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self._cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)

    # -- thread ----------------------------------------------------------------------
    def run(self):
        assert self._cap is not None, "open() d'abord"
        while not self._stopping.is_set():
            ok, bgr = self._cap.read()
            t = time.monotonic()
            if not ok:
                self.error = "lecture caméra échouée (débranchée ?)"
                break
            tip = self.tracker.find(bgr)
            if tip is not None:
                self.positions.push(t, tip.x, tip.y)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = (t, rgb, tip)
        self._cap.release()
        self._cap = None

    # -- appelés par le thread Qt ------------------------------------------------------
    def latest(self) -> Optional[Tuple[float, np.ndarray, Optional[Tip]]]:
        with self._lock:
            return self._latest

    def stop(self, timeout: float = 2.0):
        self._stopping.set()
        if self.is_alive():
            self.join(timeout)
        elif self._cap is not None:  # ouvert mais jamais démarré
            self._cap.release()
            self._cap = None
