"""Détection de la bague vert clair marquant l'extrémité de la canule.

Segmentation HSV (OpenCV : H ∈ [0, 179]) + ouverture morphologique + plus grande
composante connexe. Robuste seulement si l'exposition et la balance des blancs de la
caméra sont figées (voir vision.camera.CameraStream.set_controls) : en auto, la teinte
dérive et le seuil décroche.

Le centroïde de la bague n'est PAS l'extrémité de la canule : un décalage constant
(dx, dy) en pixels est appliqué côté GUI (canule ≈ perpendiculaire au tissu, caméra fixe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Tip:
    x: float  # colonne (px)
    y: float  # ligne (px)
    area: float  # aire de la composante (px²)


_KERNEL = np.ones((3, 3), np.uint8)


class RingTracker:
    """Les attributs sont modifiables à chaud (lus à chaque trame, écriture atomique GIL)."""

    def __init__(
        self,
        h_center: float = 60.0,
        h_width: float = 12.0,
        s_min: int = 70,
        v_min: int = 70,
        min_area: float = 40.0,
    ):
        self.h_center = h_center
        self.h_width = h_width
        self.s_min = s_min
        self.v_min = v_min
        self.min_area = min_area

    def find(self, bgr: np.ndarray) -> Optional[Tip]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lo = (max(0.0, self.h_center - self.h_width), self.s_min, self.v_min)
        hi = (min(179.0, self.h_center + self.h_width), 255, 255)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        best, best_area = None, self.min_area
        for i in range(1, n):  # 0 = fond
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area >= best_area:
                best, best_area = i, area
        if best is None:
            return None
        cx, cy = centroids[best]
        return Tip(float(cx), float(cy), best_area)
