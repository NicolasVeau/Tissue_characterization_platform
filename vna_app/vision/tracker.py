"""Détection de la bague vert clair marquant l'extrémité de la canule.

Segmentation HSV (OpenCV : H ∈ [0, 179]) + ouverture morphologique + plus grande
composante connexe. Robuste seulement si l'exposition et la balance des blancs de la
caméra sont figées (voir vision.camera.CameraStream.set_controls) : en auto, la teinte
dérive et le seuil décroche.

Le centroïde de la bague n'est PAS l'extrémité de la canule : un décalage constant
(dx, dy) en pixels est appliqué côté GUI (canule ≈ perpendiculaire au tissu, caméra fixe).

Axe longitudinal : la gaine thermo bleu foncé (H ≈ 102–107) derrière la bague sert de
second repère. La direction gaine → bague est la projection image de l'axe de la
canule, orientée vers la pointe.
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
    ax: Optional[float] = None  # axe longitudinal unitaire (vers la pointe), ou None
    ay: Optional[float] = None


_KERNEL = np.ones((3, 3), np.uint8)


class RingTracker:
    """Les attributs sont modifiables à chaud (lus à chaque trame, écriture atomique GIL).

    ATTENTION : les teintes rendues par la D455 dépendent fortement de l'éclairage,
    de la balance des blancs et de l'exposition (mesuré au banc : la même bague passe
    de H ≈ 99 sous éclairage chaud surexposé à H ≈ 83 sous éclairage réduit). Les
    défauts ci-dessous correspondent au banc du 2026-11-24 (éclairage réduit, WB 4600 K,
    expo ≈ 80) : bague verte H 80–89 / S ≥ 100, gaine bleu sarcelle H 95–104.
    Les recaler avec les boutons « Mesurer » de l'onglet Carto après tout changement
    d'éclairage — c'est la mesure qui fait foi, pas ces constantes.

    s_min élimine les bleus-gris peu saturés que la WB figée crée dans les ombres
    (S ≤ ~95 mesuré). La gaine donne l'axe ; sheath_h_width = 0 le désactive.

    Verrouillage spatial : s'il existe une détection récente, le blob le plus PROCHE
    d'elle (dans lock_radius) est préféré au plus gros — un écran ou un objet coloré
    qui entre dans le champ ne vole plus le suivi. Sans détection récente (ou perdue
    depuis plus de lock_misses trames), retour au plus gros blob.
    """

    def __init__(
        self,
        h_center: float = 84.0,
        h_width: float = 6.0,
        s_min: int = 90,
        v_min: int = 40,
        min_area: float = 20.0,
        sheath_h_center: float = 99.0,
        sheath_h_width: float = 5.0,
        axis_min_dist: float = 8.0,
        axis_max_dist: float = 80.0,
        lock_radius: float = 120.0,
        lock_misses: int = 15,
    ):
        self.h_center = h_center
        self.h_width = h_width
        self.s_min = s_min
        self.v_min = v_min
        self.min_area = min_area
        self.sheath_h_center = sheath_h_center
        self.sheath_h_width = sheath_h_width
        self.axis_min_dist = axis_min_dist  # px : en deçà, direction trop bruitée
        self.axis_max_dist = axis_max_dist  # px : la gaine est ADJACENTE à la bague
        self.lock_radius = lock_radius  # px : rayon de validité de la dernière position
        self.lock_misses = lock_misses  # trames perdues avant de relâcher le verrou
        self._last = None  # dernière position de la bague (x, y)
        self._missed = 0

    def seed(self, x: float, y: float):
        """Amorce le verrou spatial (clic « Mesurer la bague » dans la GUI) : le suivi
        démarre sur CE blob même si un objet de la même teinte est plus gros ailleurs."""
        self._last = (float(x), float(y))
        self._missed = 0

    def band_mask(self, hsv, h_center, h_width):
        """Masque binaire de la bande HSV (seuils S/V communs). hsv = image HSV OpenCV."""
        lo = (max(0.0, h_center - h_width), self.s_min, self.v_min)
        hi = (min(179.0, h_center + h_width), 255, 255)
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)

    def _blobs(self, hsv, h_center, h_width, min_area):
        """[(aire, cx, cy), …] des composantes ≥ min_area de la bande."""
        mask = self.band_mask(hsv, h_center, h_width)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        out = []
        for i in range(1, n):  # 0 = fond
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                out.append((area, float(centroids[i][0]), float(centroids[i][1])))
        return out

    def _pick_ring(self, blobs):
        """Verrou posé : blob le plus proche s'il est dans le rayon, sinon RIEN (une
        trame sans bague ne doit pas téléporter le suivi sur un intrus plus gros).
        Verrou relâché : le plus gros blob."""
        if self._last is not None:
            lx, ly = self._last
            near = min(blobs, key=lambda b: (b[1] - lx) ** 2 + (b[2] - ly) ** 2)
            return near if np.hypot(near[1] - lx, near[2] - ly) <= self.lock_radius else None
        return max(blobs)

    def find(self, bgr: np.ndarray) -> Optional[Tip]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blobs = self._blobs(hsv, self.h_center, self.h_width, self.min_area)
        picked = self._pick_ring(blobs) if blobs else None
        if picked is None:
            self._missed += 1
            if self._missed > self.lock_misses:
                self._last = None
            return None
        area, cx, cy = picked
        self._last = (cx, cy)
        self._missed = 0
        ax = ay = None
        if self.sheath_h_width > 0:
            # la gaine est adjacente à la bague : blob de SA bande le plus proche de la
            # bague (jamais le plus gros : un objet de même teinte ailleurs fausserait l'axe)
            cand = self._blobs(hsv, self.sheath_h_center, self.sheath_h_width, self.min_area)
            if cand:
                _, sx, sy = min(cand, key=lambda b: (b[1] - cx) ** 2 + (b[2] - cy) ** 2)
                dx, dy = cx - sx, cy - sy  # gaine -> bague = vers la pointe
                d = float(np.hypot(dx, dy))
                if self.axis_min_dist <= d <= self.axis_max_dist:
                    ax, ay = dx / d, dy / d
        return Tip(cx, cy, area, ax, ay)
