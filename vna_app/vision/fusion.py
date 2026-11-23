"""Fusion temporelle caméra ↔ VNA et carte des propriétés diélectriques.

Les deux flux sont horodatés sur la même horloge (time.monotonic) : la caméra à ~30 Hz
(PositionBuffer), le VNA à quelques mesures/s. Chaque mesure ε* reçoit la position de la
pointe interpolée à son instant (milieu du sweep). Pas de dépendance Qt ni OpenCV ici.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np


class PositionBuffer:
    """Positions (t, x, y) de la pointe, thread-safe ; interpolation linéaire en t.

    Au-delà de max_age (s) du dernier échantillon voisin, at() renvoie None :
    une mesure sans position contemporaine n'est pas cartographiée.
    """

    def __init__(self, maxlen: int = 600, max_age: float = 0.5):
        self.max_age = max_age
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, t: float, x: float, y: float):
        with self._lock:
            self._buf.append((float(t), float(x), float(y)))

    def clear(self):
        with self._lock:
            self._buf.clear()

    def at(self, t: float) -> Optional[Tuple[float, float]]:
        with self._lock:
            pts = list(self._buf)
        if not pts:
            return None
        before = after = None
        for p in pts:  # trié par construction (push à cadence caméra)
            if p[0] <= t:
                before = p
            elif after is None:
                after = p
                break
        if before is not None and after is not None:
            dt = after[0] - before[0]
            if dt > 2 * self.max_age:  # trou de suivi : ne pas inventer de trajectoire
                near = before if t - before[0] <= after[0] - t else after
                return (near[1], near[2]) if abs(near[0] - t) <= self.max_age else None
            a = 0.0 if dt <= 0 else (t - before[0]) / dt
            return (before[1] + a * (after[1] - before[1]), before[2] + a * (after[2] - before[2]))
        near = before or after
        if abs(near[0] - t) > self.max_age:
            return None
        return (near[1], near[2])


def summarize_eps(freqs, eps, f_lo: float, f_hi: float) -> Tuple[float, float]:
    """(ε', ε'') médians sur [f_lo, f_hi] — la valeur scalaire affichée/cartographiée.

    Convention ε* = ε' − jε''. NaN, NaN si rien d'exploitable dans la bande.
    """
    freqs = np.asarray(freqs, dtype=float)
    eps = np.asarray(eps, dtype=complex)
    m = (freqs >= f_lo) & (freqs <= f_hi) & np.isfinite(eps)
    if not m.any():
        return float("nan"), float("nan")
    return float(np.median(eps.real[m])), float(np.median(-eps.imag[m]))


class ScanMap:
    """Échantillons (t, x, y, ε', ε'') accumulés pendant le contact ; export CSV."""

    def __init__(self):
        self.t: list = []
        self.x: list = []
        self.y: list = []
        self.e1: list = []
        self.e2: list = []

    def __len__(self):
        return len(self.t)

    def add(self, t: float, x: float, y: float, e1: float, e2: float):
        self.t.append(float(t))
        self.x.append(float(x))
        self.y.append(float(y))
        self.e1.append(float(e1))
        self.e2.append(float(e2))

    def clear(self):
        for v in (self.t, self.x, self.y, self.e1, self.e2):
            v.clear()

    def arrays(self):
        return (np.asarray(self.t), np.asarray(self.x), np.asarray(self.y),
                np.asarray(self.e1), np.asarray(self.e2))

    def save_csv(self, path):
        t, x, y, e1, e2 = self.arrays()
        np.savetxt(
            path,
            np.column_stack([t, x, y, e1, e2]),
            delimiter=",",
            header="t_s,x_px,y_px,eps_real,eps_loss",
            comments="",
            fmt="%.9g",
        )
