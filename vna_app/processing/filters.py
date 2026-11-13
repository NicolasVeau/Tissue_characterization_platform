"""Lissage de spectres complexes et moyennage de sweeps."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_complex(x, window: int = 7, polyorder: int = 2):
    """Savitzky-Golay appliqué séparément à Re et Im.

    `window` est ramené au plus grand impair ≤ N si le spectre est trop court ;
    window <= 1 retourne x inchangé.
    """
    x = np.asarray(x, dtype=complex)
    if window <= 1:
        return x.copy()
    n = len(x)
    w = min(window, n if n % 2 else n - 1)
    if w % 2 == 0:
        w += 1 if w + 1 <= n else -1
    if w <= polyorder:
        return x.copy()
    return savgol_filter(x.real, w, polyorder) + 1j * savgol_filter(x.imag, w, polyorder)


def average_sweeps(sweeps):
    """Moyenne cohérente de plusieurs sweeps (liste de (N,) ou (N,2) complexes)."""
    arr = np.asarray(sweeps, dtype=complex)
    if arr.ndim < 2:
        raise ValueError("au moins deux sweeps attendus")
    return arr.mean(axis=0)
