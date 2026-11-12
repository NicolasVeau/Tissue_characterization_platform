"""Kits d'étalonnage : coefficients de réflexion théoriques des standards.

Un kit retourne, pour une grille de fréquences (Hz), le Γ de chaque standard.
Le kit idéal est la première approche (les standards fournis ne sont pas
caractérisés) ; un kit caractérisé pourra le remplacer sans toucher à `solt`.
"""

from __future__ import annotations

import numpy as np


class Kit:
    def open(self, freqs):  # pragma: no cover - interface
        raise NotImplementedError

    def short(self, freqs):  # pragma: no cover
        raise NotImplementedError

    def load(self, freqs):  # pragma: no cover
        raise NotImplementedError


class IdealKit(Kit):
    """Open = +1, Short = -1, Load = 0 (50 Ω parfaite), Thru S21 = 1."""

    name = "ideal"

    def open(self, freqs):
        return np.ones(len(freqs), dtype=complex)

    def short(self, freqs):
        return -np.ones(len(freqs), dtype=complex)

    def load(self, freqs):
        return np.zeros(len(freqs), dtype=complex)
