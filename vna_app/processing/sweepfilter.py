"""Rejet des sweeps incohérents d'après le coefficient de réflexion aux extrémités.

Critère (sur S11, calibré si la calibration est appliquée) :
  1. d_min ≤ |Γ(f_début) − Γ(f_fin)| ≤ d_max ;
  2. chaque point intermédiaire est dans le voisinage d'une des deux extrémités :
     min(|Γ_k − Γ_début|, |Γ_k − Γ_fin|) ≤ rayon.
Un sweep qui ne respecte pas ces conditions est rejeté (non affiché, non empilé).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SweepFilter:
    enabled: bool = False
    d_min: float = 0.05
    d_max: float = 2.0
    radius: float = 0.5
    accepted: int = 0
    rejected: int = 0

    def reset_counts(self):
        self.accepted = self.rejected = 0

    def reason(self, gamma) -> Optional[str]:
        """None si le sweep est conforme, sinon le motif du rejet."""
        g = np.asarray(gamma, dtype=complex)
        if g.size < 2 or not np.all(np.isfinite(g)):
            return "valeurs non finies"
        g0, g1 = g[0], g[-1]
        d = abs(g1 - g0)
        if d < self.d_min:
            return f"|Γ_fin−Γ_début| = {d:.3f} < {self.d_min:.3f}"
        if d > self.d_max:
            return f"|Γ_fin−Γ_début| = {d:.3f} > {self.d_max:.3f}"
        dist = np.minimum(np.abs(g - g0), np.abs(g - g1))
        k = int(np.argmax(dist))
        if dist[k] > self.radius:
            return f"point {k} à {dist[k]:.3f} des extrémités (> {self.radius:.3f})"
        return None

    def check(self, gamma) -> Tuple[bool, Optional[str]]:
        """Met à jour les compteurs. Retourne (accepté, motif)."""
        if not self.enabled:
            return True, None
        why = self.reason(gamma)
        if why is None:
            self.accepted += 1
        else:
            self.rejected += 1
        return why is None, why
