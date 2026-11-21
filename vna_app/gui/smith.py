"""Abaque de Smith (pyqtgraph) : grille r/x constants + tracé de Γ = S11."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

R_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)
X_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)
_BIG = 1e4  # « infini » pour l'échantillonnage des cercles


def _gamma(z):
    return (z - 1) / (z + 1)


def r_circle(r, n=400):
    """Cercle r = cste (Γ à partir de z = r + jx, x ∈ [-∞, ∞])."""
    x = np.sinh(np.linspace(-np.arcsinh(_BIG), np.arcsinh(_BIG), n))
    return _gamma(r + 1j * x)


def x_arc(x, n=400):
    """Arc x = cste (z = r + jx, r ∈ [0, ∞])."""
    r = np.sinh(np.linspace(0, np.arcsinh(_BIG), n))
    return _gamma(r + 1j * x)


class SmithChart(pg.PlotWidget):
    def __init__(self, title="Smith (S11)"):
        super().__init__(title=title)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0)
        self._draw_grid()
        self.trace = self.plot(pen=pg.mkPen("y", width=2))
        self.start_pt = self.plot(
            symbol="o", symbolBrush="g", symbolSize=8, pen=None
        )  # premier point
        self.stop_pt = self.plot(symbol="s", symbolBrush="r", symbolSize=8, pen=None)

    def _draw_grid(self):
        grid = pg.mkPen((120, 120, 120), width=1)
        t = np.linspace(0, 2 * np.pi, 400)
        self.plot(np.cos(t), np.sin(t), pen=pg.mkPen((200, 200, 200), width=2))
        self.plot([-1, 1], [0, 0], pen=grid)
        for r in R_VALUES:
            g = r_circle(r)
            self.plot(g.real, g.imag, pen=grid)
            lab = pg.TextItem(f"{r:g}", color=(150, 150, 150), anchor=(0.5, 1.0))
            lab.setPos((r - 1) / (r + 1), 0)
            self.addItem(lab)
        for x in X_VALUES:
            for sgn in (1, -1):
                g = x_arc(sgn * x)
                self.plot(g.real, g.imag, pen=grid)

    def set_gamma(self, gamma):
        gamma = np.asarray(gamma, dtype=complex)
        if gamma.size == 0:
            self.trace.setData([], [])
            self.start_pt.setData([], [])
            self.stop_pt.setData([], [])
            return
        self.trace.setData(gamma.real, gamma.imag)
        self.start_pt.setData([gamma[0].real], [gamma[0].imag])
        self.stop_pt.setData([gamma[-1].real], [gamma[-1].imag])
