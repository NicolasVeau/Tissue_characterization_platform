"""Vue cartographie : image caméra + chemin de la pointe coloré par ε' (pyqtgraph).

Repère = pixels de l'image (caméra fixe par rapport au tissu : aucun recalage).
Aucune I/O ici : le thread caméra remplit une boîte aux lettres, la GUI pousse
les trames via set_frame() et le chemin via set_path().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg

# Rampe séquentielle violet → magenta → rose (luminosité monotone : sombre = ε' faible).
# Choisie pour rester lisible SUR la vidéo : la famille magenta est absente de la scène
# (tissus beige/orangé, bague turquoise, gaine et tube bleus, embout doré) et ne recoupe
# pas les couleurs d'état de la croix (vert = suivie, rouge = perdue). viridis/plasma
# recoupent respectivement la bague turquoise et le fond beige.
EPS_CMAP = pg.ColorMap(
    pos=np.array([0.0, 0.35, 0.70, 1.0]),
    color=np.array(
        [(42, 10, 74, 255), (122, 31, 124, 255), (196, 62, 159, 255), (255, 179, 224, 255)],
        dtype=np.ubyte,
    ),
)


class ScanView(pg.GraphicsLayoutWidget):
    def __init__(self, cmap: Optional[str] = None):
        super().__init__()
        self.vb = self.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        # liseré sombre : sépare les points du fond quelle que soit la luminance locale
        self.path_pen = pg.mkPen((0, 0, 0, 180), width=1)
        self.path = pg.ScatterPlotItem(size=9, pen=self.path_pen, pxMode=True)
        self.path.setZValue(10)
        self.vb.addItem(self.path)
        self.tip_ok = pg.mkPen("#00ff00", width=2)
        self.tip_lost = pg.mkPen("r", width=2)
        self.tip = pg.ScatterPlotItem(size=18, symbol="+", brush=None, pen=self.tip_ok)
        self.tip.setZValue(11)
        self.vb.addItem(self.tip)
        self.cmap = EPS_CMAP if cmap is None else pg.colormap.get(cmap)
        self.bar = pg.ColorBarItem(
            values=(0, 1), colorMap=self.cmap, label="ε'", interactive=False, width=15
        )
        self.addItem(self.bar, row=0, col=1)
        self._has_frame = False

    def set_frame(self, rgb: np.ndarray):
        self.img.setImage(rgb, autoLevels=False, levels=[0, 255])
        if not self._has_frame:
            self._has_frame = True
            self.vb.autoRange(padding=0)

    def set_tip(self, tip: Optional[object]):
        """tip : vision.tracker.Tip ou None (bague perdue : la croix reste, en rouge)."""
        if tip is None:
            self.tip.setPen(self.tip_lost)
        else:
            self.tip.setPen(self.tip_ok)
            self.tip.setData([tip.x], [tip.y])

    def set_path(self, x, y, values, lo: float, hi: float):
        """Points du chemin colorés par `values` (ε') entre lo et hi."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            self.path.setData([], [])
            return
        span = (hi - lo) or 1.0
        frac = np.clip((np.asarray(values, dtype=float) - lo) / span, 0.0, 1.0)
        colors = self.cmap.map(frac, mode="byte")
        self.path.setData(
            x=x,
            y=np.asarray(y, dtype=float),
            brush=[pg.mkBrush(*c) for c in colors],
            pen=self.path_pen,
        )
        self.bar.setLevels((lo, hi))

    def set_point_size(self, px: int):
        self.path.setSize(int(px))

    def clear_path(self):
        self.path.setData([], [])
