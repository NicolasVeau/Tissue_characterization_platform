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


def _cmap(pos, colors):
    return pg.ColorMap(
        pos=np.asarray(pos, dtype=float),
        color=np.array([c + (255,) for c in colors], dtype=np.ubyte),
    )


# Palettes embarquées (pas de matplotlib dans la GUI ; pyqtgraph 0.12.4 ne fournit
# que les CET). turbo/viridis/plasma = points d'ancrage échantillonnés des originaux.
# turbo = « arc-en-ciel » moderne : différenciation maximale, mais luminosité non
# monotone et teintes qui recoupent la scène (turquoise de la bague, rouge de la croix).
CMAPS = {
    "magenta": EPS_CMAP,
    "turbo": _cmap(
        [0.0, 0.07, 0.14, 0.21, 0.29, 0.36, 0.43, 0.50,
         0.57, 0.64, 0.71, 0.79, 0.86, 0.93, 1.0],
        [(48, 18, 59), (65, 69, 171), (70, 117, 237), (57, 162, 252), (27, 207, 212),
         (36, 236, 166), (97, 252, 108), (164, 252, 59), (209, 232, 52), (243, 198, 58),
         (254, 155, 45), (243, 99, 21), (217, 56, 6), (177, 25, 1), (122, 4, 2)],
    ),
    "viridis": _cmap(
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        [(68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141),
         (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)],
    ),
    "plasma": _cmap(
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        [(13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150), (203, 70, 121),
         (229, 107, 93), (248, 148, 65), (253, 195, 40), (240, 249, 33)],
    ),
}


class ScanView(pg.GraphicsLayoutWidget):
    clicked = pg.QtCore.Signal(float, float)  # clic dans l'image, en pixels image

    def __init__(self, cmap: Optional[str] = None):
        super().__init__()
        self.vb = self.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.scene().sigMouseClicked.connect(self._on_click)
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
        # axe longitudinal estimé (gaine → bague), prolongé au-delà de la pointe : bleu
        self.axis_line = pg.PlotCurveItem(pen=pg.mkPen("#3070ff", width=2))
        self.axis_line.setZValue(12)
        self.vb.addItem(self.axis_line)
        # normale à la surface (projection image) : turquoise + inclinaison en texte
        self.normal_line = pg.PlotCurveItem(pen=pg.mkPen("#40e0d0", width=2))
        self.normal_line.setZValue(13)
        self.vb.addItem(self.normal_line)
        self.normal_text = pg.TextItem(color="#40e0d0", anchor=(0, 1))
        self.normal_text.setZValue(14)
        self.vb.addItem(self.normal_text)
        self.normal_text.setVisible(False)
        self.cmap = EPS_CMAP if cmap is None else pg.colormap.get(cmap)
        self.bar = pg.ColorBarItem(
            values=(0, 1), colorMap=self.cmap, label="ε'", interactive=False, width=15
        )
        self.addItem(self.bar, row=0, col=1)
        self._has_frame = False

    def _on_click(self, ev):
        if ev.button() != pg.QtCore.Qt.LeftButton:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        self.clicked.emit(float(p.x()), float(p.y()))

    def set_frame(self, rgb: np.ndarray, full_wh=None):
        """full_wh = (W, H) d'origine si `rgb` est sous-échantillonné pour l'affichage :
        l'image est étirée sur le repère pleine résolution, donc les overlays (croix,
        chemin, normale) et les clics restent en pixels pleine résolution."""
        self.img.setImage(rgb, autoLevels=False, levels=[0, 255])
        h, w = rgb.shape[:2]
        fw, fh = full_wh if full_wh is not None else (w, h)
        self.img.setRect(pg.QtCore.QRectF(0.0, 0.0, float(fw), float(fh)))
        if not self._has_frame:
            self._has_frame = True
            self.vb.autoRange(padding=0)

    def set_tip(self, tip: Optional[object]):
        """tip : vision.tracker.Tip ou None (bague perdue : la croix reste, en rouge)."""
        if tip is None:
            self.tip.setPen(self.tip_lost)
            self.axis_line.setData([], [])
        else:
            self.tip.setPen(self.tip_ok)
            self.tip.setData([tip.x], [tip.y])
            ax, ay = getattr(tip, "ax", None), getattr(tip, "ay", None)
            if ax is None or ay is None:
                self.axis_line.setData([], [])
            else:  # de la gaine (−60 px) jusqu'au-delà de la pointe (+25 px)
                self.axis_line.setData(
                    [tip.x - 60 * ax, tip.x + 25 * ax], [tip.y - 60 * ay, tip.y + 25 * ay]
                )

    def set_normal(self, x: float, y: float, normal, length: float = 55.0):
        """Projette la normale 3D (unitaire, n_z < 0) en flèche image depuis (x, y).

        normal = None efface (profondeur indisponible ou pointe perdue).
        """
        if normal is None:
            self.normal_line.setData([], [])
            self.normal_text.setVisible(False)
            return
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
        tilt = float(np.degrees(np.arccos(min(1.0, abs(nz)))))
        x1, y1 = x + length * nx, y + length * ny
        self.normal_line.setData([x, x1], [y, y1])
        self.normal_text.setText(f"{tilt:.0f}°")
        self.normal_text.setPos(x1, y1)
        self.normal_text.setVisible(True)

    def set_cmap(self, name: str):
        """Change la palette du chemin ; le chemin est recoloré au prochain set_path()."""
        self.cmap = CMAPS.get(name, EPS_CMAP)
        self.bar.setColorMap(self.cmap)

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
