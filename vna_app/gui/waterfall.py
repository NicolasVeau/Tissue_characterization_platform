"""Waterfall (fréquence × n° de sweep) avec échelle de couleurs calibrée (pyqtgraph)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF


class Waterfall(pg.PlotWidget):
    """Une ligne par sweep (le plus récent en haut), couleur = valeur, barre de couleurs à droite.

    Échelle : automatique (percentiles 1–99 % des données affichées) ou manuelle (lo, hi).
    """

    def __init__(
        self,
        title: str,
        label: str,
        depth: int = 100,
        cmap: str = "viridis",
        aux_label: Optional[str] = None,
        n_iso: int = 6,
    ):
        """aux_label : si donné, une 2e grandeur (aux) est tracée en isocontours par-dessus."""
        super().__init__(title=title)
        self.base_title = title
        self.depth = depth
        self.aux_label = aux_label
        self.n_iso = n_iso
        self.rows: list = []  # tableaux (N,) ; le plus ancien en premier
        self.aux_rows: list = []  # idem pour la grandeur tracée en isocontours
        self._iso: list = []
        self._iso_txt: Optional[pg.TextItem] = None  # liste des niveaux, dans le graphe
        self.first_seq = 0
        self.seq = 0
        self.freqs: Optional[np.ndarray] = None
        self.auto = True
        self.manual: Tuple[float, float] = (0.0, 1.0)
        self.levels: Tuple[float, float] = (0.0, 1.0)  # niveaux effectivement appliqués
        self._dirty = False  # rendu image/isocontours différé tant que le widget est masqué
        self.setLabel("bottom", "Fréquence", units="Hz")
        self.setLabel("left", "n° de sweep")
        self.img = pg.ImageItem(axisOrder="col-major")
        self.getPlotItem().addItem(self.img)
        self.bar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=pg.colormap.get(cmap),
            label=label,
            interactive=False,
            width=15,
        )
        self.bar.setImageItem(self.img, insert_in=self.getPlotItem())
        if aux_label is not None:
            self._iso_txt = pg.TextItem(color="w", anchor=(0, 0), fill=(0, 0, 0, 150))
            self._iso_txt.setZValue(20)
            self.addItem(self._iso_txt, ignoreBounds=True)
            self._iso_txt.setVisible(False)

    # -- données ---------------------------------------------------------------
    def clear_data(self):
        self.rows.clear()
        self.aux_rows.clear()
        self.freqs = None
        self.img.clear()
        self._clear_iso()

    def _clear_iso(self):
        if self._iso_txt is not None:
            self._iso_txt.setVisible(False)
        for it in self._iso:
            it.setParentItem(None)
            if it.scene() is not None:
                it.scene().removeItem(it)
        self._iso.clear()

    def push(self, freqs, values, aux=None):
        freqs = np.asarray(freqs, dtype=float)
        values = np.asarray(values, dtype=float)
        if (
            self.freqs is None
            or self.freqs.shape != freqs.shape
            or not np.allclose(self.freqs, freqs, rtol=0, atol=1.0)
        ):
            self.clear_data()  # autre grille de fréquences : on repart de zéro
            self.freqs = freqs
            self.first_seq = self.seq
        self.rows.append(values)
        if self.aux_label is not None:
            self.aux_rows.append(np.asarray(aux, dtype=float))
        self.seq += 1
        self._trim()
        self._render()

    def _trim(self):
        extra = len(self.rows) - self.depth
        if extra > 0:
            del self.rows[:extra]
            del self.aux_rows[:extra]
            self.first_seq = self.seq - len(self.rows)

    def set_depth(self, depth: int):
        self.depth = max(1, int(depth))
        if len(self.rows) > self.depth:
            self._trim()
            self._render()

    def set_n_iso(self, n: int):
        self.n_iso = max(0, int(n))
        self._render()

    # -- échelle de couleurs ---------------------------------------------------
    def set_auto(self, on: bool):
        self.auto = bool(on)
        self._render()

    def set_manual_levels(self, lo: float, hi: float):
        if hi > lo:
            self.manual = (float(lo), float(hi))
            self._render()

    def set_scale(self, lo: float, hi: float, auto: bool):
        """Échelle manuelle + mode auto en UN seul rendu (évite 2 rendus complets par sweep)."""
        if hi > lo:
            self.manual = (float(lo), float(hi))
        self.auto = bool(auto)
        self._render()

    def _compute_levels(self, arr):
        if not self.auto:
            return self.manual
        ok = arr[np.isfinite(arr)]
        if ok.size == 0:
            return self.levels
        lo, hi = np.percentile(ok, [1, 99])
        if hi <= lo:
            hi = lo + 1e-9 + abs(lo) * 1e-6
        return float(lo), float(hi)

    def _render(self):
        if not self.rows:
            self.img.clear()
            return
        arr = np.array(self.rows)  # (n, N)
        lo, hi = self.levels = self._compute_levels(arr)  # peu coûteux : toujours à jour
        if not self.isVisibleTo(self.window()):
            self._dirty = True  # image + isocontours (coûteux) seulement quand visible
            return
        self._dirty = False
        shown = np.where(np.isfinite(arr), arr, lo)  # non fini -> bas de l'échelle
        self.img.setImage(shown.T, autoLevels=False)  # col-major : (N freqs, n sweeps)
        self.bar.setLevels((lo, hi))
        f0, f1 = float(self.freqs[0]), float(self.freqs[-1])
        step = (f1 - f0) / max(len(self.freqs) - 1, 1)
        n = len(self.rows)
        self.img.setRect(QRectF(f0 - step / 2, self.first_seq, f1 - f0 + step, n))
        self.setXRange(f0, f1, padding=0)
        self.setYRange(self.first_seq, self.first_seq + n, padding=0)
        self._render_iso()

    def _render_iso(self):
        """Isocontours de la grandeur aux, dans les coordonnées de l'image (parent = img)."""
        self._clear_iso()
        if self.aux_label is None:
            return
        if self.n_iso < 1 or len(self.aux_rows) < 2:
            return
        a = np.array(self.aux_rows).T  # (N freqs, n sweeps), comme l'image col-major
        ok = np.isfinite(a)
        if not ok.any():
            return
        a = np.where(ok, a, np.median(a[ok]))
        a = pg.gaussianFilter(a, (1.0, 0.7))  # atténue le bruit avant de tracer les niveaux
        lo, hi = np.percentile(a, [5, 95])
        if hi <= lo:
            return
        levels = np.linspace(lo, hi, self.n_iso + 2)[1:-1]
        for lv in levels:
            it = pg.IsocurveItem(data=a, level=float(lv), pen=pg.mkPen("w", width=1))
            it.setParentItem(self.img)
            self._iso.append(it)
        vals = ", ".join(f"{lv:.3g}" for lv in levels)
        self._iso_txt.setText(f"isocontours {self.aux_label} : {vals}")
        n = len(self.rows)
        self._iso_txt.setPos(float(self.freqs[0]), self.first_seq + n)  # coin haut-gauche
        self._iso_txt.setVisible(True)

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            self._render()
