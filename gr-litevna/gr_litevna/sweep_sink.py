"""LiteVNA Sweep Sink : enregistre un flux de sweeps au format vna_app.processing.record."""

from __future__ import annotations

import time

import numpy as np
import pmt
from gnuradio import gr

from gr_litevna import common
from vna_app.processing.record import RecordError, SweepRecorder


class sweep_sink(gr.sync_block):
    """Enregistre S11/S21 (deux flux de vecteurs complex64) dans un fichier .npz
    relisible par LiteVNA Replay Source et par vna_app.processing.record.

    Les tags rx_time, sweep_idx, f_start, f_step et calibrated posés par les blocs
    source sont repris ; à défaut, f_start / f_step des paramètres et l'heure de
    réception sont utilisés. Le fichier est écrit à l'arrêt du flowgraph.

    Les échantillons traversent GNU Radio en complex64 : l'enregistrement est exact
    à cette précision (pour un enregistrement brut en double précision, utiliser
    `python -m vna_app.cli record`).
    """

    def __init__(self, filename="sweeps.npz", points=201, f_start=400e6, f_step=13e6):
        self.points = int(points)
        gr.sync_block.__init__(
            self,
            name="LiteVNA Sweep Sink",
            in_sig=[(np.complex64, self.points), (np.complex64, self.points)],
            out_sig=None,
        )
        self.filename = str(filename)
        self.f_start, self.f_step = float(f_start), float(f_step)
        self._rec = None
        self._count = 0
        self._calibrated = None
        self._warned = False

    def start(self):
        self._rec = SweepRecorder(self.filename, {"source": "gr-litevna sweep_sink"})
        self._count = 0
        self._calibrated = None
        return True

    def stop(self):
        rec, self._rec = self._rec, None
        if rec is not None:
            rec.meta.update(
                {
                    "points": self.points,
                    "start_hz": self.f_start,
                    "step_hz": self.f_step,
                    "calibrated": bool(self._calibrated),
                    "precision": "complex64",
                }
            )
            path = rec.close()
            if path is not None:
                common.log_info(f"LiteVNA Sweep Sink: {len(rec)} sweep(s) écrits dans {path}")
        return True

    def work(self, input_items, output_items):
        n = len(input_items[0])
        first = self.nitems_read(0)
        tags = {}
        for t in self.get_tags_in_window(0, 0, n):
            tags.setdefault(t.offset - first, {})[pmt.symbol_to_string(t.key)] = t.value
        now = time.time()
        for i in range(n):
            tg = tags.get(i, {})
            if "f_start" in tg:
                self.f_start = pmt.to_double(tg["f_start"])
            if "f_step" in tg:
                self.f_step = pmt.to_double(tg["f_step"])
            if "calibrated" in tg:
                self._calibrated = pmt.to_bool(tg["calibrated"])
            seq = pmt.to_long(tg["sweep_idx"]) if "sweep_idx" in tg else self._count
            ts = common.rx_time_float(tg["rx_time"]) if "rx_time" in tg else now
            freqs = self.f_start + np.arange(self.points) * self.f_step
            s = np.stack([input_items[0][i], input_items[1][i]], axis=1)
            try:
                self._rec.add(freqs, s, seq=seq, timestamp=ts)
            except RecordError as e:  # bande changée en cours d'enregistrement
                if not self._warned:
                    common.log_error(f"LiteVNA Sweep Sink: sweeps ignorés ({e})")
                    self._warned = True
                continue
            self._count += 1
        return n
