"""LiteVNA Replay Source : rejoue un enregistrement de sweeps (vna_app.processing.record)."""

from __future__ import annotations

import time

import numpy as np
from gnuradio import gr

from gr_litevna import common
from vna_app.calibration import CalibrationError
from vna_app.processing.record import load_recording


class replay_source(gr.sync_block):
    """Rejoue un fichier .npz de sweeps avec les mêmes sorties, tags et PDU que
    LiteVNA Sweep Source : un flowgraph se développe et se compare hors banc.

    points    : doit être égal au nombre de points de l'enregistrement.
    cal_file  : .cal appliqué à la relecture (les enregistrements du CLI sont bruts).
    repeat    : reboucle en fin de fichier (sweep_idx continue de croître).
    realtime  : respecte les intervalles des horodatages enregistrés, divisés par
                `speed` ; sinon rejoue aussi vite que le flowgraph consomme.
    """

    def __init__(
        self, filename="", points=201, cal_file="", repeat=False, realtime=True, speed=1.0
    ):
        self.points = int(points)
        gr.sync_block.__init__(
            self,
            name="LiteVNA Replay Source",
            in_sig=None,
            out_sig=[(np.complex64, self.points), (np.complex64, self.points)],
        )
        self.filename = str(filename)
        self.cal_file = str(cal_file or "")
        self.repeat = bool(repeat)
        self.realtime = bool(realtime)
        self.speed = float(speed) if float(speed) > 0 else 1.0
        self.message_port_register_out(common.PORT_SWEEP)
        # Fichier lu et vérifié à la construction : une exception dans start()
        # bloquerait l'ordonnanceur de GNU Radio.
        rec = load_recording(self.filename)
        if rec.points != self.points:
            raise ValueError(
                f"{self.filename}: {rec.points} points, le bloc en attend {self.points}"
            )
        s = rec.s
        self._calibrated = bool(rec.meta.get("calibrated", False))
        cal = common.load_cal(self.cal_file)
        if cal is not None:
            if self._calibrated:
                raise CalibrationError(f"{self.filename} est déjà calibré : retirer cal_file")
            s = np.stack([cal.apply(rec.freqs, x) for x in s])
            self._calibrated = True
        self._rec, self._s = rec, s
        self._i = 0
        self._emitted = 0
        self._t_wall = None
        self._t_rec = None

    def set_speed(self, speed):
        self.speed = float(speed) if float(speed) > 0 else 1.0
        self._t_wall = None  # recale l'horloge de relecture

    def start(self):
        self._i = self._emitted = 0
        self._t_wall = None
        return True

    def _wait(self, ts):
        """Attend l'instant de relecture du sweep horodaté `ts` ; False si l'attente continue."""
        if not self.realtime or not np.isfinite(ts):
            return True
        now = time.monotonic()
        if self._t_wall is None or ts < self._t_rec:  # premier sweep, ou rebouclage
            self._t_wall, self._t_rec = now, ts
            return True
        due = self._t_wall + (ts - self._t_rec) / self.speed
        if due > now:
            time.sleep(min(due - now, 0.1))  # rend la main régulièrement pour l'arrêt
            return time.monotonic() >= due
        return True

    def work(self, input_items, output_items):
        rec = self._rec
        if rec is None:
            return -1
        if self._i >= len(rec):
            if not self.repeat or len(rec) == 0:
                return -1
            self._i = 0
        i = self._i
        ts = float(rec.timestamps[i])
        if not self._wait(ts):
            return 0
        s = self._s[i]
        output_items[0][0][:] = s[:, 0]
        output_items[1][0][:] = s[:, 1]
        seq = int(rec.seq[i]) if not self.repeat else self._emitted
        f0 = float(rec.freqs[0])
        step = float(rec.freqs[1] - rec.freqs[0]) if rec.points > 1 else 0.0
        common.add_sweep_tags(self, self.nitems_written(0), seq, ts, f0, step, self._calibrated)
        self.message_port_pub(
            common.PORT_SWEEP, common.sweep_pdu(seq, ts, f0, step, self._calibrated, s)
        )
        self._i += 1
        self._emitted += 1
        return 1
