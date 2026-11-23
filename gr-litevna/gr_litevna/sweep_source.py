"""LiteVNA Sweep Source : un item = un sweep complet (vecteur complexe de N points)."""

from __future__ import annotations

import queue
import threading

import numpy as np
import pmt
from gnuradio import gr

from gr_litevna import common
from vna_app.calibration import CalibrationError
from vna_app.driver.litevna import LiteVNAError
from vna_app.driver.sim import open_device


class sweep_source(gr.sync_block):
    """Source de sweeps S11/S21 du LiteVNA 64.

    Sorties : deux flux de vecteurs complex64 de longueur `points` (S11, S21).
    Chaque sweep porte sur la sortie 0 les tags rx_time, sweep_idx, f_start, f_step
    et calibrated ; il est aussi publié en PDU sur le port message `sweep`.
    Le port message `cmd` accepte un dictionnaire {f_start, f_stop, averaging} pour
    changer la bande à chaud ; `points` est fixé à la construction (longueur des
    vecteurs).

    port      : /dev/litevna, ou "sim[:options]" pour le simulateur (vna_app.driver.sim)
    cal_file  : fichier .cal (vna_app.calibration) ; vide = données brutes. La grille
                du .cal doit être celle du sweep, sinon la construction du bloc
                est refusée (CalibrationError).
                Après un changement de bande à chaud, la sortie repasse en brut
                (tag calibrated = False) tant que la grille ne correspond plus.

    Le débit est celui de l'appareil (~300 points/s) : si le flowgraph est plus lent,
    le driver écarte les sweeps les plus anciens ; les trous se voient dans sweep_idx.

    Si l'appareil est absent ou refuse la configuration, l'erreur est journalisée,
    gardée dans `self.error`, et le flowgraph se termine (une exception dans start()
    bloquerait l'ordonnanceur de GNU Radio).
    """

    def __init__(
        self,
        port="/dev/litevna",
        f_start=400e6,
        f_stop=3e9,
        points=201,
        averaging=1,
        cal_file="",
    ):
        self.points = int(points)
        if self.points < 2:
            raise ValueError("points doit être >= 2 (fréquence fixe : bloc CW Source)")
        gr.sync_block.__init__(
            self,
            name="LiteVNA Sweep Source",
            in_sig=None,
            out_sig=[(np.complex64, self.points), (np.complex64, self.points)],
        )
        self.port = str(port)
        self.f_start, self.f_stop = float(f_start), float(f_stop)
        self.averaging = int(averaging)
        self.cal_file = str(cal_file or "")
        self._cal = common.load_cal(self.cal_file)
        if self._cal is not None:  # refus dès la construction, avec un message explicite
            self._cal._check_grid(common.sweep_grid(self.f_start, self.f_stop, self.points))
        self._cal_ok = False
        self.error = None
        self._vna = None
        self._q = None
        self._lock = threading.Lock()
        self._pending = None  # (f_start, f_stop, averaging) demandé à chaud
        self.message_port_register_out(common.PORT_SWEEP)
        self.message_port_register_in(common.PORT_CMD)
        self.set_msg_handler(common.PORT_CMD, self._handle_cmd)

    # -- réglages à chaud (callbacks GRC, thread quelconque) ----------------------
    def set_f_start(self, f_start):
        self._request(f_start=float(f_start))

    def set_f_stop(self, f_stop):
        self._request(f_stop=float(f_stop))

    def set_averaging(self, averaging):
        self._request(averaging=int(averaging))

    def _request(self, f_start=None, f_stop=None, averaging=None):
        with self._lock:
            cur = self._pending or (self.f_start, self.f_stop, self.averaging)
            self._pending = (
                cur[0] if f_start is None else f_start,
                cur[1] if f_stop is None else f_stop,
                cur[2] if averaging is None else averaging,
            )

    def _handle_cmd(self, msg):
        if not pmt.is_dict(msg):
            common.log_warn("LiteVNA Sweep Source: message cmd ignoré (dictionnaire attendu)")
            return
        d = pmt.to_python(msg)
        if "points" in d and int(d["points"]) != self.points:
            common.log_warn("LiteVNA Sweep Source: points n'est pas modifiable à chaud, ignoré")
        self._request(
            f_start=float(d["f_start"]) if "f_start" in d else None,
            f_stop=float(d["f_stop"]) if "f_stop" in d else None,
            averaging=int(d["averaging"]) if "averaging" in d else None,
        )

    # -- cycle de vie ---------------------------------------------------------------
    def start(self):
        self.error = None
        try:
            self._vna = open_device(self.port)
            self._vna.connect()
            freqs = self._vna.configure_sweep(
                self.f_start, self.f_stop, self.points, self.averaging
            )
            self._cal_ok = common.cal_matches(self._cal, freqs)
            if self._cal is not None and not self._cal_ok:
                raise CalibrationError("grille programmée différente de celle du .cal")
            self._q = self._vna.start_streaming()
        except (LiteVNAError, CalibrationError, ValueError) as e:
            self._close()
            self.error = str(e)
            common.log_error(f"LiteVNA Sweep Source: démarrage impossible: {e}")
        return True

    def stop(self):
        self._close()
        return True

    def _close(self):
        vna, self._vna, self._q = self._vna, None, None
        if vna is not None:
            try:
                vna.close()  # arrête le streaming puis libère le port
            except LiteVNAError as e:
                common.log_warn(f"LiteVNA Sweep Source: fermeture: {e}")

    def _apply_pending(self):
        with self._lock:
            req, self._pending = self._pending, None
        if req is None or req == (self.f_start, self.f_stop, self.averaging):
            return
        # le driver impose : arrêt du streaming, reconfiguration, relance
        self._vna.stop_streaming()
        try:
            freqs = self._vna.configure_sweep(req[0], req[1], self.points, req[2])
            self.f_start, self.f_stop, self.averaging = req
        except (LiteVNAError, ValueError) as e:
            common.log_error(
                f"LiteVNA Sweep Source: reconfiguration refusée ({e}), bande inchangée"
            )
            freqs = self._vna.configure_sweep(
                self.f_start, self.f_stop, self.points, self.averaging
            )
        was_ok = self._cal_ok
        self._cal_ok = common.cal_matches(self._cal, freqs)
        if was_ok and not self._cal_ok:
            common.log_warn("LiteVNA Sweep Source: grille différente du .cal, sortie en BRUT")
        self._q = self._vna.start_streaming()

    # -- traitement -----------------------------------------------------------------
    def work(self, input_items, output_items):
        if self._vna is None or self._q is None:
            return -1
        try:
            self._apply_pending()
            item = self._q.get(timeout=0.1)
        except queue.Empty:
            return 0
        except LiteVNAError as e:
            common.log_error(f"LiteVNA Sweep Source: {e}")
            return -1
        if isinstance(item, Exception):
            common.log_error(f"LiteVNA Sweep Source: {item}")
            return -1
        if item.s_raw.shape[0] != self.points:
            return 0  # sweep d'une configuration précédente
        s = self._cal.apply(item.freqs, item.s_raw) if self._cal_ok else item.s_raw
        output_items[0][0][:] = s[:, 0]
        output_items[1][0][:] = s[:, 1]
        f0 = float(item.freqs[0])
        step = float(item.freqs[1] - item.freqs[0])
        common.add_sweep_tags(
            self, self.nitems_written(0), item.seq, item.timestamp, f0, step, self._cal_ok
        )
        self.message_port_pub(
            common.PORT_SWEEP,
            common.sweep_pdu(item.seq, item.timestamp, f0, step, self._cal_ok, s),
        )
        return 1
