"""LiteVNA CW Source : fréquence fixe, flux scalaires S11/S21 (mode zero-span)."""

from __future__ import annotations

import queue
import threading

import numpy as np
import pmt
from gnuradio import gr

from gr_litevna import common
from vna_app.driver.litevna import LiteVNAError
from vna_app.driver.sim import open_device


class cw_source(gr.sync_block):
    """Source S11/S21 à fréquence fixe.

    Le LiteVNA est configuré avec start = stop = `freq` et `block_len` points
    (pas de fréquence nul) : chaque acquisition livre `block_len` échantillons
    successifs à la même fréquence, à la cadence de l'appareil (~300/s, divisée
    par `averaging`).

    Sorties : deux flux scalaires complex64 (S11, S21), BRUTS : la calibration du
    dépôt est définie sur une grille de sweep, pas à une fréquence isolée.
    Tags sur la sortie 0, au premier échantillon de chaque acquisition : rx_time
    (heure de FIN de l'acquisition), sweep_idx, freq.

    L'échantillonnage est régulier à l'intérieur d'une acquisition ; entre deux
    acquisitions le driver relance le balayage, ce qui laisse un court trou.
    Choisir `block_len` grand pour espacer ces trous, petit pour réduire la latence.
    Le port message `cmd` accepte {freq, averaging}.
    """

    def __init__(self, port="/dev/litevna", freq=1e9, block_len=256, averaging=1):
        gr.sync_block.__init__(
            self,
            name="LiteVNA CW Source",
            in_sig=None,
            out_sig=[np.complex64, np.complex64],
        )
        self.port = str(port)
        self.freq = float(freq)
        self.block_len = int(block_len)
        if self.block_len < 1:
            raise ValueError("block_len doit être >= 1")
        self.averaging = int(averaging)
        self._vna = None
        self._q = None
        self._buf = np.zeros((0, 2), dtype=complex)
        self._buf_tag = None  # (seq, timestamp) à poser sur le premier échantillon restant
        self._lock = threading.Lock()
        self._pending = None
        self.error = None
        self.message_port_register_in(common.PORT_CMD)
        self.set_msg_handler(common.PORT_CMD, self._handle_cmd)

    def set_freq(self, freq):
        with self._lock:
            cur = self._pending or (self.freq, self.averaging)
            self._pending = (float(freq), cur[1])

    def set_averaging(self, averaging):
        with self._lock:
            cur = self._pending or (self.freq, self.averaging)
            self._pending = (cur[0], int(averaging))

    def _handle_cmd(self, msg):
        if not pmt.is_dict(msg):
            common.log_warn("LiteVNA CW Source: message cmd ignoré (dictionnaire attendu)")
            return
        d = pmt.to_python(msg)
        if "freq" in d:
            self.set_freq(d["freq"])
        if "averaging" in d:
            self.set_averaging(d["averaging"])

    def start(self):
        self.error = None
        try:
            self._vna = open_device(self.port)
            self._vna.connect()
            self._vna.configure_sweep(self.freq, self.freq, self.block_len, self.averaging)
            self._q = self._vna.start_streaming()
        except (LiteVNAError, ValueError) as e:  # ne pas lever dans start() : voir sweep_source
            self._close()
            self.error = str(e)
            common.log_error(f"LiteVNA CW Source: démarrage impossible: {e}")
        return True

    def stop(self):
        self._close()
        return True

    def _close(self):
        vna, self._vna, self._q = self._vna, None, None
        if vna is not None:
            try:
                vna.close()
            except LiteVNAError as e:
                common.log_warn(f"LiteVNA CW Source: fermeture: {e}")

    def _apply_pending(self):
        with self._lock:
            req, self._pending = self._pending, None
        if req is None or req == (self.freq, self.averaging):
            return
        self._vna.stop_streaming()
        try:
            self._vna.configure_sweep(req[0], req[0], self.block_len, req[1])
            self.freq, self.averaging = req
        except (LiteVNAError, ValueError) as e:
            common.log_error(f"LiteVNA CW Source: reconfiguration refusée ({e}), inchangée")
            self._vna.configure_sweep(self.freq, self.freq, self.block_len, self.averaging)
        self._buf = self._buf[:0]  # les échantillons de l'ancienne fréquence sont jetés
        self._buf_tag = None
        self._q = self._vna.start_streaming()

    def work(self, input_items, output_items):
        if self._vna is None or self._q is None:
            return -1
        if len(self._buf) == 0:
            try:
                self._apply_pending()
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                return 0
            except LiteVNAError as e:
                common.log_error(f"LiteVNA CW Source: {e}")
                return -1
            if isinstance(item, Exception):
                common.log_error(f"LiteVNA CW Source: {item}")
                return -1
            self._buf = item.s_raw
            self._buf_tag = (item.seq, item.timestamp, float(item.freqs[0]))
        n = min(len(output_items[0]), len(self._buf))
        output_items[0][:n] = self._buf[:n, 0]
        output_items[1][:n] = self._buf[:n, 1]
        if self._buf_tag is not None:
            seq, ts, f = self._buf_tag
            off = self.nitems_written(0)
            self.add_item_tag(0, off, common.KEY_RX_TIME, common.rx_time_pmt(ts))
            self.add_item_tag(0, off, common.KEY_SWEEP_IDX, pmt.from_long(int(seq)))
            self.add_item_tag(0, off, common.KEY_FREQ, pmt.from_double(f))
            self._buf_tag = None
        self._buf = self._buf[n:]
        return n
