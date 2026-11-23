"""Utilitaires communs aux blocs : tags, PDU, chargement de calibration."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pmt
from gnuradio import gr

from vna_app.calibration import Calibration, CalibrationError

KEY_RX_TIME = pmt.intern("rx_time")
KEY_SWEEP_IDX = pmt.intern("sweep_idx")
KEY_F_START = pmt.intern("f_start")
KEY_F_STEP = pmt.intern("f_step")
KEY_CALIBRATED = pmt.intern("calibrated")
KEY_FREQ = pmt.intern("freq")
PORT_SWEEP = pmt.intern("sweep")
PORT_CMD = pmt.intern("cmd")

try:
    _LOG = gr.logger("gr_litevna")
except Exception:  # pragma: no cover - versions sans gr.logger côté Python
    import logging

    _LOG = logging.getLogger("gr_litevna")


def log_info(msg: str) -> None:
    _LOG.info(msg)


def log_warn(msg: str) -> None:
    getattr(_LOG, "warn", getattr(_LOG, "warning", _LOG.info))(msg)


def log_error(msg: str) -> None:
    _LOG.error(msg)


def sweep_grid(f_start: float, f_stop: float, points: int) -> np.ndarray:
    """Grille de fréquences que le driver programmera (même arrondi que configure_sweep)."""
    start, stop = int(round(f_start)), int(round(f_stop))
    step = (stop - start) // (points - 1) if points > 1 else 0
    return (start + np.arange(points, dtype=np.int64) * step).astype(float)


def rx_time_pmt(timestamp: float):
    """Horodatage au format conventionnel GNU Radio : tuple (secondes uint64, fraction double)."""
    if not math.isfinite(timestamp) or timestamp < 0:
        timestamp = 0.0
    whole = int(timestamp)
    return pmt.make_tuple(pmt.from_uint64(whole), pmt.from_double(timestamp - whole))


def rx_time_float(value) -> float:
    """Inverse de `rx_time_pmt` ; accepte aussi un double simple."""
    if pmt.is_tuple(value):
        return pmt.to_uint64(pmt.tuple_ref(value, 0)) + pmt.to_double(pmt.tuple_ref(value, 1))
    return float(pmt.to_python(value))


def add_sweep_tags(block, offset, seq, timestamp, f_start, f_step, calibrated) -> None:
    """Pose sur la sortie 0, à `offset`, les tags décrivant un sweep."""
    block.add_item_tag(0, offset, KEY_RX_TIME, rx_time_pmt(timestamp))
    block.add_item_tag(0, offset, KEY_SWEEP_IDX, pmt.from_long(int(seq)))
    block.add_item_tag(0, offset, KEY_F_START, pmt.from_double(float(f_start)))
    block.add_item_tag(0, offset, KEY_F_STEP, pmt.from_double(float(f_step)))
    block.add_item_tag(0, offset, KEY_CALIBRATED, pmt.from_bool(bool(calibrated)))


def sweep_pdu(seq, timestamp, f_start, f_step, calibrated, s):
    """PDU d'un sweep : métadonnées + vecteur c32 de longueur 2N (S11 puis S21)."""
    meta = pmt.make_dict()
    meta = pmt.dict_add(meta, KEY_SWEEP_IDX, pmt.from_long(int(seq)))
    meta = pmt.dict_add(meta, KEY_RX_TIME, rx_time_pmt(timestamp))
    meta = pmt.dict_add(meta, KEY_F_START, pmt.from_double(float(f_start)))
    meta = pmt.dict_add(meta, KEY_F_STEP, pmt.from_double(float(f_step)))
    meta = pmt.dict_add(meta, KEY_CALIBRATED, pmt.from_bool(bool(calibrated)))
    meta = pmt.dict_add(meta, pmt.intern("points"), pmt.from_long(int(s.shape[0])))
    data = np.concatenate([s[:, 0], s[:, 1]]).astype(np.complex64)
    return pmt.cons(meta, pmt.init_c32vector(len(data), data.tolist()))


def load_cal(path: str) -> Optional[Calibration]:
    """Charge un .cal ; chaîne vide -> None (données brutes)."""
    return Calibration.load(path) if path else None


def cal_matches(cal: Optional[Calibration], freqs) -> bool:
    if cal is None:
        return False
    try:
        cal._check_grid(freqs)
    except CalibrationError:
        return False
    return True
