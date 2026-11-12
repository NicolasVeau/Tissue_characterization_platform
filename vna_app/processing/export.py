"""Export des mesures : Touchstone .s2p + métadonnées JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def to_network(freqs, s_raw, name: str = "vna"):
    """(freqs Hz, s_raw (N,2) [S11,S21]) -> skrf.Network 2 ports, S12=S22=0, z0=50."""
    import skrf

    s = np.zeros((len(freqs), 2, 2), dtype=complex)
    s[:, 0, 0] = s_raw[:, 0]
    s[:, 1, 0] = s_raw[:, 1]
    return skrf.Network(frequency=skrf.Frequency.from_f(freqs, unit="hz"), s=s, z0=50.0, name=name)


def write_s2p(
    out,
    freqs,
    s_raw,
    *,
    step_hz,
    points,
    averaging,
    info,
    stop_hz_requested=None,
    calibration_file=None,
) -> Path:
    """Écrit out (.s2p) et out.json. Retourne le chemin du .s2p."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net = to_network(freqs, s_raw, out.stem)
    net.write_touchstone(filename=out.stem, dir=str(out.parent))
    meta = {
        "date": datetime.now(timezone.utc).isoformat(),
        "start_hz": float(freqs[0]),
        "stop_hz_requested": stop_hz_requested,
        "stop_hz_actual": float(freqs[-1]),
        "step_hz": step_hz,
        "points": points,
        "averaging": averaging,
        "firmware": info["firmware"],
        "hardware_revision": info["hardware_revision"],
        "calibration_file": str(calibration_file) if calibration_file else None,
        "calibrated": calibration_file is not None,
        "note": ("S11/S21 calibrés" if calibration_file else "données brutes S11/S21 (rev/fwd)")
        + ", S12=S22=0",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out
