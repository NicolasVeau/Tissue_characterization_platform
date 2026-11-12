"""CLI d'acquisition sans GUI : `python -m vna_app.cli {info,sweep}`."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vna_app.driver.litevna import LiteVNA, LiteVNAError


def cmd_info(args) -> int:
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
    print(f"variant         : 0x{info['variant']:02x} (SAA2/LiteVNA)")
    print(f"protocol version: {info['protocol_version']}")
    print(f"hardware rev    : {info['hardware_revision']}")
    print(f"firmware        : {info['firmware']}")
    return 0


def cmd_sweep(args) -> int:
    import skrf

    out = Path(args.out)
    if out.suffix.lower() != ".s2p":
        print("erreur: --out doit se terminer par .s2p", file=sys.stderr)
        return 2
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
        vna.configure_sweep(args.start, args.stop, args.points, args.avg)
        freqs, s_raw = vna.sweep()
        step = vna.step_hz

    s = np.zeros((len(freqs), 2, 2), dtype=complex)
    s[:, 0, 0] = s_raw[:, 0]  # S11
    s[:, 1, 0] = s_raw[:, 1]  # S21
    # S12 et S22 non mesurés (VNA 1 sens) : laissés à zéro.
    net = skrf.Network(frequency=skrf.Frequency.from_f(freqs, unit="hz"), s=s,
                       z0=50.0, name=out.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_touchstone(filename=out.stem, dir=str(out.parent))

    meta = {
        "date": datetime.now(timezone.utc).isoformat(),
        "start_hz": float(freqs[0]),
        "stop_hz_requested": args.stop,
        "stop_hz_actual": float(freqs[-1]),
        "step_hz": step,
        "points": args.points,
        "averaging": args.avg,
        "firmware": info["firmware"],
        "hardware_revision": info["hardware_revision"],
        "calibration_file": None,
        "calibrated": False,
        "note": "données brutes S11/S21 (rev/fwd), S12=S22=0",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"{len(freqs)} points écrits dans {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vna_app.cli")
    p.add_argument("--port", default="/dev/litevna")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="variant et firmware")
    sw = sub.add_parser("sweep", help="sweep S11/S21 -> .s2p + .json")
    sw.add_argument("--start", type=float, required=True, help="Hz")
    sw.add_argument("--stop", type=float, required=True, help="Hz")
    sw.add_argument("--points", type=int, required=True)
    sw.add_argument("--avg", type=int, default=1)
    sw.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        return {"info": cmd_info, "sweep": cmd_sweep}[args.cmd](args)
    except LiteVNAError as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
