"""CLI d'acquisition sans GUI : `python -m vna_app.cli {info,sweep}`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vna_app.driver.litevna import LiteVNA, LiteVNAError
from vna_app.processing.export import write_s2p


def cmd_info(args) -> int:
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
    print(f"variant         : 0x{info['variant']:02x} (SAA2/LiteVNA)")
    print(f"protocol version: {info['protocol_version']}")
    print(f"hardware rev    : {info['hardware_revision']}")
    print(f"firmware        : {info['firmware']}")
    return 0


def cmd_sweep(args) -> int:
    out = Path(args.out)
    if out.suffix.lower() != ".s2p":
        print("erreur: --out doit se terminer par .s2p", file=sys.stderr)
        return 2
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
        vna.configure_sweep(args.start, args.stop, args.points, args.avg)
        freqs, s_raw = vna.sweep()
        step = vna.step_hz

    write_s2p(
        out,
        freqs,
        s_raw,
        step_hz=step,
        points=args.points,
        averaging=args.avg,
        info=info,
        stop_hz_requested=args.stop,
    )
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
