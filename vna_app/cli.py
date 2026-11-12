"""CLI d'acquisition sans GUI : `python -m vna_app.cli {info,sweep,calibrate}`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vna_app.calibration import Calibration, CalibrationError
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
    cal = Calibration.load(args.cal) if args.cal else None
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
        vna.configure_sweep(args.start, args.stop, args.points, args.avg)
        freqs, s_raw = vna.sweep()
        step = vna.step_hz

    if cal is not None:
        s_raw = cal.apply(freqs, s_raw)  # CalibrationError si grille différente

    write_s2p(
        out,
        freqs,
        s_raw,
        step_hz=step,
        points=args.points,
        averaging=args.avg,
        info=info,
        stop_hz_requested=args.stop,
        calibration_file=args.cal,
    )
    print(f"{len(freqs)} points écrits dans {out}")
    return 0


def _load_s(path):
    import skrf

    n = skrf.Network(path)
    if n.nports != 2:
        raise CalibrationError(f"{path}: .s2p attendu")
    return n.f, n.s[:, 0, 0], n.s[:, 1, 0]


def cmd_calibrate(args) -> int:
    """Construit un .cal à partir de sweeps BRUTS (cli sweep sans --cal) des standards."""
    f, o, _ = _load_s(args.open)
    for name, path in (("short", args.short), ("load", args.load)):
        fx = _load_s(path)[0]
        if fx.shape != f.shape or abs(fx - f).max() > 1.0:
            raise CalibrationError(f"{name}: grille de fréquences différente de --open")
    s = _load_s(args.short)[1]
    ld = _load_s(args.load)[1]
    thru = None
    if args.thru:
        ft, _, thru = _load_s(args.thru)
        if ft.shape != f.shape or abs(ft - f).max() > 1.0:
            raise CalibrationError("thru: grille de fréquences différente de --open")
    cal = Calibration.from_measurements(f, o, s, ld, thru_s21=thru)
    cal.save(args.out)
    print(f"calibration écrite dans {args.out} ({len(f)} pts, {cal.meta['type']})")
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
    sw.add_argument("--cal", help="fichier .cal à appliquer (même grille de fréquences)")
    cp = sub.add_parser("calibrate", help="sweeps bruts O/S/L(/Thru) -> .cal")
    for k in ("open", "short", "load"):
        cp.add_argument(f"--{k}", required=True, help=f"sweep brut .s2p du standard {k}")
    cp.add_argument("--thru", help="sweep brut .s2p du Thru (normalisation S21)")
    cp.add_argument("--out", required=True, help="fichier .cal")
    args = p.parse_args(argv)
    try:
        return {"info": cmd_info, "sweep": cmd_sweep, "calibrate": cmd_calibrate}[args.cmd](args)
    except (LiteVNAError, CalibrationError) as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
