"""CLI d'acquisition sans GUI : `python -m vna_app.cli {info,sweep,calibrate,eps}`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vna_app.calibration import Calibration, CalibrationError
from vna_app.driver.litevna import LiteVNA, LiteVNAError
from vna_app.processing.export import write_s2p
from vna_app.processing.filters import smooth_complex
from vna_app.processing.permittivity import (
    REFERENCE_LIQUIDS,
    eps_capacitive,
    eps_three_reference,
    to_eps_columns,
)


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
        delay = None if args.delay_ns is None else args.delay_ns * 1e-9
        s_raw = cal.apply(freqs, s_raw, delay_s=delay)  # CalibrationError si grille différente

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
    cal.delay_s = args.delay_ns * 1e-9
    cal.save(args.out)
    print(f"calibration écrite dans {args.out} ({len(f)} pts, {cal.meta['type']})")
    return 0


def _same_grid(name, f, ref):
    if f.shape != ref.shape or abs(f - ref).max() > 1.0:
        raise CalibrationError(f"{name}: grille de fréquences différente de --sample")


def cmd_eps(args) -> int:
    """ε* d'une sonde coaxiale ouverte à partir de S11 de fichiers .s2p."""
    import numpy as np

    f, gx, _ = _load_s(args.sample)
    if args.c0 is not None:
        meta = Path(args.sample).with_suffix(".json")
        if meta.exists() and '"calibrated": false' in meta.read_text():
            print("attention: --c0 suppose un S11 calibré, ce sweep est brut", file=sys.stderr)
        eps = eps_capacitive(f, gx, args.c0)
        method = f"capacitif C0={args.c0:g} F"
    else:
        if not (args.air and args.short and args.ref):
            raise CalibrationError("sans --c0, fournir --air, --short et --ref")
        gs = []
        for name, path in (("air", args.air), ("short", args.short), ("ref", args.ref)):
            fx, g, _ = _load_s(path)
            _same_grid(name, fx, f)
            gs.append(g)
        eps_ref = REFERENCE_LIQUIDS[args.ref_liquid](f)
        eps = eps_three_reference(gx, gs[0], gs[1], gs[2], eps_ref)
        method = f"3 références (air, court-circuit, {args.ref_liquid})"
    if args.smooth > 1:
        eps = smooth_complex(eps, args.smooth)
    e1, e2, tan = to_eps_columns(eps)
    data = np.column_stack([f, e1, e2, tan])
    np.savetxt(
        args.out,
        data,
        delimiter=",",
        header="f_Hz,eps_real,eps_loss,tan_delta",
        comments="",
        fmt="%.9g",
    )
    print(f"{len(f)} points écrits dans {args.out} — méthode : {method}")
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
    sw.add_argument(
        "--delay-ns", type=float, help="délai simple trajet du câble (ns), prioritaire sur le .cal"
    )
    cp = sub.add_parser("calibrate", help="sweeps bruts O/S/L(/Thru) -> .cal")
    for k in ("open", "short", "load"):
        cp.add_argument(f"--{k}", required=True, help=f"sweep brut .s2p du standard {k}")
    cp.add_argument("--thru", help="sweep brut .s2p du Thru (normalisation S21)")
    cp.add_argument(
        "--delay-ns", type=float, default=0.0, help="délai simple trajet câble->capteur (ns)"
    )
    cp.add_argument("--out", required=True, help="fichier .cal")
    ep = sub.add_parser("eps", help="ε* (sonde coaxiale ouverte) -> CSV")
    ep.add_argument("--sample", required=True, help=".s2p de l'échantillon (S11)")
    ep.add_argument("--air", help=".s2p sonde dans l'air (ε=1)")
    ep.add_argument("--short", help=".s2p sonde court-circuitée")
    ep.add_argument("--ref", help=".s2p sonde dans le liquide de référence")
    ep.add_argument("--ref-liquid", default="water", choices=sorted(REFERENCE_LIQUIDS))
    ep.add_argument("--c0", type=float, help="capacité de frange (F) : méthode capacitive")
    ep.add_argument("--smooth", type=int, default=1, help="fenêtre Savitzky-Golay (impair)")
    ep.add_argument("--out", required=True, help="fichier .csv")
    args = p.parse_args(argv)
    try:
        return {"info": cmd_info, "sweep": cmd_sweep, "calibrate": cmd_calibrate, "eps": cmd_eps}[
            args.cmd
        ](args)
    except (LiteVNAError, CalibrationError) as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
