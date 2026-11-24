"""CLI d'acquisition sans GUI : `python -m vna_app.cli {info,sweep,record,calibrate,eps,noise}`."""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

from vna_app.calibration import Calibration, CalibrationError
from vna_app.driver.litevna import LiteVNA, LiteVNAError, antispur_grid
from vna_app.driver.sim import open_device
from vna_app.processing.export import write_s2p
from vna_app.processing.filters import smooth_complex
from vna_app.processing.permittivity import (
    REFERENCE_LIQUIDS,
    eps_capacitive,
    eps_three_reference,
    to_eps_columns,
)
from vna_app.processing.record import RecordError, SweepRecorder


def _apply_antispur(args) -> None:
    """Décale la grille demandée pour éviter le peigne de raies du LiteVNA (--antispur)."""
    if not getattr(args, "antispur", False):
        return
    start, stop, points, clr = antispur_grid(
        int(round(args.start)), int(round(args.stop)), args.points
    )
    print(
        f"grille anti-raies : {start / 1e6:g}–{stop / 1e6:g} MHz, {points} pts "
        f"(écart au peigne 12 MHz ≥ {clr / 1e6:g} MHz)"
    )
    args.start, args.stop, args.points = float(start), float(stop), points


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
    _apply_antispur(args)
    with LiteVNA(args.port) as vna:
        info = vna.device_info()
        if args.power is not None:
            vna.set_power(adf4350=args.power)
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


def cmd_record(args) -> int:
    """Série de sweeps BRUTS horodatés -> .npz (processing/record.py). --port sim accepté."""
    if args.count is None and args.duration is None:
        print("erreur: fournir --count et/ou --duration", file=sys.stderr)
        return 2
    _apply_antispur(args)
    with open_device(args.port) as vna:
        info = vna.device_info()
        vna.configure_sweep(args.start, args.stop, args.points, args.avg)
        meta = {
            "start_hz": vna.start_hz,
            "step_hz": vna.step_hz,
            "points": vna.points,
            "averaging": vna.averaging,
            "firmware": info["firmware"],
            "hardware_revision": info["hardware_revision"],
            "port": args.port,
            "calibrated": False,
            "note": "données brutes S11/S21 (rev/fwd)",
        }
        t_end = None if args.duration is None else time.monotonic() + args.duration
        with SweepRecorder(args.out, meta) as rec:
            q = vna.start_streaming(maxsize=64)
            try:
                while (args.count is None or len(rec) < args.count) and (
                    t_end is None or time.monotonic() < t_end
                ):
                    try:
                        item = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if isinstance(item, LiteVNAError):
                        raise item
                    rec.append(item)
            except KeyboardInterrupt:
                pass
            finally:
                vna.stop_streaming()
            n = len(rec)
    print(f"{n} sweep(s) de {args.points} points écrits dans {args.out}")
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


def _cstd(s):
    """Écart-type complexe (ddof=1) par fréquence : s (sweeps, N) -> (N,)."""
    m = s.mean(axis=0)
    return np.sqrt((np.abs(s - m) ** 2).sum(axis=0) / (s.shape[0] - 1))


def _db(x) -> str:
    return f"{20 * np.log10(max(float(x), 1e-12)):.1f} dB"


def _eps_noise_gain(gx, ga, gs, gr, eps_ref):
    """|dε*/dΓx| du birapport (permittivity.eps_three_reference) au point de mesure Γx.

    K = (Γx−Γa)(Γs−Γr)/((Γx−Γr)(Γs−Γa)), ε = (1−K·εref)/(1−K) ; la dérivée est
    analytique (fonctions de Möbius) : dK/dΓx = (Γs−Γr)(Γa−Γr)/((Γs−Γa)(Γx−Γr)²)
    et dε/dK = (1−εref)/(1−K)². Le module est le facteur d'amplification du bruit.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (gx - ga) * (gs - gr) / ((gx - gr) * (gs - ga))
        dk = (gs - gr) * (ga - gr) / ((gs - ga) * (gx - gr) ** 2)
        return np.abs((1 - eps_ref) / (1 - k) ** 2 * dk)


def _load_eps_refs(path, freqs):
    """Références ε* (.npz de la GUI / Pipeline.save_refs) -> (Γair, Γcc, Γref, ε_ref)."""
    from vna_app.processing.pipeline import EPS_STANDARDS

    try:
        with np.load(path, allow_pickle=False) as z:
            names = [str(n) for n in z["names"]]
            missing = [k for k in EPS_STANDARDS if k not in names]
            if missing:
                raise CalibrationError(f"références incomplètes : manque {', '.join(missing)}")
            g = []
            for k in EPS_STANDARDS:
                f = np.asarray(z[f"f_{k}"], dtype=float)
                if f.shape != freqs.shape or float(np.abs(f - freqs).max()) > 1.0:
                    raise CalibrationError(f"{k}: grille différente du sweep demandé")
                g.append(np.asarray(z[f"s_{k}"], dtype=complex)[:, 0])
            liquid = str(z["ref_liquid"]) if "ref_liquid" in z else "water"
    except (OSError, KeyError, ValueError) as e:
        raise CalibrationError(f"références illisibles : {e}") from e
    if liquid not in REFERENCE_LIQUIDS:
        raise CalibrationError(f"liquide de référence inconnu : {liquid}")
    print(f"références ε* chargées : {path} (liquide {liquid})")
    return g[0], g[1], g[2], REFERENCE_LIQUIDS[liquid](freqs)


def cmd_noise(args) -> int:
    """Banc de bruit : sweeps répétés sur une charge FIXE -> σ(S11)/σ(S21) par averaging.

    Ne rien toucher pendant la mesure (sonde dans l'air, ou sur une charge).
    Avec --refs, propage σ(S11) en Δε* résoluble via la dérivée du birapport.
    """
    try:
        avg_list = sorted({int(a) for a in args.avg_list.split(",")})
    except ValueError:
        raise ValueError(f"--avg-list invalide : {args.avg_list!r} (attendu ex. 1,4,16)") from None
    if not avg_list or not all(1 <= a <= 0xFFFF for a in avg_list):
        raise ValueError("--avg-list : valeurs dans 1..65535")
    if args.sweeps < 2:
        raise ValueError("--sweeps doit être ≥ 2")
    _apply_antispur(args)
    refs = None
    freqs = None
    meds, dts, sig11s, sig21s, mean11s, gains, depss = [], [], [], [], [], [], []
    with open_device(args.port) as vna:
        info = vna.device_info()
        print(
            f"firmware {info['firmware']} — {args.points} pts, "
            f"{args.start / 1e6:g}–{args.stop / 1e6:g} MHz, {args.sweeps} sweeps par averaging"
        )
        print("charge fixe requise : ne pas toucher la sonde pendant la mesure")
        head = (
            f"{'avg':>5}  {'s/sweep':>8}  {'entrées/s':>9}  "
            f"{'σ(S11) méd':>11}  {'p95':>9}  {'σ(S21) méd':>11}"
        )
        if args.refs:
            head += f"  {'Δε méd':>8}  {'Δε p95':>8}"
        for avg in avg_list:
            freqs = vna.configure_sweep(args.start, args.stop, args.points, avg)
            if refs is None and args.refs:
                refs = _load_eps_refs(args.refs, freqs)
                print(head)
            elif avg == avg_list[0] and not args.refs:
                print(head)
            vna.sweep()  # sweep de chauffe écarté (points de stabilisation après reconfig)
            t0 = time.monotonic()
            s = np.stack([vna.sweep()[1] for _ in range(args.sweeps)])
            dt = (time.monotonic() - t0) / args.sweeps
            sig11, sig21 = _cstd(s[:, :, 0]), _cstd(s[:, :, 1])
            mean11 = s[:, :, 0].mean(axis=0)
            meds.append(float(np.median(sig11)))
            dts.append(dt)
            sig11s.append(sig11)
            sig21s.append(sig21)
            mean11s.append(mean11)
            line = (
                f"{avg:>5}  {dt:>8.2f}  {args.points * avg / dt:>9.0f}  "
                f"{_db(np.median(sig11)):>11}  {_db(np.percentile(sig11, 95)):>9}  "
                f"{_db(np.median(sig21)):>11}"
            )
            if refs is not None:
                ga, gcc, gr, eps_ref = refs
                gain = _eps_noise_gain(mean11, ga, gcc, gr, eps_ref)
                deps = gain * sig11
                gains.append(gain)
                depss.append(deps)
                line += f"  {np.median(deps):>8.3g}  {np.percentile(deps, 95):>8.3g}"
            print(line)
    if len(avg_list) > 1 and meds[0] > 0:
        parts = ", ".join(
            f"avg={a} ×{m / meds[0]:.2f} (attendu {np.sqrt(avg_list[0] / a):.2f})"
            for a, m in zip(avg_list[1:], meds[1:])
        )
        print(f"échelle du bruit vs avg={avg_list[0]} : {parts}")
    if args.out:
        data = {
            "freqs": freqs,
            "avgs": np.array(avg_list),
            "sweeps": args.sweeps,
            "t_sweep": np.array(dts),
            "sigma_s11": np.stack(sig11s),
            "sigma_s21": np.stack(sig21s),
            "mean_s11": np.stack(mean11s),
        }
        if gains:
            data["eps_gain"] = np.stack(gains)
            data["deps_s11"] = np.stack(depss)
        np.savez(args.out, **data)
        print(f"détail par fréquence écrit dans {args.out}")
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
    sw.add_argument(
        "--power",
        type=int,
        choices=range(4),
        help="puissance source > 140 MHz : 0 (min)..3 (max, défaut)",
    )
    sw.add_argument("--out", required=True)
    sw.add_argument("--cal", help="fichier .cal à appliquer (même grille de fréquences)")
    sw.add_argument(
        "--delay-ns", type=float, help="délai simple trajet du câble (ns), prioritaire sur le .cal"
    )
    rc = sub.add_parser("record", help="série de sweeps bruts horodatés -> .npz")
    rc.add_argument("--start", type=float, required=True, help="Hz")
    rc.add_argument("--stop", type=float, required=True, help="Hz (= --start : fréquence fixe)")
    rc.add_argument("--points", type=int, required=True)
    rc.add_argument("--avg", type=int, default=1)
    rc.add_argument("--count", type=int, help="nombre de sweeps")
    rc.add_argument("--duration", type=float, help="durée maximale (s)")
    rc.add_argument("--out", required=True, help="fichier .npz")
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
    nz = sub.add_parser("noise", help="banc de bruit : sweeps répétés sur charge fixe -> σ(Γ)")
    nz.add_argument("--start", type=float, default=400e6, help="Hz (défaut 400 MHz)")
    nz.add_argument("--stop", type=float, default=3e9, help="Hz (défaut 3 GHz)")
    nz.add_argument("--points", type=int, default=201)
    nz.add_argument("--avg-list", default="1,4,16", help="valeurs d'averaging, ex. 1,4,16")
    nz.add_argument(
        "--sweeps", type=int, default=10, help="sweeps par valeur d'averaging (≥ 2, défaut 10)"
    )
    nz.add_argument(
        "--refs",
        help="références ε* .npz (GUI « Sauver les références ») : propage σ(S11) en Δε*",
    )
    nz.add_argument("--out", help="détail par fréquence -> fichier .npz")
    for sp in (sw, rc, nz):
        sp.add_argument(
            "--antispur",
            action="store_true",
            help="décale la grille pour éviter les raies internes du LiteVNA (peigne 12 MHz)",
        )
    args = p.parse_args(argv)
    try:
        return {
            "info": cmd_info,
            "sweep": cmd_sweep,
            "record": cmd_record,
            "calibrate": cmd_calibrate,
            "eps": cmd_eps,
            "noise": cmd_noise,
        }[args.cmd](args)
    except (LiteVNAError, CalibrationError, RecordError) as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"erreur: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
