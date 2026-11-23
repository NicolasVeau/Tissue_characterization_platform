"""Vérifie sur l'appareil le mode fréquence fixe (start = stop, pas nul) du bloc CW Source.

    python gr-litevna/tools/check_fixed_freq.py --port /dev/litevna --freq 1e9 --points 300

Affiche si le firmware accepte un pas nul, la cadence obtenue et la dispersion des
échantillons (DUT stable attendu : charge 50 Ω ou court-circuit). `--port sim` pour essayer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # racine du dépôt

from vna_app.driver.litevna import LiteVNAError  # noqa: E402
from vna_app.driver.sim import open_device  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", default="/dev/litevna")
    p.add_argument("--freq", type=float, default=1e9, help="Hz")
    p.add_argument("--points", type=int, default=300)
    p.add_argument("--avg", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3, help="nombre d'acquisitions")
    args = p.parse_args(argv)
    try:
        with open_device(args.port) as vna:
            print("appareil :", vna.device_info())
            freqs = vna.configure_sweep(args.freq, args.freq, args.points, args.avg)
            print(f"pas programmé : {vna.step_hz} Hz, {len(freqs)} points à {freqs[0]:.0f} Hz")
            for i in range(args.repeat):
                t0 = time.monotonic()
                _, s = vna.sweep()
                dt = time.monotonic() - t0
                m = np.mean(s, axis=0)
                sd = np.std(s, axis=0)
                print(
                    f"acquisition {i}: {dt:.2f} s, {args.points / dt:.0f} échantillons/s | "
                    f"S11 moyen {abs(m[0]):.4f} ∠{np.degrees(np.angle(m[0])):.1f}°, "
                    f"écart-type {sd[0]:.2e} | |S21| moyen {abs(m[1]):.4f}, écart-type {sd[1]:.2e}"
                )
    except (LiteVNAError, ValueError) as e:
        print(f"ÉCHEC : {e}", file=sys.stderr)
        print("Le firmware refuse peut-être un pas nul : le signaler dans docs/protocol.md.")
        return 1
    print("OK : le mode fréquence fixe fonctionne ; noter la cadence dans CLAUDE.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
