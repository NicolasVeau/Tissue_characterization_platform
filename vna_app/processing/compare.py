"""Comparaison de deux mesures mémorisées (|Z|, arg Z, ε*), sans Qt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vna_app.processing.impedance import gamma_to_z


@dataclass
class Snapshot:
    freqs: np.ndarray  # Hz
    z: np.ndarray  # (N,) complexe, Ω
    eps: Optional[np.ndarray]  # (N,) complexe (convention ε' - jε'') ou None
    label: str = ""


def snapshot(freqs, gamma, eps=None, label: str = "") -> Snapshot:
    return Snapshot(
        np.array(freqs, dtype=float),
        gamma_to_z(np.asarray(gamma, dtype=complex)),
        None if eps is None else np.array(eps, dtype=complex),
        label,
    )


def mod_db(z):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20 * np.log10(np.abs(z))


def unwrap_deg(x):
    """Phase déroulée en degrés ; les valeurs non finies restent NaN sans casser le déroulage."""
    x = np.asarray(x, dtype=complex)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x != 0)
    if ok.any():
        out[ok] = np.degrees(np.unwrap(np.angle(x[ok])))
    return out


def _resample(f_ref, f, y):
    """Interpole y(f) sur f_ref ; NaN hors de la plage de f. Identité si grilles égales."""
    f_ref, f = np.asarray(f_ref, float), np.asarray(f, float)
    y = np.asarray(y, dtype=complex)
    if f_ref.shape == f.shape and float(np.max(np.abs(f_ref - f))) <= 1.0:
        return y
    re = np.interp(f_ref, f, y.real, left=np.nan, right=np.nan)
    im = np.interp(f_ref, f, y.imag, left=np.nan, right=np.nan)
    return re + 1j * im


def compare(a: Snapshot, b: Snapshot) -> dict:
    """Mémoire 2 ramenée sur la grille de la mémoire 1. Différences = 2 − 1.

    d_arg_deg = phase déroulée de Z2/Z1 (pas de saut de 360° entre les deux mesures).
    Les clés eps* valent None si l'une des deux mémoires n'a pas d'ε*.
    """
    f = a.freqs
    z1 = a.z
    z2 = _resample(f, b.freqs, b.z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = z2 / z1
    out = {
        "freqs": f,
        "mod1_db": mod_db(z1),
        "mod2_db": mod_db(z2),
        "d_mod_db": mod_db(z2) - mod_db(z1),
        "arg1_deg": unwrap_deg(z1),
        "arg2_deg": unwrap_deg(z2),
        "d_arg_deg": unwrap_deg(ratio),
        "eps1_re": None,
        "eps1_im": None,
        "eps2_re": None,
        "eps2_im": None,
        "d_eps_re": None,
        "d_eps_im": None,
    }
    if a.eps is not None and b.eps is not None:
        e1 = a.eps
        e2 = _resample(f, b.freqs, b.eps)
        out.update(
            eps1_re=e1.real,
            eps1_im=-e1.imag,  # ε'' ≥ 0
            eps2_re=e2.real,
            eps2_im=-e2.imag,
            d_eps_re=e2.real - e1.real,
            d_eps_im=-(e2.imag - e1.imag),
        )
    return out
