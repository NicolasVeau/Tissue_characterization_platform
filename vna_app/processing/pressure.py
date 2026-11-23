"""Réduction de l'effet de la pression d'appui (contact partiel) sur ε*.

Modèle : une fraction de contact `a` du champ de bord traverse le tissu, le reste de l'air
(ε = 1), en parallèle :  ε*_mes = a·ε* + (1 − a)  ⇒  ε*_mes − 1 = a·(ε* − 1).
Le facteur `a` (pression, pose à la main) est donc un simple facteur d'échelle complexe
réel sur (ε* − 1). Sont invariants à `a` :
  - la forme normalisée  u(f) = (ε* − 1)(f) / (ε* − 1)(f_réf) ;
  - le rapport de pertes  ε'' / (ε' − 1).
Non corrigés : entrefer en série (dépendant de f) et compression réelle du tissu.
Convention ε* = ε' − jε''.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def air_reduced(eps) -> np.ndarray:
    """ε* − 1 : proportionnel à la fraction de contact."""
    return np.asarray(eps, dtype=complex) - 1.0


def contact_level(eps) -> float:
    """Niveau de contact : médiane de |ε* − 1| sur la bande (≈ a·|ε*−1| du tissu)."""
    d = np.abs(air_reduced(eps))
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")


def _ref_value(freqs, v, f_ref, half_width: float = 0.03):
    """Valeur de référence de v à f_ref : médiane sur ±3 % de la bande (≥ 3 points)."""
    freqs = np.asarray(freqs, dtype=float)
    k0 = int(np.argmin(np.abs(freqs - f_ref)))
    n = max(1, int(round(half_width * len(freqs))))
    sl = slice(max(0, k0 - n), min(len(freqs), k0 + n + 1))
    w = v[sl]
    return np.median(w.real) + 1j * np.median(w.imag)


def normalize_shape(freqs, eps, f_ref: float) -> np.ndarray:
    """u(f) = (ε* − 1) / (ε* − 1)(f_réf). u(f_réf) ≈ 1 ; invariant à la fraction de contact."""
    v = air_reduced(eps)
    ref = _ref_value(freqs, v, f_ref)
    if not np.isfinite(ref) or abs(ref) < 1e-12:
        raise ValueError("référence de normalisation nulle (pas de contact ?)")
    return v / ref


def loss_ratio(eps) -> np.ndarray:
    """ε'' / (ε' − 1) : invariant à la fraction de contact (remplace tan δ)."""
    eps = np.asarray(eps, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -eps.imag / (eps.real - 1.0)


def band_features(freqs, eps, f_ref: float, n_bands: int = 8) -> np.ndarray:
    """Vecteur de descripteurs invariants : Re u, Im u et ε''/(ε'−1), moyennés sur
    `n_bands` bandes log-espacées. Taille 3·n_bands. Le niveau de contact est exclu."""
    freqs = np.asarray(freqs, dtype=float)
    u = normalize_shape(freqs, eps, f_ref)
    lr = loss_ratio(eps)
    edges = np.geomspace(freqs[0], freqs[-1], n_bands + 1)
    idx = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, n_bands - 1)
    out = []
    for q in (u.real, u.imag, lr):
        out.extend(np.nanmean(q[idx == b]) if np.any(idx == b) else np.nan for b in range(n_bands))
    return np.asarray(out)


def fisher_ratio(features, zones) -> np.ndarray:
    """Rapport (variance des moyennes par zone) / (variance intra-zone moyenne), par colonne.

    `features` (n_sweeps, n_feat), `zones` (n_sweeps,). La variance intra-zone contient la
    variation de pression si les poses la couvrent : plus le rapport est grand, plus le
    descripteur sépare les zones malgré la pression. Sert à comparer des jeux de descripteurs.
    """
    x = np.asarray(features, dtype=float)
    z = np.asarray(zones)
    labels = np.unique(z)
    means = np.stack([x[z == k].mean(axis=0) for k in labels])
    within = np.mean([x[z == k].var(axis=0) for k in labels], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return means.var(axis=0) / within


def passes_contact(eps, minimum: float) -> Optional[bool]:
    """True/False selon le niveau de contact ; None si ε* est indisponible."""
    if eps is None:
        return None
    lvl = contact_level(eps)
    return bool(np.isfinite(lvl) and lvl >= minimum)
