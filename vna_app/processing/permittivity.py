"""Permittivité complexe ε* = ε' - jε'' à partir de S11 d'une sonde coaxiale ouverte.

Convention e^{+jωt} : ε'' > 0 pour un diélectrique à pertes.

Deux méthodes :

1. `eps_three_reference` — Γ est une fonction homographique (Möbius) de ε* pour
   tout modèle sonde+boîte d'erreur de type admittance linéaire en ε*
   (Y = jωC0·ε*). Le birapport est donc invariant : avec trois références
   (air ε=1, court-circuit ε→∞, liquide connu ε_ref) :
       K = (Γx−Γair)(Γcc−Γref) / ((Γx−Γref)(Γcc−Γair)) = (ε−1)/(ε−ε_ref)
       ε = (1 − K·ε_ref) / (1 − K)
   Absorbe directivité, suivi, adaptation et C0 : marche sur S11 brut comme
   calibré, pourvu que les 4 mesures aient les mêmes plans et la même grille.
   Ne modélise PAS le terme de rayonnement (en ω^{5/2}) : valide tant que la
   sonde est électriquement petite devant la longueur d'onde dans l'échantillon.

2. `eps_capacitive` — S11 calibré au plan de la sonde, C0 connu :
       ε* = Y / (jω C0),  Y = (1−Γ)/(Z0 (1+Γ)).
"""

from __future__ import annotations

import numpy as np

from vna_app.processing.impedance import gamma_to_y


def eps_three_reference(gamma_x, gamma_air, gamma_short, gamma_ref, eps_ref):
    g = [np.asarray(v, dtype=complex) for v in (gamma_x, gamma_air, gamma_short, gamma_ref)]
    if len({v.shape for v in g}) != 1:
        raise ValueError("les 4 mesures doivent avoir la même grille de fréquences")
    gx, ga, gs, gr = g
    eps_ref = np.asarray(eps_ref, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (gx - ga) * (gs - gr) / ((gx - gr) * (gs - ga))
        return (1 - k * eps_ref) / (1 - k)


def eps_capacitive(freqs, gamma, c0_farad: float, z0: float = 50.0):
    if c0_farad <= 0:
        raise ValueError("C0 doit être > 0")
    w = 2 * np.pi * np.asarray(freqs, dtype=float)
    return gamma_to_y(gamma, z0) / (1j * w * c0_farad)


def debye(freqs, eps_s: float, eps_inf: float, tau_s: float):
    """ε* Debye (convention e^{+jωt}) : ε∞ + (εs−ε∞)/(1 + jωτ)."""
    w = 2 * np.pi * np.asarray(freqs, dtype=float)
    return eps_inf + (eps_s - eps_inf) / (1 + 1j * w * tau_s)


def water_25c(freqs):
    """Eau pure à 25 °C, Debye à un pôle : εs=78.4, ε∞=5.2, τ=8.27 ps.

    Approximation (écart de quelques % à > 10 GHz ; conductivité ionique
    ignorée) — à remplacer par un modèle de référence pour une métrologie fine.
    """
    return debye(freqs, 78.4, 5.2, 8.27e-12)


REFERENCE_LIQUIDS = {"water": water_25c}


def to_eps_columns(eps):
    """(ε', ε'', tanδ) avec ε'' ≥ 0 pour un matériau à pertes."""
    eps = np.asarray(eps, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return eps.real, -eps.imag, -eps.imag / eps.real
