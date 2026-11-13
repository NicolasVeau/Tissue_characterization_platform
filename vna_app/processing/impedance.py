"""S11 -> impédance / admittance, décalage du plan de référence."""

from __future__ import annotations

import numpy as np

C0_M_S = 299_792_458.0


def gamma_to_z(gamma, z0: float = 50.0):
    """Z = Z0 (1+Γ)/(1-Γ). Γ=1 (open) donne inf (warning numpy supprimé)."""
    gamma = np.asarray(gamma, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return z0 * (1 + gamma) / (1 - gamma)


def gamma_to_y(gamma, z0: float = 50.0):
    """Y = (1-Γ)/(Z0 (1+Γ))."""
    gamma = np.asarray(gamma, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - gamma) / (z0 * (1 + gamma))


def shift_reference_plane(freqs, gamma, length_m: float, er: float = 1.0):
    """Ramène Γ au plan situé `length_m` en retrait (de-embedding d'une ligne sans perte).

    Γ_plan = Γ_mesuré · exp(+j·4π·f·l·√er / c). length_m > 0 : on recule vers le DUT
    de la longueur de ligne entre le plan de calibration et le DUT. Pertes de
    ligne non corrigées.
    """
    f = np.asarray(freqs, dtype=float)
    return np.asarray(gamma, dtype=complex) * np.exp(
        1j * 4 * np.pi * f * length_m * np.sqrt(er) / C0_M_S
    )
