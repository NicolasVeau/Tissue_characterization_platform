"""Calibration hôte à partir des mesures brutes du LiteVNA.

Le LiteVNA ne mesure que dans le sens direct (S11 et S21 ; pas de S12/S22).
Une calibration SOLT 12 termes n'est donc pas possible. On fait :

- S11 : calibration 1 port complète, 3 termes d'erreur (directivité,
  adaptation source, suivi de réflexion) à partir de Open/Short/Load ;
  modèle  Γm = a + b·Γ + c·Γ·Γm  avec  a = e00, c = e11, b = e01·e10 − e00·e11,
  résolu aux moindres carrés par fréquence (≥ 3 standards), d'où
  Γ = (Γm − a) / (b + c·Γm).
- S21 : normalisation par le Thru (réponse en transmission) :
  S21_cal = S21_brut / S21_brut(thru). Ne corrige pas l'adaptation port 2.

- Délai de câble (optionnel, variable) : si un câble coaxial 50 Ω relie le plan de
  calibration au capteur, S11 est ramené au plan du capteur par
  Γ_capteur = Γ_cal · exp(+j·2ω·τ) (aller-retour). Câble supposé sans perte et
  adapté. S21 n'est pas modifié (la normalisation Thru est faite au plan de cal).

Fichier .cal : npz (termes d'erreur) + métadonnées JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from vna_app.calibration.kits import IdealKit, Kit


class CalibrationError(Exception):
    pass


class Calibration:
    def __init__(
        self, freqs, a, b, c, thru=None, meta: Optional[dict] = None, delay_s: float = 0.0
    ):
        self.freqs = np.asarray(freqs, dtype=float)
        self.a, self.b, self.c = (np.asarray(x, dtype=complex) for x in (a, b, c))
        self.thru = None if thru is None else np.asarray(thru, dtype=complex)
        self.meta = meta or {}
        self.delay_s = float(delay_s)  # délai simple trajet plan de cal -> capteur (s)

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_measurements(
        cls, freqs, open_s11, short_s11, load_s11, thru_s21=None, kit: Optional[Kit] = None
    ) -> "Calibration":
        """Arguments : S11 bruts mesurés sur Open/Short/Load (N,), S21 brut du Thru (N,)."""
        kit = kit or IdealKit()
        freqs = np.asarray(freqs, dtype=float)
        meas = [np.asarray(m, dtype=complex) for m in (open_s11, short_s11, load_s11)]
        if any(m.shape != freqs.shape for m in meas):
            raise CalibrationError("les mesures doivent avoir la même grille de fréquences")
        gam = [kit.open(freqs), kit.short(freqs), kit.load(freqs)]
        n = len(freqs)
        a = np.empty(n, complex)
        b = np.empty(n, complex)
        c = np.empty(n, complex)
        for k in range(n):
            # Γm = a + b Γ + c Γ Γm
            A = np.array([[1.0, g[k], g[k] * m[k]] for g, m in zip(gam, meas)], dtype=complex)
            y = np.array([m[k] for m in meas], dtype=complex)
            if np.linalg.cond(A) > 1e12:
                raise CalibrationError(f"standards dégénérés à {freqs[k]:.0f} Hz")
            a[k], b[k], c[k] = np.linalg.lstsq(A, y, rcond=None)[0]
        thru = None
        if thru_s21 is not None:
            thru = np.asarray(thru_s21, dtype=complex)
            if thru.shape != freqs.shape:
                raise CalibrationError("Thru : grille de fréquences différente")
            if np.any(thru == 0):
                raise CalibrationError("Thru nul : liaison absente ?")
        meta = {
            "date": datetime.now(timezone.utc).isoformat(),
            "kit": getattr(kit, "name", type(kit).__name__),
            "type": "S11 1-port (OSL) + S21 normalisation thru"
            if thru is not None
            else "S11 1-port (OSL)",
        }
        return cls(freqs, a, b, c, thru, meta)

    # -- application -----------------------------------------------------------
    def _check_grid(self, freqs):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.shape != self.freqs.shape or np.max(np.abs(freqs - self.freqs)) > 1.0:
            raise CalibrationError(
                "grille de fréquences différente de celle de la calibration "
                f"({self.freqs[0]:.0f}–{self.freqs[-1]:.0f} Hz, {len(self.freqs)} pts) : "
                "refaire la calibration pour ce sweep"
            )

    def apply(self, freqs, s_raw, delay_s: Optional[float] = None):
        """s_raw (N,2) [S11,S21] bruts -> (N,2) calibrés. S21 inchangé sans Thru.

        delay_s : délai simple trajet du câble (s) ; None -> self.delay_s.
        """
        self._check_grid(freqs)
        s_raw = np.asarray(s_raw, dtype=complex)
        out = np.empty_like(s_raw)
        m = s_raw[:, 0]
        out[:, 0] = (m - self.a) / (self.b + self.c * m)
        tau = self.delay_s if delay_s is None else float(delay_s)
        if tau != 0.0:
            out[:, 0] *= np.exp(2j * np.pi * np.asarray(freqs, dtype=float) * 2.0 * tau)
        out[:, 1] = s_raw[:, 1] / self.thru if self.thru is not None else s_raw[:, 1]
        return out

    # -- fichier .cal ----------------------------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "freqs": self.freqs,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "meta": np.array(json.dumps(self.meta)),
            "delay_s": np.array(self.delay_s),
        }
        if self.thru is not None:
            arrays["thru"] = self.thru
        with open(path, "wb") as f:  # évite l'ajout automatique de .npz
            np.savez(f, **arrays)
        return path

    @classmethod
    def load(cls, path) -> "Calibration":
        try:
            with np.load(path, allow_pickle=False) as d:
                thru = d["thru"] if "thru" in d.files else None
                delay = float(d["delay_s"]) if "delay_s" in d.files else 0.0
                return cls(
                    d["freqs"], d["a"], d["b"], d["c"], thru, json.loads(str(d["meta"])), delay
                )
        except (OSError, KeyError, ValueError) as e:
            raise CalibrationError(f"fichier .cal illisible ({path}): {e}") from e
