"""Chaîne de traitement de l'affichage : calibration puis ε*, sans dépendance Qt.

`process()` ne lève pas pour les problèmes d'utilisateur (grille différente,
référence manquante) : le résultat tombe en brut / sans ε* et `notes` explique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from vna_app.calibration import Calibration, CalibrationError
from vna_app.processing.filters import smooth_complex
from vna_app.processing.permittivity import (
    REFERENCE_LIQUIDS,
    eps_capacitive,
    eps_three_reference,
)

CAL_STANDARDS = ("open", "short", "load", "thru")
EPS_STANDARDS = ("air", "probe_short", "ref")


@dataclass
class Result:
    s: np.ndarray  # (N,2) affiché (calibré si applicable)
    calibrated: bool
    eps: Optional[np.ndarray] = None  # (N,) complexe, convention ε' - jε''
    notes: List[str] = field(default_factory=list)


def _same_grid(a, b) -> bool:
    return a.shape == b.shape and (a.size == 0 or float(np.max(np.abs(a - b))) <= 1.0)


class Pipeline:
    def __init__(self):
        self.standards = {}  # nom -> (freqs, s_raw)
        self.cal: Optional[Calibration] = None
        self.cal_path: Optional[str] = None
        self.apply_cal = False
        self.eps_mode = "off"  # "off" | "capacitive" | "three_ref"
        self.c0 = 0.08e-12
        self.ref_liquid = "water"
        self.smooth = 1
        self.delay_s = 0.0  # câble plan de cal -> capteur (simple trajet, s)

    # -- standards -------------------------------------------------------------
    def capture(self, name, freqs, s_raw):
        if name not in CAL_STANDARDS + EPS_STANDARDS:
            raise ValueError(f"standard inconnu: {name}")
        self.standards[name] = (np.array(freqs, dtype=float), np.array(s_raw, dtype=complex))

    def captured(self):
        return sorted(self.standards)

    # -- calibration -----------------------------------------------------------
    def build_cal(self) -> Calibration:
        for k in ("open", "short", "load"):
            if k not in self.standards:
                raise CalibrationError(f"standard manquant : {k}")
        f = self.standards["open"][0]
        for k in ("short", "load", "thru"):
            if k in self.standards and not _same_grid(self.standards[k][0], f):
                raise CalibrationError(f"{k}: grille de fréquences différente de open")
        thru = self.standards["thru"][1][:, 1] if "thru" in self.standards else None
        self.cal = Calibration.from_measurements(
            f,
            self.standards["open"][1][:, 0],
            self.standards["short"][1][:, 0],
            self.standards["load"][1][:, 0],
            thru_s21=thru,
        )
        self.cal.delay_s = self.delay_s
        self.cal_path = None
        self.apply_cal = True
        return self.cal

    def save_cal(self, path):
        if self.cal is None:
            raise CalibrationError("aucune calibration à enregistrer")
        self.cal.delay_s = self.delay_s
        self.cal.save(path)
        self.cal_path = str(path)

    def load_cal(self, path):
        self.cal = Calibration.load(path)
        self.delay_s = self.cal.delay_s
        self.cal_path = str(path)
        self.apply_cal = True
        return self.cal

    # -- références ε* (air / court-circuit / liquide) ---------------------------
    def save_refs(self, path):
        """Sauve les références ε* capturées (.npz) pour éviter de les refaire.

        Elles ne restent valables que sur la même grille de fréquences, le même
        câble/capteur et la même puissance de source ; `process()` vérifie la grille.
        """
        got = [k for k in EPS_STANDARDS if k in self.standards]
        if not got:
            raise CalibrationError("aucune référence ε* à enregistrer")
        data = {"names": np.array(got), "ref_liquid": np.array(self.ref_liquid)}
        for k in got:
            f, s = self.standards[k]
            data[f"f_{k}"] = f
            data[f"s_{k}"] = s
        np.savez(path, **data)

    def load_refs(self, path):
        """Charge des références ε* sauvées ; renvoie la liste des noms chargés."""
        try:
            with np.load(path, allow_pickle=False) as z:
                names = [str(n) for n in z["names"]]
                loaded = {k: (np.array(z[f"f_{k}"], dtype=float),
                              np.array(z[f"s_{k}"], dtype=complex)) for k in names}
                liquid = str(z["ref_liquid"]) if "ref_liquid" in z else None
        except (OSError, KeyError, ValueError) as e:
            raise CalibrationError(f"références illisibles : {e}") from e
        self.standards.update(loaded)
        if liquid in REFERENCE_LIQUIDS:
            self.ref_liquid = liquid
        return names

    # -- traitement ------------------------------------------------------------
    def process(self, freqs, s_raw) -> Result:
        freqs = np.asarray(freqs, dtype=float)
        s_raw = np.asarray(s_raw, dtype=complex)
        notes: List[str] = []
        s, calibrated = s_raw, False
        if self.apply_cal and self.cal is not None:
            try:
                s, calibrated = self.cal.apply(freqs, s_raw, delay_s=self.delay_s), True
            except CalibrationError as e:
                notes.append(f"calibration non appliquée : {e}")
        eps = None
        if self.eps_mode != "off":
            try:
                eps = self._eps(freqs, s_raw, s, calibrated, notes)
                if eps is not None and self.smooth > 1:
                    eps = smooth_complex(eps, self.smooth)
            except (CalibrationError, ValueError) as e:
                notes.append(f"ε* indisponible : {e}")
        return Result(s=s, calibrated=calibrated, eps=eps, notes=notes)

    def _eps(self, freqs, s_raw, s, calibrated, notes):
        if self.eps_mode == "capacitive":
            if not calibrated:
                notes.append("ε* capacitif sur S11 non calibré")
            return eps_capacitive(freqs, s[:, 0], self.c0)
        if self.eps_mode == "three_ref":
            g = []
            for name in EPS_STANDARDS:
                if name not in self.standards:
                    raise CalibrationError(f"référence manquante : {name}")
                fx, sx = self.standards[name]
                if not _same_grid(fx, freqs):
                    raise CalibrationError(f"{name}: grille différente du sweep courant")
                g.append(sx[:, 0])
            eps_ref = REFERENCE_LIQUIDS[self.ref_liquid](freqs)
            # birapport : invariant, donc sur S11 brut avec références brutes
            return eps_three_reference(s_raw[:, 0], g[0], g[1], g[2], eps_ref)
        raise ValueError(f"mode ε* inconnu: {self.eps_mode}")
