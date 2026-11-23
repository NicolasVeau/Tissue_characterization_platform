"""Identification de la zone du corps : LDA régularisé sur descripteurs invariants à la pression.

Pas de dépendance supplémentaire (numpy seul). Principe :
  - descripteurs : `pressure.band_features` (forme normalisée de ε*−1 et rapport de pertes) ;
  - classifieur : LDA à covariance commune rétrécie (shrinkage), a priori uniformes, donc
    P(zone | x) = softmax(−d²/2) avec d² la distance de Mahalanobis aux moyennes de classe ;
  - rejet « inconnu » : d² minimal supérieur à un seuil estimé par validation croisée.
La validation est faite par session (une pose = un groupe) : jamais par sweep, les sweeps
d'une même pose étant quasi identiques.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ZoneClassifier:
    def __init__(self, shrinkage: float = 0.2):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage dans [0, 1]")
        self.shrinkage = shrinkage
        self.classes_: np.ndarray = np.array([])
        self.d2_max: Optional[float] = None  # seuil de rejet (None = pas de rejet)

    def fit(self, X, y) -> "ZoneClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X (n, p) et y (n,) incohérents")
        if not np.all(np.isfinite(X)):
            raise ValueError("descripteurs non finis")
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("au moins 2 zones sont nécessaires")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        z = (X - self.mean_) / self.scale_
        self.means_ = np.stack([z[y == c].mean(axis=0) for c in self.classes_])
        idx = np.searchsorted(self.classes_, y)
        r = z - self.means_[idx]
        dof = max(len(z) - k, 1)
        sw = r.T @ r / dof
        p = sw.shape[0]
        sw = (1 - self.shrinkage) * sw + self.shrinkage * np.trace(sw) / p * np.eye(p)
        self.prec_ = np.linalg.pinv(sw)
        return self

    def d2(self, X) -> np.ndarray:
        """Distances de Mahalanobis² (n, K) aux moyennes de classe."""
        z = (np.atleast_2d(np.asarray(X, dtype=float)) - self.mean_) / self.scale_
        diff = z[:, None, :] - self.means_[None, :, :]
        return np.einsum("nkp,pq,nkq->nk", diff, self.prec_, diff)

    def predict_proba(self, X) -> np.ndarray:
        a = -0.5 * self.d2(X)
        a -= a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict_one(self, x) -> Tuple[Optional[str], Dict[str, float], float]:
        """(zone ou None si rejeté, {zone: probabilité}, d² minimal)."""
        d = self.d2(x)[0]
        p = self.predict_proba(x)[0]
        dmin = float(d.min())
        best = str(self.classes_[int(np.argmax(p))])
        unknown = self.d2_max is not None and dmin > self.d2_max
        return (
            (None if unknown else best),
            {str(c): float(v) for c, v in zip(self.classes_, p)},
            dmin,
        )


class Dataset:
    """Descripteurs étiquetés (zone, session) + métadonnées de grille et de normalisation."""

    def __init__(self, meta: Optional[dict] = None):
        self.X: List[np.ndarray] = []
        self.zones: List[str] = []
        self.sessions: List[str] = []  # identifiant unique de pose, "zone#k"
        self.meta = dict(meta or {})

    def __len__(self):
        return len(self.X)

    def n_sessions(self, zone: str) -> int:
        return len({s for z, s in zip(self.zones, self.sessions) if z == zone})

    def new_session(self, zone: str) -> str:
        return f"{zone}#{self.n_sessions(zone) + 1}"

    def add(self, x, zone: str, session: str):
        self.X.append(np.asarray(x, dtype=float))
        self.zones.append(zone)
        self.sessions.append(session)

    def arrays(self):
        return np.array(self.X), np.array(self.zones), np.array(self.sessions)

    def save(self, path) -> Path:
        path = Path(path)
        X, z, s = self.arrays() if self.X else (np.zeros((0, 0)), np.array([]), np.array([]))
        import json

        np.savez(path, X=X, zones=z, sessions=s, meta=json.dumps(self.meta))
        return path

    @classmethod
    def load(cls, path) -> "Dataset":
        import json

        with np.load(path, allow_pickle=False) as d:
            ds = cls(json.loads(str(d["meta"])))
            for x, z, s in zip(d["X"], d["zones"], d["sessions"]):
                ds.add(x, str(z), str(s))
        return ds


def cross_validate(
    X, zones, sessions, shrinkage: float = 0.2, reject_quantile: float = 0.95
) -> dict:
    """Validation « une session laissée de côté ».

    Retourne accuracy par sweep et par session (vote majoritaire), matrice de confusion
    (lignes = vraie zone), rappel par zone, et `d2_max` = quantile des d² de la vraie zone
    sur les sessions laissées de côté (seuil de rejet « inconnu »).
    """
    X = np.asarray(X, dtype=float)
    zones = np.asarray(zones)
    sessions = np.asarray(sessions)
    classes = np.unique(zones)
    ci = {c: i for i, c in enumerate(classes)}
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    sess_conf = np.zeros_like(conf)
    d2_true: List[float] = []
    for s in np.unique(sessions):
        te = sessions == s
        tr = ~te
        true = zones[te][0]
        if len(np.unique(zones[tr])) < 2 or true not in set(zones[tr]):
            continue  # classe absente de l'apprentissage : session non évaluable
        m = ZoneClassifier(shrinkage).fit(X[tr], zones[tr])
        d = m.d2(X[te])
        pred = m.classes_[np.argmax(m.predict_proba(X[te]), axis=1)]
        for p_ in pred:
            conf[ci[true], ci[p_]] += 1
        votes = np.bincount([ci[p_] for p_ in pred], minlength=len(classes))
        sess_conf[ci[true], int(np.argmax(votes))] += 1
        d2_true.extend(d[:, int(np.searchsorted(m.classes_, true))])
    n = conf.sum()
    ns = sess_conf.sum()
    recall = np.divide(
        np.diag(sess_conf), sess_conf.sum(axis=1), out=np.full(len(classes), np.nan),
        where=sess_conf.sum(axis=1) > 0,
    )  # fmt: skip
    return {
        "classes": [str(c) for c in classes],
        "confusion": conf,
        "session_confusion": sess_conf,
        "acc_sweeps": float(np.trace(conf) / n) if n else float("nan"),
        "acc_sessions": float(np.trace(sess_conf) / ns) if ns else float("nan"),
        "recall_sessions": recall,
        "d2_max": float(np.quantile(d2_true, reject_quantile)) if d2_true else None,
        "n_sessions": int(ns),
    }


def top_confusions(result: dict, n: int = 3) -> List[Tuple[str, str, int]]:
    """Paires (vraie, prédite, nb de sessions) les plus confondues."""
    c = result["session_confusion"].copy()
    np.fill_diagonal(c, 0)
    names = result["classes"]
    out = []
    for flat in np.argsort(c, axis=None)[::-1][:n]:
        i, j = divmod(int(flat), c.shape[1])
        if c[i, j] > 0:
            out.append((names[i], names[j], int(c[i, j])))
    return out


def format_report(result: dict, shrinkage: float) -> str:
    lines = [
        f"{result['n_sessions']} sessions évaluées (une laissée de côté à la fois)",
        f"Exactitude par session : {100 * result['acc_sessions']:.1f} %",
        f"Exactitude par sweep   : {100 * result['acc_sweeps']:.1f} %",
        f"Rétrécissement λ = {shrinkage:g}",
    ]
    if result["d2_max"] is not None:
        lines.append(f"Seuil « inconnu » d² > {result['d2_max']:.1f}")
    lines.append("Rappel par zone :")
    for n_, r in zip(result["classes"], result["recall_sessions"]):
        lines.append(f"  {n_:<14s} {'—' if np.isnan(r) else f'{100 * r:5.1f} %'}")
    conf = top_confusions(result)
    if conf:
        lines.append("Confusions principales :")
        lines += [f"  {a} → {b} : {k}" for a, b, k in conf]
    return "\n".join(lines)


def features_for(freqs, eps, f_ref: float, n_bands: int = 8) -> Optional[np.ndarray]:
    """Descripteurs d'un sweep, ou None si indisponibles (pas d'ε*, pas de contact, NaN)."""
    from vna_app.processing.pressure import band_features

    if eps is None:
        return None
    try:
        x = band_features(freqs, eps, f_ref, n_bands)
    except ValueError:
        return None
    return x if np.all(np.isfinite(x)) else None


def grid_signature(freqs: Sequence[float]) -> List[float]:
    f = np.asarray(freqs, dtype=float)
    return [float(f[0]), float(f[-1]), float(len(f))]
