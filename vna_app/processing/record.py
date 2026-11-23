"""Enregistrement et relecture de séries de sweeps horodatés (fichier .npz).

Un enregistrement est une suite de sweeps sur UNE grille de fréquences :
  freqs       (N,)     float64, Hz
  s           (M,N,2)  complex128, [S11, S21] par sweep
  seq         (M,)     int64, numéro d'ordre livré par le streaming (trous = sweeps perdus)
  timestamps  (M,)     float64, time.time() à la fin de chaque sweep
  meta        JSON     paramètres d'acquisition, firmware, calibration, ...

Le mode fréquence fixe (start = stop) s'enregistre de la même façon : `freqs` est
alors constant et chaque « sweep » est un bloc de N échantillons temporels.

Format npz numpy, sans dépendance nouvelle. Les sweeps sont gardés en mémoire et
écrits à la fermeture : 1 h à 1,5 sweep/s en 201 points ≈ 35 Mo.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional

import numpy as np

from vna_app.driver.litevna import SweepResult

FORMAT_VERSION = 1


class RecordError(Exception):
    pass


class Recording(NamedTuple):
    freqs: np.ndarray  # (N,)
    s: np.ndarray  # (M,N,2)
    seq: np.ndarray  # (M,)
    timestamps: np.ndarray  # (M,)
    meta: dict

    def __len__(self) -> int:  # nombre de sweeps
        return int(self.s.shape[0])

    @property
    def points(self) -> int:
        return int(self.freqs.shape[0])

    def results(self) -> Iterator[SweepResult]:
        for i in range(len(self)):
            yield SweepResult(int(self.seq[i]), float(self.timestamps[i]), self.freqs, self.s[i])

    def lost(self) -> int:
        """Nombre de sweeps manquants d'après les trous de `seq`."""
        if len(self) < 2:
            return 0
        return int(np.sum(np.diff(self.seq) - 1))


class SweepRecorder:
    """Accumule des `SweepResult` et écrit le fichier à `close()` (ou en sortie de `with`)."""

    def __init__(self, path, meta: Optional[dict] = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self._freqs: Optional[np.ndarray] = None
        self._s: List[np.ndarray] = []
        self._seq: List[int] = []
        self._ts: List[float] = []
        self._closed = False

    def __len__(self) -> int:
        return len(self._s)

    def append(self, result: SweepResult) -> None:
        self.add(result.freqs, result.s_raw, seq=result.seq, timestamp=result.timestamp)

    def add(self, freqs, s, seq: Optional[int] = None, timestamp: Optional[float] = None) -> None:
        if self._closed:
            raise RecordError("enregistreur fermé")
        freqs = np.asarray(freqs, dtype=float)
        s = np.asarray(s, dtype=complex)
        if s.shape != (freqs.shape[0], 2):
            raise RecordError(f"s doit être de forme (N,2), reçu {s.shape}")
        if self._freqs is None:
            self._freqs = freqs.copy()
        elif freqs.shape != self._freqs.shape or np.max(np.abs(freqs - self._freqs)) > 1.0:
            raise RecordError("grille de fréquences différente : un fichier par configuration")
        self._s.append(s.copy())
        self._seq.append(len(self._seq) if seq is None else int(seq))
        self._ts.append(float("nan") if timestamp is None else float(timestamp))

    def close(self) -> Optional[Path]:
        """Écrit le fichier. Retourne son chemin, ou None si aucun sweep n'a été reçu."""
        if self._closed:
            return self.path if self._s else None
        self._closed = True
        if not self._s:
            return None
        meta = dict(self.meta)
        meta.setdefault("date", datetime.now(timezone.utc).isoformat())
        meta["format_version"] = FORMAT_VERSION
        meta["sweeps"] = len(self._s)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:  # évite l'ajout automatique de .npz
            np.savez_compressed(
                f,
                freqs=self._freqs,
                s=np.stack(self._s),
                seq=np.asarray(self._seq, dtype=np.int64),
                timestamps=np.asarray(self._ts, dtype=float),
                meta=np.array(json.dumps(meta)),
            )
        return self.path

    def __enter__(self) -> "SweepRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_recording(path) -> Recording:
    try:
        with np.load(path, allow_pickle=False) as d:
            rec = Recording(
                np.asarray(d["freqs"], dtype=float),
                np.asarray(d["s"], dtype=complex),
                np.asarray(d["seq"], dtype=np.int64),
                np.asarray(d["timestamps"], dtype=float),
                json.loads(str(d["meta"])),
            )
    except (OSError, KeyError, ValueError) as e:
        raise RecordError(f"{path}: enregistrement illisible ({e})") from e
    if rec.s.ndim != 3 or rec.s.shape[1:] != (rec.points, 2) or rec.seq.shape != (len(rec),):
        raise RecordError(f"{path}: dimensions incohérentes")
    return rec
