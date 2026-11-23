"""Simulateur de LiteVNA 64 : même protocole binaire que l'appareil, sans matériel.

`SimSerial` émule le port série (sous-ensemble du protocole de docs/protocol.md
utilisé par le driver) et fabrique les entrées FIFO à partir d'un modèle de DUT.
`SimLiteVNA` est un `LiteVNA` branché sur ce faux port : toute la chaîne réelle
(configure_sweep, _acquire_once, streaming) est exercée telle quelle.

`open_device("sim")` retourne un `SimLiteVNA`, tout autre port un `LiteVNA` réel.
Options du port simulé : `sim:rate=300,noise=1e-3,drift=1e5,seed=1`
  rate   entrées FIFO par seconde (défaut 300, comme l'appareil ; 0 = sans limite)
  noise  écart-type du bruit complexe ajouté à S11/S21 (défaut 0)
  drift  dérive de la fréquence de résonance du DUT, en Hz/s (défaut 0)
  seed   graine du générateur de bruit (défaut 0)

Le temps vu par le DUT est le temps SIMULÉ (nombre d'entrées produites / débit
nominal de 300 entrées/s), donc reproductible quel que soit `rate`.
`SimSerial()` construit directement est sans limite de débit (tests rapides) ;
sans limite, le streaming produit plus vite qu'on ne consomme et écarte des sweeps.

Les tests d'injection de fautes (résiduels, débranchement) restent dans
tests/test_driver.py avec leur propre faux port.
"""

from __future__ import annotations

import struct
import time
from typing import Callable, Optional, Tuple

import numpy as np

from vna_app.driver import litevna as L

NOMINAL_RATE = 300.0  # entrées FIFO/s mesurées sur l'appareil (CLAUDE.md)
FWD_REF = 1 << 20  # amplitude entière de l'onde incidente simulée

DutFn = Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]


class ResonatorDUT:
    """Résonateur à un pôle vu en réflexion et en transmission.

    S11 = 1 − d / (1 + j·2Q·(f − f0)/f0)   (creux de profondeur d à f0)
    S21 = t / (1 + j·2Q·(f − f0)/f0)
    f0 dérive linéairement : f0(t) = f0 + drift_hz_s · t.
    """

    def __init__(
        self,
        f0: float = 1.5e9,
        q: float = 30.0,
        depth: float = 0.8,
        trans: float = 0.5,
        drift_hz_s: float = 0.0,
    ):
        self.f0, self.q, self.depth, self.trans = float(f0), float(q), float(depth), float(trans)
        self.drift_hz_s = float(drift_hz_s)

    def __call__(self, freqs: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        f0 = self.f0 + self.drift_hz_s * t
        lor = 1.0 / (1.0 + 2j * self.q * (np.asarray(freqs, dtype=float) - f0) / f0)
        return 1.0 - self.depth * lor, self.trans * lor


class SimSerial:
    """Faux port série : répond comme un LiteVNA dont le DUT est `dut(freqs, t)`.

    errors=(a, b, c) applique à S11 le modèle d'erreur 1 port de calibration/solt.py
    (Γm = a + b·Γ + c·Γ·Γm), pour produire des données « brutes » à calibrer.
    """

    def __init__(
        self,
        dut: Optional[DutFn] = None,
        rate: float = 0.0,
        noise: float = 0.0,
        seed: int = 0,
        errors: Optional[Tuple[complex, complex, complex]] = None,
        firmware: Tuple[int, int] = (2, 2),
    ):
        self.dut: DutFn = dut or ResonatorDUT()
        self.rate = float(rate)
        self.noise = float(noise)
        self.errors = errors
        self._rng = np.random.default_rng(seed)
        self.regs = {
            L.REG_DEVICE_VARIANT: L.VARIANT_SAA2,
            L.REG_PROTOCOL_VERSION: 1,
            L.REG_HW_REVISION: 2,
            L.REG_FW_MAJOR: firmware[0],
            L.REG_FW_MINOR: firmware[1],
            L.REG_SWEEP_START: 0,
            L.REG_SWEEP_STEP: 0,
            L.REG_SWEEP_POINTS: 0,
            L.REG_VALUES_PER_FREQ: 1,
            L.REG_ADF4350_POWER: 3,
            L.REG_SI5351_POWER: 1,
        }
        self.rx = bytearray()
        self.closed = False
        self.produced = 0  # entrées FIFO produites depuis l'ouverture (horloge simulée)
        self._pos = 0  # position dans le balayage courant (0 .. points*avg − 1)
        self._due = time.monotonic()  # instant réel où la prochaine entrée est disponible

    # -- API pyserial utilisée par le driver -----------------------------------
    def reset_input_buffer(self) -> None:
        self.rx.clear()

    def close(self) -> None:
        self.closed = True

    def write(self, data) -> None:
        d = bytes(data)
        if not d:
            return
        op = d[0]
        if op == L.OP_NOP:
            return
        if op == L.OP_INDICATE:
            self.rx += bytes([L.INDICATE_REPLY])
        elif op in (L.OP_READ, L.OP_READ2, L.OP_READ4):
            size = {L.OP_READ: 1, L.OP_READ2: 2, L.OP_READ4: 4}[op]
            val = int(self.regs.get(d[1], 0)) & ((1 << (8 * size)) - 1)
            self.rx += val.to_bytes(size, "little")
        elif op in (L.OP_WRITE, L.OP_WRITE2, L.OP_WRITE4, L.OP_WRITE8):
            size = {L.OP_WRITE: 1, L.OP_WRITE2: 2, L.OP_WRITE4: 4, L.OP_WRITE8: 8}[op]
            addr = d[1]
            if addr == L.REG_FIFO:
                self._restart()
                return
            self.regs[addr] = int.from_bytes(d[2 : 2 + size], "little")
            if addr in (
                L.REG_SWEEP_START,
                L.REG_SWEEP_STEP,
                L.REG_SWEEP_POINTS,
                L.REG_VALUES_PER_FREQ,
            ):
                self._restart()  # écrire un registre de sweep vide le FIFO et relance
        elif op == L.OP_READFIFO:
            self.rx += self._make_entries(d[2])

    def read(self, n: int) -> bytes:
        out, self.rx = bytes(self.rx[:n]), self.rx[n:]
        return out

    # -- émulation ---------------------------------------------------------------
    @property
    def sim_time(self) -> float:
        """Temps simulé (s) : entrées produites au débit nominal de l'appareil."""
        return self.produced / NOMINAL_RATE

    def _cnoise(self, n: int) -> np.ndarray:
        return self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)

    def _restart(self) -> None:
        self._pos = 0
        self._due = time.monotonic()

    def _make_entries(self, n: int) -> bytes:
        points = int(self.regs[L.REG_SWEEP_POINTS])
        avg = max(1, int(self.regs[L.REG_VALUES_PER_FREQ]))
        if points < 1 or n < 1:
            return b""
        start = int(self.regs[L.REG_SWEEP_START])
        step = int(self.regs[L.REG_SWEEP_STEP])
        total = points * avg
        pos = (self._pos + np.arange(n)) % total
        k = pos // avg
        t = (self.produced + np.arange(n)) / NOMINAL_RATE
        freqs = start + k.astype(np.int64) * step
        s11 = np.empty(n, dtype=complex)
        s21 = np.empty(n, dtype=complex)
        for i in range(n):  # le DUT dépend du temps : un appel par entrée
            a, b = self.dut(freqs[i : i + 1].astype(float), float(t[i]))
            s11[i], s21[i] = a[0], b[0]
        if self.errors is not None:
            ea, eb, ec = self.errors
            s11 = (ea + eb * s11) / (1.0 - ec * s11)
        if self.noise > 0.0:
            s11 = s11 + self.noise * self._cnoise(n)
            s21 = s21 + self.noise * self._cnoise(n)
        self._pos = int((self._pos + n) % total)
        self.produced += n
        if self.rate > 0.0:  # cadence réelle : l'appareil ne produit que `rate` entrées/s
            self._due = max(self._due, time.monotonic() - 0.5) + n / self.rate
            delay = self._due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        out = bytearray()
        for i in range(n):
            out += struct.pack(
                L.FRAME_FMT,
                FWD_REF,
                0,
                int(round(s11[i].real * FWD_REF)),
                int(round(s11[i].imag * FWD_REF)),
                int(round(s21[i].real * FWD_REF)),
                int(round(s21[i].imag * FWD_REF)),
                int(k[i]) & 0xFFFF,
            )
        return bytes(out)


class SimLiteVNA(L.LiteVNA):
    """`LiteVNA` relié à un `SimSerial` : même API, aucun matériel."""

    def __init__(self, sim: Optional[SimSerial] = None, port: str = "sim", **sim_kwargs):
        super().__init__(port=port)
        self.sim = sim or SimSerial(**sim_kwargs)

    def connect(self) -> None:
        if self._ser is not None:
            return
        self.sim.closed = False
        self._ser = self.sim  # type: ignore[assignment]
        try:
            self._write(bytes([L.OP_INDICATE]))
            if self._read_exact(1)[0] != L.INDICATE_REPLY:
                raise L.LiteVNAError("simulateur : réponse INDICATE inattendue")
            self._read_identity()
        except Exception:
            self.close()
            raise


def is_sim_port(port: str) -> bool:
    return port == "sim" or port.startswith("sim:")


def parse_sim_port(port: str) -> dict:
    """'sim:rate=300,noise=1e-3,drift=1e5,seed=1' -> dict d'options numériques."""
    opts = {"rate": NOMINAL_RATE, "noise": 0.0, "drift": 0.0, "seed": 0.0}
    _, _, spec = port.partition(":")
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, sep, val = item.partition("=")
        if not sep or key not in opts:
            raise ValueError(f"option de simulateur inconnue: {item!r} (attendu {sorted(opts)})")
        opts[key] = float(val)
    return opts


def open_device(port: str = "/dev/litevna") -> L.LiteVNA:
    """Retourne un `SimLiteVNA` si `port` commence par 'sim', sinon un `LiteVNA` réel.

    L'appareil n'est pas connecté : appeler `connect()` ou l'utiliser avec `with`.
    """
    if not is_sim_port(port):
        return L.LiteVNA(port)
    o = parse_sim_port(port)
    sim = SimSerial(
        dut=ResonatorDUT(drift_hz_s=o["drift"]),
        rate=o["rate"],
        noise=o["noise"],
        seed=int(o["seed"]),
    )
    return SimLiteVNA(sim, port=port)
