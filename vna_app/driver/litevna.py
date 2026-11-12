"""Driver LiteVNA 64 (protocole binaire NanoVNA V2 / SAA2).

Référence : docs/protocol.md. Les données renvoyées sont BRUTES (non calibrées).
Ce module est synchrone ; l'appelant GUI doit l'exécuter hors du thread Qt.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import serial

# --- Opcodes -----------------------------------------------------------------
OP_NOP = 0x00
OP_INDICATE = 0x0D
OP_READ = 0x10
OP_READ2 = 0x11
OP_READ4 = 0x12
OP_READFIFO = 0x18
OP_WRITE = 0x20
OP_WRITE2 = 0x21
OP_WRITE4 = 0x22
OP_WRITE8 = 0x23

# --- Registres ---------------------------------------------------------------
REG_SWEEP_START = 0x00
REG_SWEEP_STEP = 0x10
REG_SWEEP_POINTS = 0x20
REG_VALUES_PER_FREQ = 0x22
REG_FIFO = 0x30
REG_DEVICE_VARIANT = 0xF0
REG_PROTOCOL_VERSION = 0xF1
REG_HW_REVISION = 0xF2
REG_FW_MAJOR = 0xF3
REG_FW_MINOR = 0xF4

VARIANT_SAA2 = 0x02
INDICATE_REPLY = 0x32

FRAME_FMT = "<iiiiiiH6x"
FRAME_SIZE = struct.calcsize(FRAME_FMT)  # 32
assert FRAME_SIZE == 32

MAX_POINTS = 65535
MAX_FIFO_READ = 255


class LiteVNAError(Exception):
    """Erreur générique du driver."""


class LiteVNAIncomplete(LiteVNAError):
    """Des points n'ont pas reçu toutes leurs entrées."""


class LiteVNATimeout(LiteVNAError):
    """Timeout de lecture."""


@dataclass(frozen=True)
class Frame:
    """Une entrée FIFO décodée (valeurs brutes)."""

    fwd0: complex
    rev0: complex
    rev1: complex
    freq_index: int


def decode_frame(raw: bytes) -> Frame:
    """Décode une entrée FIFO de 32 octets."""
    if len(raw) != FRAME_SIZE:
        raise ValueError(f"frame de {len(raw)} octets, attendu {FRAME_SIZE}")
    f0r, f0i, r0r, r0i, r1r, r1i, idx = struct.unpack(FRAME_FMT, raw)
    return Frame(complex(f0r, f0i), complex(r0r, r0i), complex(r1r, r1i), idx)


def decode_frames(buf: bytes) -> List[Frame]:
    """Décode un buffer contenant un nombre entier d'entrées de 32 octets."""
    if len(buf) % FRAME_SIZE:
        raise ValueError("taille de buffer non multiple de 32")
    return [decode_frame(buf[i : i + FRAME_SIZE]) for i in range(0, len(buf), FRAME_SIZE)]


def assemble_sweep(frames: Iterable[Frame], points: int, averaging: int) -> np.ndarray:
    """Range les frames par freqIndex et moyenne.

    Les complexes bruts (fwd0, rev0, rev1) sont moyennés par point avant la
    division : S11 = <rev0>/<fwd0>, S21 = <rev1>/<fwd0>.
    Retourne un ndarray complexe (points, 2). Lève LiteVNAError si un point
    n'a pas exactement `averaging` entrées utilisables (les entrées au-delà
    de `averaging` pour un même index sont ignorées ; index hors plage ignorés).
    """
    acc = np.zeros((points, 3), dtype=complex)
    cnt = np.zeros(points, dtype=int)
    for fr in frames:
        k = fr.freq_index
        if k >= points or cnt[k] >= averaging:
            continue
        acc[k] += (fr.fwd0, fr.rev0, fr.rev1)
        cnt[k] += 1
    missing = np.flatnonzero(cnt < averaging)
    if missing.size:
        raise LiteVNAIncomplete(
            f"{missing.size} point(s) incomplet(s), premier index {int(missing[0])}"
        )
    acc /= averaging
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.empty((points, 2), dtype=complex)
        s[:, 0] = acc[:, 1] / acc[:, 0]
        s[:, 1] = acc[:, 2] / acc[:, 0]
    return s


class LiteVNA:
    """Accès synchrone au LiteVNA 64."""

    def __init__(self, port: str = "/dev/litevna", timeout: float = 1.0):
        self.port = port
        self.timeout = max(timeout, 1.0)
        self._ser: Optional[serial.Serial] = None
        self.variant: Optional[int] = None
        self.protocol_version: Optional[int] = None
        self.hw_revision: Optional[int] = None
        self.fw_major: Optional[int] = None
        self.fw_minor: Optional[int] = None
        self.start_hz = 0
        self.step_hz = 0
        self.points = 0
        self.averaging = 1

    # -- connexion -------------------------------------------------------------
    def connect(self) -> None:
        if self._ser is not None:
            return
        try:
            ser = serial.Serial(self.port, timeout=self.timeout, write_timeout=self.timeout)
        except (serial.SerialException, OSError) as e:
            raise LiteVNAError(f"impossible d'ouvrir {self.port}: {e}") from e
        self._ser = ser
        try:
            self._resync()
            self._read_identity()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except (serial.SerialException, OSError):
                pass
            self._ser = None

    def __enter__(self) -> "LiteVNA":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- E/S bas niveau --------------------------------------------------------
    def _port(self) -> serial.Serial:
        if self._ser is None:
            raise LiteVNAError("appareil non connecté")
        return self._ser

    def _write(self, data: bytes) -> None:
        try:
            self._port().write(data)
        except (serial.SerialException, OSError) as e:
            raise LiteVNAError(f"écriture série: {e}") from e

    def _read_exact(self, n: int, deadline: Optional[float] = None) -> bytes:
        """Lit exactement n octets ; LiteVNATimeout si `deadline` (monotonic) dépassée."""
        ser = self._port()
        buf = bytearray()
        while len(buf) < n:
            if deadline is not None and time.monotonic() > deadline:
                raise LiteVNATimeout(f"timeout: {len(buf)}/{n} octets reçus")
            try:
                chunk = ser.read(n - len(buf))
            except (serial.SerialException, OSError) as e:
                raise LiteVNAError(f"lecture série (débranché ?): {e}") from e
            if chunk:
                buf += chunk
            elif deadline is None:
                raise LiteVNATimeout(f"timeout: {len(buf)}/{n} octets reçus")
        return bytes(buf)

    def _resync(self) -> None:
        ser = self._port()
        self._write(bytes(8))
        time.sleep(0.05)
        try:
            ser.reset_input_buffer()
        except (serial.SerialException, OSError) as e:
            raise LiteVNAError(f"flush RX: {e}") from e
        self._write(bytes([OP_INDICATE]))
        reply = self._read_exact(1)
        if reply[0] != INDICATE_REPLY:
            raise LiteVNAError(f"réponse INDICATE inattendue: {reply!r} (attendu 0x32)")

    # -- registres -------------------------------------------------------------
    def read_reg(self, addr: int, size: int = 1) -> int:
        op = {1: OP_READ, 2: OP_READ2, 4: OP_READ4}[size]
        self._write(bytes([op, addr]))
        return int.from_bytes(self._read_exact(size), "little")

    def write_reg(self, addr: int, value: int, size: int = 1) -> None:
        op = {1: OP_WRITE, 2: OP_WRITE2, 4: OP_WRITE4, 8: OP_WRITE8}[size]
        self._write(bytes([op, addr]) + int(value).to_bytes(size, "little"))

    def clear_fifo(self) -> None:
        self.write_reg(REG_FIFO, 0, 1)

    def _read_identity(self) -> None:
        self.variant = self.read_reg(REG_DEVICE_VARIANT)
        self.protocol_version = self.read_reg(REG_PROTOCOL_VERSION)
        self.hw_revision = self.read_reg(REG_HW_REVISION)
        self.fw_major = self.read_reg(REG_FW_MAJOR)
        self.fw_minor = self.read_reg(REG_FW_MINOR)
        if self.variant != VARIANT_SAA2:
            raise LiteVNAError(f"deviceVariant=0x{self.variant:02x}, attendu 0x02 (SAA2/LiteVNA)")

    def device_info(self) -> dict:
        self._port()
        return {
            "variant": self.variant,
            "protocol_version": self.protocol_version,
            "hardware_revision": self.hw_revision,
            "firmware": f"{self.fw_major}.{self.fw_minor}",
        }

    # -- sweep -----------------------------------------------------------------
    def configure_sweep(
        self, start_hz: float, stop_hz: float, points: int, averaging: int = 1
    ) -> np.ndarray:
        """Configure le sweep, relit les registres pour vérifier, retourne les fréquences."""
        if not 1 <= points <= MAX_POINTS:
            raise ValueError(f"points doit être dans 1..{MAX_POINTS}")
        if not 1 <= averaging <= 0xFFFF:
            raise ValueError("averaging doit être dans 1..65535")
        start = int(round(start_hz))
        stop = int(round(stop_hz))
        if stop < start or start < 0:
            raise ValueError("plage de fréquences invalide")
        step = (stop - start) // (points - 1) if points > 1 else 0
        self._port()
        self.write_reg(REG_SWEEP_START, start, 8)
        self.write_reg(REG_SWEEP_STEP, step, 8)
        self.write_reg(REG_SWEEP_POINTS, points, 2)
        self.write_reg(REG_VALUES_PER_FREQ, averaging, 2)
        got_pts = self.read_reg(REG_SWEEP_POINTS, 2)
        got_avg = self.read_reg(REG_VALUES_PER_FREQ, 2)
        if got_pts != points or got_avg != averaging:
            raise LiteVNAError(
                f"paramètres refusés: sweepPoints={got_pts} (voulu {points}), "
                f"valuesPerFrequency={got_avg} (voulu {averaging})"
            )
        self.start_hz, self.step_hz = start, step
        self.points, self.averaging = points, averaging
        return self.frequencies()

    def frequencies(self) -> np.ndarray:
        return (self.start_hz + np.arange(self.points, dtype=np.int64) * self.step_hz).astype(float)

    def sweep(self, retries: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Lance un sweep avec la config courante. Retourne (freqs, s_raw (N,2) [S11,S21]).

        Constaté sur matériel (fw 2.2) : le premier sweep après configure_sweep() peut
        renvoyer des entrées résiduelles de l'ancien sweep (index 0..k) avant le
        nouveau, ce qui laisse des points incomplets. Dans ce cas on relance le sweep
        (la réécriture d'un registre vide le FIFO), jusqu'à `retries` fois.
        """
        if self.points < 1:
            raise LiteVNAError("configure_sweep() doit être appelé avant sweep()")
        last: Optional[LiteVNAError] = None
        for _ in range(retries + 1):
            try:
                return self.frequencies(), self._acquire_once()
            except LiteVNAIncomplete as e:
                last = e
        raise LiteVNAError(f"sweep incomplet après {retries + 1} tentatives: {last}")

    def _acquire_once(self) -> np.ndarray:
        n_total = self.points * self.averaging
        # Réécrire un registre de sweep vide le FIFO et relance un sweep propre.
        self.write_reg(REG_SWEEP_POINTS, self.points, 2)
        deadline = time.monotonic() + n_total / 500.0 + 2.0
        frames: List[Frame] = []
        while len(frames) < n_total:
            n = min(MAX_FIFO_READ, n_total - len(frames))
            self._write(bytes([OP_READFIFO, REG_FIFO, n]))
            frames.extend(decode_frames(self._read_exact(n * FRAME_SIZE, deadline)))
        return assemble_sweep(frames, self.points, self.averaging)
