"""Flux profondeur Z16 de la RealSense D4xx en V4L2 brut, et normale à la surface.

OpenCV (videoio) ne sait pas négocier le format Z16 du nœud profondeur (blocage à
l'ouverture constaté sur ce banc) : on parle directement à uvcvideo par ioctl + mmap
(S_FMT, REQBUFS, STREAMON, select, DQBUF). Aucune dépendance librealsense ni OpenCV ici.

Même modèle « boîte aux lettres » que vision.camera : le thread garde la DERNIÈRE
carte de profondeur ; le thread Qt la lit avec latest() et calcule la normale locale
(ajustement de plan robuste) au rythme de l'affichage.

Limites assumées (suffisant pour une direction de normale) :
- profondeur NON alignée sur le RGB : correspondance approchée par rapport des FOV
  (RGB 90°×65°, profondeur 87°×58°, D455) autour du centre image, voir rgb_to_depth_px ;
- intrinsèques approchées par les FOV nominaux (fx ≈ 447 px à 848 de large) ;
- unités brutes du capteur (la direction de la normale est invariante d'échelle).
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import select
import struct
import threading
import time
from math import radians, tan
from typing import Optional, Tuple

import numpy as np

# -- ABI V4L2 (vérifiée sur aarch64 / kernel 5.10) -------------------------------------
_VIDIOC_S_FMT = 0xC0D05605
_VIDIOC_REQBUFS = 0xC0145608
_VIDIOC_QUERYBUF = 0xC0585609
_VIDIOC_QBUF = 0xC058560F
_VIDIOC_DQBUF = 0xC0585611
_VIDIOC_STREAMON = 0x40045612
_VIDIOC_STREAMOFF = 0x40045613
_CAPTURE = 1  # V4L2_BUF_TYPE_VIDEO_CAPTURE
_MMAP = 1  # V4L2_MEMORY_MMAP
_Z16 = ord("Z") | (ord("1") << 8) | (ord("6") << 16) | (ord(" ") << 24)


class _PixFormat(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in (
        "width", "height", "pixelformat", "field", "bytesperline", "sizeimage",
        "colorspace", "priv", "flags", "enc", "quantization", "xfer_func")]


class _Format(ctypes.Structure):
    # l'union du kernel contient des pointeurs : alignée sur 8, taille totale 208
    class _U(ctypes.Union):
        _fields_ = [("pix", _PixFormat), ("raw", ctypes.c_uint8 * 200)]

    _fields_ = [("type", ctypes.c_uint32), ("_pad", ctypes.c_uint32), ("u", _U)]


class _RequestBuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 2)]


class _Timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _Buffer(ctypes.Structure):
    class _M(ctypes.Union):
        _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                    ("planes", ctypes.c_void_p), ("fd", ctypes.c_int32)]

    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", ctypes.c_long * 2),
                ("timecode", _Timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m", _M),
                ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]


class DepthError(RuntimeError):
    pass


class DepthStream(threading.Thread):
    """Thread de capture Z16. open() échoue immédiatement si le flux ne démarre pas."""

    def __init__(self, device: str = "/dev/video0", width: int = 848, height: int = 480):
        super().__init__(daemon=True, name="depth")
        self.device = device
        self.width, self.height = width, height
        self.error: Optional[str] = None  # consulté par le thread Qt
        self._fd: Optional[int] = None
        self._maps: list = []
        self._latest: Optional[Tuple[float, np.ndarray]] = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    # -- appelé par le thread Qt AVANT start() ------------------------------------------
    def open(self):
        try:
            fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        except OSError as e:
            raise DepthError(f"profondeur : {self.device} inaccessible ({e})") from e
        try:
            fmt = _Format()
            fmt.type = _CAPTURE
            fmt.u.pix.width, fmt.u.pix.height = self.width, self.height
            fmt.u.pix.pixelformat = _Z16
            fmt.u.pix.field = 1  # NONE
            fcntl.ioctl(fd, _VIDIOC_S_FMT, fmt)
            if fmt.u.pix.pixelformat != _Z16:
                raise DepthError(f"profondeur : Z16 refusé par {self.device}")
            self.width, self.height = fmt.u.pix.width, fmt.u.pix.height
            req = _RequestBuffers()
            req.count, req.type, req.memory = 4, _CAPTURE, _MMAP
            fcntl.ioctl(fd, _VIDIOC_REQBUFS, req)
            for i in range(req.count):
                b = _Buffer()
                b.index, b.type, b.memory = i, _CAPTURE, _MMAP
                fcntl.ioctl(fd, _VIDIOC_QUERYBUF, b)
                m = mmap.mmap(fd, b.length, mmap.MAP_SHARED,
                              mmap.PROT_READ | mmap.PROT_WRITE, offset=b.m.offset)
                self._maps.append(m)
                fcntl.ioctl(fd, _VIDIOC_QBUF, b)
            fcntl.ioctl(fd, _VIDIOC_STREAMON, struct.pack("i", _CAPTURE))
        except DepthError:
            self._cleanup(fd)
            raise
        except OSError as e:
            self._cleanup(fd)
            raise DepthError(
                f"profondeur : démarrage impossible sur {self.device} ({e}). "
                "Si l'erreur persiste, débrancher/rebrancher la caméra."
            ) from e
        self._fd = fd

    def _cleanup(self, fd):
        try:
            fcntl.ioctl(fd, _VIDIOC_STREAMOFF, struct.pack("i", _CAPTURE))
        except OSError:
            pass
        for m in self._maps:
            m.close()
        self._maps = []
        os.close(fd)

    # -- thread -------------------------------------------------------------------------
    def run(self):
        assert self._fd is not None, "open() d'abord"
        fd = self._fd
        misses = 0
        try:
            while not self._stopping.is_set():
                r, _, _ = select.select([fd], [], [], 1.0)
                if not r:
                    misses += 1
                    if misses >= 3:
                        self.error = "profondeur : plus de trames (flux interrompu ?)"
                        break
                    continue
                misses = 0
                b = _Buffer()
                b.type, b.memory = _CAPTURE, _MMAP
                try:
                    fcntl.ioctl(fd, _VIDIOC_DQBUF, b)
                except OSError:
                    continue  # EAGAIN : trame pas prête malgré select
                t = time.monotonic()
                raw = self._maps[b.index][: self.width * self.height * 2]
                depth = np.frombuffer(raw, dtype=np.uint16).reshape(self.height, self.width).copy()
                fcntl.ioctl(fd, _VIDIOC_QBUF, b)
                with self._lock:
                    self._latest = (t, depth)
        finally:
            self._cleanup(fd)
            self._fd = None

    # -- appelés par le thread Qt ---------------------------------------------------------
    def latest(self) -> Optional[Tuple[float, np.ndarray]]:
        with self._lock:
            return self._latest

    def stop(self, timeout: float = 3.0):
        self._stopping.set()
        if self.is_alive():
            self.join(timeout)
        elif self._fd is not None:  # ouvert mais jamais démarré
            self._cleanup(self._fd)
            self._fd = None


# -- géométrie ---------------------------------------------------------------------------

# FOV nominaux D455 : profondeur 87°×58°, RGB 90°×65° (même définition 848×480 ici).
_HFOV_D, _VFOV_D = radians(87.0), radians(58.0)
_HFOV_C, _VFOV_C = radians(90.0), radians(65.0)


def rgb_to_depth_px(x: float, y: float, width: int = 848, height: int = 480):
    """Pixel RGB → pixel profondeur, par rapport des FOV autour du centre (approché).

    Ignore la parallaxe entre modules (< 10 px au-delà de ~30 cm) : suffisant pour
    choisir la zone d'ajustement du plan, pas pour une mesure métrique.
    """
    cx, cy = width / 2.0, height / 2.0
    sx = tan(_HFOV_D / 2) / tan(_HFOV_C / 2)
    sy = tan(_VFOV_D / 2) / tan(_VFOV_C / 2)
    return cx + (x - cx) / sx, cy + (y - cy) / sy


def depth_intrinsics(width: int = 848, height: int = 480):
    """(fx, fy, cx, cy) approchés du module profondeur à partir des FOV nominaux."""
    fx = (width / 2.0) / tan(_HFOV_D / 2)
    fy = (height / 2.0) / tan(_VFOV_D / 2)
    return fx, fy, width / 2.0, height / 2.0


def surface_normal(
    depth: np.ndarray,
    u: float,
    v: float,
    half: int = 30,
    min_valid: int = 120,
) -> Optional[np.ndarray]:
    """Normale unitaire (vers la caméra, n_z < 0) du plan local autour du pixel (u, v).

    Ajustement Z = aX + bY + c sur les points 3D de la fenêtre (2·half+1)², puis un
    passage de rejet des points à plus de 3 MAD du plan (la canule au-dessus de la
    surface est éliminée comme aberrante). None si pas assez de profondeur valide.
    """
    h, w = depth.shape
    fx, fy, cx, cy = depth_intrinsics(w, h)
    u0, v0 = int(round(u)), int(round(v))
    us = slice(max(0, u0 - half), min(w, u0 + half + 1))
    vs = slice(max(0, v0 - half), min(h, v0 + half + 1))
    z = depth[vs, us].astype(np.float64)
    uu, vv = np.meshgrid(np.arange(us.start, us.stop), np.arange(vs.start, vs.stop))
    ok = z > 0
    if int(ok.sum()) < min_valid:
        return None
    z = z[ok]
    x = (uu[ok] - cx) * z / fx
    y = (vv[ok] - cy) * z / fy
    a = np.column_stack([x, y, np.ones_like(z)])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    res = z - a @ coef
    mad = np.median(np.abs(res - np.median(res)))
    if mad > 0:
        keep = np.abs(res) <= 3.0 * 1.4826 * mad
        if int(keep.sum()) >= min_valid:
            coef, *_ = np.linalg.lstsq(a[keep], z[keep], rcond=None)
    n = np.array([coef[0], coef[1], -1.0])
    return n / np.linalg.norm(n)
