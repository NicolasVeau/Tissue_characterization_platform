"""Détection de la bague vert clair sur images synthétiques."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vna_app.vision.tracker import RingTracker  # noqa: E402


def green_bgr(h=60, s=150, v=220):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def blank(w=320, h=240):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 40, 40)  # fond gris sombre
    return img


def test_detects_ring_center():
    img = blank()
    cv2.circle(img, (200, 150), 12, green_bgr(), -1)
    tip = RingTracker().find(img)
    assert tip is not None
    assert abs(tip.x - 200) < 2 and abs(tip.y - 150) < 2
    assert tip.area > 100


def test_no_green_returns_none():
    assert RingTracker().find(blank()) is None


def test_red_blob_ignored():
    img = blank()
    cv2.circle(img, (100, 100), 15, (0, 0, 255), -1)  # rouge
    assert RingTracker().find(img) is None


def test_largest_blob_wins():
    img = blank()
    cv2.circle(img, (60, 60), 5, green_bgr(), -1)
    cv2.circle(img, (250, 180), 14, green_bgr(), -1)
    tip = RingTracker().find(img)
    assert abs(tip.x - 250) < 2 and abs(tip.y - 180) < 2


def test_min_area_filters_speckle():
    img = blank()
    cv2.circle(img, (60, 60), 5, green_bgr(), -1)  # ~78 px²
    assert RingTracker(min_area=500).find(img) is None


def test_desaturated_green_rejected():
    img = blank()
    cv2.circle(img, (100, 100), 12, green_bgr(s=30), -1)  # vert délavé < s_min
    assert RingTracker(s_min=70).find(img) is None
