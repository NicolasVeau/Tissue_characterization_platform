"""Détection de la bague de la canule sur images synthétiques.

Les teintes par défaut correspondent au banc (voir vision/tracker.py) : bague verte
H ≈ 84, gaine bleu sarcelle H ≈ 99. Les teintes rendues dépendent de l'éclairage ;
dans la GUI elles se recalent par clic (« Mesurer la bague/gaine »).
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vna_app.vision.tracker import RingTracker  # noqa: E402


def green_bgr(h=84, s=200, v=220):  # bague verte (défauts du tracker, banc 2026-11-24)
    px = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0])


def sheath_bgr():  # gaine bleu sarcelle H≈99
    return green_bgr(h=99, s=255, v=230)


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
    """Les bleus-verts peu saturés (ombres sous WB figée, tube) restent sous s_min."""
    img = blank()
    cv2.circle(img, (100, 100), 12, green_bgr(s=70), -1)
    assert RingTracker().find(img) is None


def test_sheath_hue_rejected_as_ring():
    """La gaine (H≈99) ne doit pas être prise pour la bague (H≈84)."""
    img = blank()
    cv2.circle(img, (100, 100), 20, sheath_bgr(), -1)
    assert RingTracker().find(img) is None


# -- verrouillage spatial ---------------------------------------------------------------


def test_lock_prefers_nearest_to_last_position():
    """Un gros blob qui entre dans le champ (écran…) ne vole pas le suivi."""
    trk = RingTracker()
    img = blank()
    cv2.circle(img, (100, 100), 10, green_bgr(), -1)
    assert trk.find(img) is not None  # verrou posé sur (100, 100)
    img2 = blank()
    cv2.circle(img2, (104, 102), 10, green_bgr(), -1)  # la bague a peu bougé
    cv2.circle(img2, (280, 200), 25, green_bgr(), -1)  # intrus bien plus gros, loin
    tip = trk.find(img2)
    assert abs(tip.x - 104) < 2 and abs(tip.y - 102) < 2


def test_lock_releases_after_misses():
    trk = RingTracker(lock_misses=3)
    img = blank()
    cv2.circle(img, (100, 100), 10, green_bgr(), -1)
    trk.find(img)
    for _ in range(4):  # bague perdue > lock_misses trames
        assert trk.find(blank()) is None
    img2 = blank()
    cv2.circle(img2, (280, 200), 25, green_bgr(), -1)  # seul blob, loin de l'ancien verrou
    tip = trk.find(img2)
    assert abs(tip.x - 280) < 2  # verrou relâché : le plus gros blob reprend


def test_lock_is_sticky_then_releases_to_largest():
    """Verrou posé : un blob hors rayon = bague perdue (pas de téléportation) ;
    après lock_misses trames, le verrou se relâche et le plus gros blob reprend."""
    trk = RingTracker(lock_radius=50.0, lock_misses=2)
    img = blank()
    cv2.circle(img, (100, 100), 10, green_bgr(), -1)
    trk.find(img)
    img2 = blank()  # tous les blobs hors du rayon de verrouillage
    cv2.circle(img2, (250, 180), 14, green_bgr(), -1)
    cv2.circle(img2, (200, 60), 8, green_bgr(), -1)
    for _ in range(3):  # verrou tenu : perte, pas de saut sur l'intrus
        assert trk.find(img2) is None
    tip = trk.find(img2)  # verrou relâché : le plus gros blob
    assert abs(tip.x - 250) < 2


# -- axe longitudinal (bague + gaine) ----------------------------------------------------


def ring_and_sheath(ring_xy=(200, 150), sheath_xy=(160, 150)):
    img = blank()
    cv2.circle(img, ring_xy, 10, green_bgr(), -1)  # bague verte
    cv2.circle(img, sheath_xy, 12, sheath_bgr(), -1)  # gaine bleu sarcelle
    return img


def test_axis_from_ring_and_sheath():
    """Direction gaine → bague = axe longitudinal, orienté vers la pointe."""
    tip = RingTracker().find(ring_and_sheath())
    assert tip is not None and tip.ax is not None
    assert tip.ax == pytest.approx(1.0, abs=0.05)  # gaine à gauche : axe vers +x
    assert tip.ay == pytest.approx(0.0, abs=0.05)


def test_axis_diagonal():
    tip = RingTracker().find(ring_and_sheath(ring_xy=(200, 150), sheath_xy=(170, 120)))
    assert tip.ax == pytest.approx(np.sqrt(0.5), abs=0.05)
    assert tip.ay == pytest.approx(np.sqrt(0.5), abs=0.05)


def test_axis_none_without_sheath():
    img = blank()
    cv2.circle(img, (200, 150), 10, green_bgr(), -1)
    tip = RingTracker().find(img)
    assert tip is not None and tip.ax is None and tip.ay is None


def test_axis_disabled():
    tip = RingTracker(sheath_h_width=0).find(ring_and_sheath())
    assert tip is not None and tip.ax is None


def test_axis_too_close_rejected():
    """Bague et gaine quasi confondues : direction trop bruitée, pas d'axe."""
    tip = RingTracker(axis_min_dist=50.0).find(ring_and_sheath())
    assert tip is not None and tip.ax is None


def test_axis_uses_nearest_sheath_not_largest():
    """Un gros objet de la teinte de la gaine loin de la bague ne fausse pas l'axe."""
    img = ring_and_sheath()  # bague (200,150), gaine (160,150)
    cv2.circle(img, (60, 220), 30, sheath_bgr(), -1)  # intrus énorme, loin
    tip = RingTracker().find(img)
    assert tip.ax == pytest.approx(1.0, abs=0.05)  # axe toujours donné par la vraie gaine
    assert tip.ay == pytest.approx(0.0, abs=0.05)


def test_axis_rejected_when_sheath_too_far():
    """Pas de gaine adjacente : pas d'axe (plutôt que d'inventer une direction)."""
    img = blank()
    cv2.circle(img, (200, 150), 10, green_bgr(), -1)
    cv2.circle(img, (60, 220), 30, sheath_bgr(), -1)  # seule « gaine » : à ~157 px
    tip = RingTracker().find(img)
    assert tip is not None and tip.ax is None


def test_seed_starts_tracking_on_clicked_blob():
    """Le clic « Mesurer la bague » amorce le verrou : l'intrus plus gros est ignoré."""
    img = blank()
    cv2.circle(img, (100, 100), 8, green_bgr(), -1)  # bague
    cv2.circle(img, (280, 200), 25, green_bgr(), -1)  # intrus plus gros dès le départ
    trk = RingTracker()
    tip = trk.find(img)
    assert abs(tip.x - 280) < 2  # sans amorce : le plus gros gagne (et c'est l'intrus)
    trk.seed(100, 100)
    tip = trk.find(img)
    assert abs(tip.x - 100) < 2 and abs(tip.y - 100) < 2
