"""Normale à la surface par ajustement de plan sur des cartes Z16 synthétiques.

Le flux V4L2 réel n'est pas testable sans la caméra ; ici on valide la géométrie
(surface_normal, rgb_to_depth_px) sur des plans générés avec les intrinsèques du module.
"""

import math

import numpy as np

from vna_app.vision.depth import depth_intrinsics, rgb_to_depth_px, surface_normal

W, H = 848, 480


def synth_depth(normal, d=-5000.0, w=W, h=H):
    """Carte Z16 d'un plan n·p = d (points p en unités capteur, caméra en 0)."""
    fx, fy, cx, cy = depth_intrinsics(w, h)
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    denom = normal[0] * (u - cx) / fx + normal[1] * (v - cy) / fy + normal[2]
    z = d / denom
    return np.round(z).astype(np.uint16)


def test_normal_flat_plane():
    z = synth_depth((0.0, 0.0, -1.0))
    n = surface_normal(z, W / 2, H / 2)
    np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=2e-3)


def test_normal_tilted_plane():
    th = math.radians(20)
    true = np.array([math.sin(th), 0.0, -math.cos(th)])
    n = surface_normal(synth_depth(true), W / 2, H / 2)
    assert float(np.dot(n, true)) > 0.999  # < ~2,5° d'écart
    assert n[2] < 0  # orientée vers la caméra


def test_normal_off_center():
    th = math.radians(15)
    true = np.array([0.0, math.sin(th), -math.cos(th)])
    n = surface_normal(synth_depth(true), 600.0, 350.0)
    assert float(np.dot(n, true)) > 0.999


def test_normal_robust_to_cannula():
    """Un objet plus proche (la canule) dans la fenêtre est rejeté comme aberrant."""
    z = synth_depth((0.0, 0.0, -1.0))
    z[228:252, 412:436] = 3500  # bloc 24×24 bien au-dessus de la surface (5000)
    n = surface_normal(z, W / 2, H / 2)
    tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
    assert tilt < 3.0


def test_normal_needs_valid_depth():
    assert surface_normal(np.zeros((H, W), np.uint16), W / 2, H / 2) is None
    z = synth_depth((0.0, 0.0, -1.0))
    z[:, :] = 0
    z[0:3, 0:3] = 1000  # trop peu de pixels valides dans la fenêtre
    assert surface_normal(z, 1.0, 1.0) is None


def test_rgb_to_depth_mapping():
    # centre invariant
    assert rgb_to_depth_px(W / 2, H / 2) == (W / 2, H / 2)
    # FOV RGB > FOV profondeur : le bord RGB tombe HORS de l'image profondeur
    u, v = rgb_to_depth_px(W, H)
    assert u > W and v > H
