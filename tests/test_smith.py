import numpy as np

from vna_app.gui.smith import r_circle, x_arc


def test_r_circle_geometry():
    g = r_circle(1.0)
    # cercle r=1 : centre 0,5 rayon 0,5
    np.testing.assert_allclose(np.abs(g - 0.5), 0.5, atol=1e-3)


def test_x_arc_geometry_and_edge():
    g = x_arc(1.0)
    # centre (1, 1/x), rayon 1/|x|, extrémité r=0 sur le cercle unité
    np.testing.assert_allclose(np.abs(g - (1 + 1j)), 1.0, atol=1e-3)
    assert abs(abs(g[0]) - 1) < 1e-9
