"""Fusion temporelle position caméra ↔ mesure ε* et carte exportable."""

import numpy as np

from vna_app.vision.fusion import PositionBuffer, ScanMap, summarize_eps


def test_position_interpolation():
    b = PositionBuffer()
    b.push(10.0, 0.0, 0.0)
    b.push(11.0, 10.0, 20.0)
    x, y = b.at(10.5)
    assert abs(x - 5.0) < 1e-9 and abs(y - 10.0) < 1e-9


def test_position_nearest_within_max_age():
    b = PositionBuffer(max_age=0.5)
    b.push(10.0, 3.0, 4.0)
    assert b.at(10.3) == (3.0, 4.0)
    assert b.at(9.8) == (3.0, 4.0)


def test_position_stale_returns_none():
    b = PositionBuffer(max_age=0.5)
    b.push(10.0, 3.0, 4.0)
    assert b.at(11.0) is None
    assert PositionBuffer().at(10.0) is None


def test_position_gap_not_interpolated():
    """Trou de suivi (> 2·max_age entre voisins) : pas de trajectoire inventée."""
    b = PositionBuffer(max_age=0.5)
    b.push(10.0, 0.0, 0.0)
    b.push(20.0, 100.0, 100.0)
    assert b.at(15.0) is None  # loin des deux
    assert b.at(10.2) == (0.0, 0.0)  # proche du premier


def test_summarize_eps_band_median():
    f = np.linspace(400e6, 3e9, 27)
    eps = np.full(27, 5.0 - 1.0j)
    e1, e2 = summarize_eps(f, eps, 500e6, 2500e6)
    assert e1 == 5.0 and e2 == 1.0


def test_summarize_eps_ignores_nonfinite_and_empty_band():
    f = np.linspace(400e6, 3e9, 11)
    eps = np.full(11, 7.0 - 2.0j)
    eps[3] = np.nan + 1j * np.nan
    e1, e2 = summarize_eps(f, eps, 400e6, 3e9)
    assert e1 == 7.0 and e2 == 2.0
    e1, _ = summarize_eps(f, eps, 10e9, 11e9)
    assert np.isnan(e1)


def test_scanmap_csv_roundtrip(tmp_path):
    m = ScanMap()
    m.add(1.0, 100.0, 50.0, 40.0, 12.0)
    m.add(2.0, 101.0, 51.0, 41.0, 13.0)
    assert len(m) == 2
    p = tmp_path / "carto.csv"
    m.save_csv(p)
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    assert data.shape == (2, 5)
    np.testing.assert_allclose(data[1], [2.0, 101.0, 51.0, 41.0, 13.0])
    m.clear()
    assert len(m) == 0
