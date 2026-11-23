import numpy as np

from vna_app.processing.sweepfilter import SweepFilter


def test_filter_criteria():
    f = SweepFilter(enabled=True, d_min=0.1, d_max=1.0, radius=0.3)
    g = np.linspace(0.2, 0.6, 21).astype(complex)
    assert f.check(g)[0]
    assert not f.check(np.full(21, 0.3 + 0j))[0]  # extrémités trop proches
    assert not f.check(np.linspace(0, 1.5, 21).astype(complex))[0]  # trop éloignées
    h = np.linspace(0.2, 0.6, 21).astype(complex)
    h[10] = 0.4 + 0.8j
    ok, why = f.check(h)
    assert not ok and "point 10" in why
    assert not f.check(np.r_[g[:-1], np.nan])[0]
    assert (f.accepted, f.rejected) == (1, 4)


def test_disabled_accepts_all():
    f = SweepFilter()
    assert f.check(np.zeros(5, dtype=complex)) == (True, None)
    assert f.rejected == 0
