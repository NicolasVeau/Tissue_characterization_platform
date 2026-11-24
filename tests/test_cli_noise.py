"""Banc de bruit `cli noise` sur simulateur (bruit par entrée FIFO -> σ en 1/√avg)."""

import numpy as np

from vna_app import cli
from vna_app.processing.permittivity import eps_three_reference, water_25c

# même grille que configure_sweep(400e6, 3e9, 21) : pas entier de 130 MHz
FREQS = 400e6 + 130e6 * np.arange(21)


def run(tmp_path, *extra, port="sim:rate=0,noise=1e-3,seed=2"):
    out = tmp_path / "noise.npz"
    rc = cli.main(
        ["--port", port, "noise", "--start", "400e6", "--stop", "3e9", "--points", "21",
         "--avg-list", "1,16", "--sweeps", "8", "--out", str(out), *extra]
    )
    return rc, out


def test_noise_scaling_sqrt_n(tmp_path, capsys):
    rc, out = run(tmp_path)
    assert rc == 0
    with np.load(out) as z:
        assert z["freqs"].shape == (21,)
        assert np.abs(z["freqs"] - FREQS).max() <= 1.0
        s1, s16 = z["sigma_s11"]
        assert z["sigma_s21"].shape == (2, 21)
    # bruit indépendant par entrée : moyenner 16 valeurs divise σ par ~4
    ratio = float(np.median(s16) / np.median(s1))
    assert 0.15 < ratio < 0.40
    assert "échelle du bruit" in capsys.readouterr().out


def mobius(eps):
    """Γ(ε) homographique arbitraire : le birapport doit la retrouver exactement."""
    return (eps - 1j) / (eps + 3)


def write_refs(path, freqs=FREQS):
    eps_ref = water_25c(freqs)
    data = {"names": np.array(["air", "probe_short", "ref"]), "ref_liquid": np.array("water")}
    for name, g in (("air", mobius(1.0 + 0j)), ("probe_short", np.ones_like(freqs) + 0j),
                    ("ref", mobius(eps_ref))):
        data[f"f_{name}"] = freqs
        data[f"s_{name}"] = np.stack([np.broadcast_to(g, freqs.shape), np.zeros_like(freqs)],
                                     axis=1).astype(complex)
    np.savez(path, **data)


def test_noise_refs_propagates_to_eps(tmp_path):
    refs = tmp_path / "refs.npz"
    write_refs(refs)
    rc, out = run(tmp_path, "--refs", str(refs))
    assert rc == 0
    with np.load(out) as z:
        gain, sig, gx = z["eps_gain"][0], z["sigma_s11"][0], z["mean_s11"][0]
        assert np.allclose(z["deps_s11"][0], gain * sig)
    # le facteur analytique |dε/dΓx| doit coller à la dérivée numérique du birapport
    ga = np.full(FREQS.shape, mobius(1.0 + 0j))
    gcc = np.ones(FREQS.shape, complex)
    eps_ref = water_25c(FREQS)
    gr = mobius(eps_ref)
    h = 1e-7
    num = np.abs(eps_three_reference(gx + h, ga, gcc, gr, eps_ref)
                 - eps_three_reference(gx, ga, gcc, gr, eps_ref)) / h
    assert np.allclose(gain, num, rtol=1e-2)
    assert np.all(np.isfinite(gain)) and np.all(gain > 0)


def test_noise_refs_grid_mismatch_exit_1(tmp_path, capsys):
    refs = tmp_path / "refs.npz"
    write_refs(refs, freqs=np.linspace(1e9, 2e9, 21))
    rc, _ = run(tmp_path, "--refs", str(refs))
    assert rc == 1
    assert "grille différente" in capsys.readouterr().err


def test_noise_bad_avg_list_exit_2(tmp_path, capsys):
    rc = cli.main(["--port", "sim:rate=0", "noise", "--avg-list", "abc", "--sweeps", "4"])
    assert rc == 2
    assert "avg-list" in capsys.readouterr().err
