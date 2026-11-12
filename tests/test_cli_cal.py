import json

import numpy as np
import pytest
import skrf

from vna_app import cli

F = np.linspace(400e6, 3e9, 21)
E00 = 0.1 * np.exp(1j * F / 1e9)
E11 = 0.15 * np.exp(-1j * F / 2e9)
E0110 = 0.85 * np.exp(-1j * F / 5e8)
STATE = {"gamma": 0.0, "thru": 1.0}


def measure(g):
    return E00 + E0110 * g / (1 - E11 * g)


class FakeVNA:
    step_hz = int(F[1] - F[0])

    def __init__(self, *a):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def device_info(self):
        return {"firmware": "2.2", "hardware_revision": 2}

    def configure_sweep(self, *a):
        pass

    def sweep(self):
        s11 = measure(np.full(len(F), STATE["gamma"], complex))
        s21 = np.full(len(F), 0.02 * STATE["thru"], complex)
        return F, np.stack([s11, s21], axis=1)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(cli, "LiteVNA", FakeVNA)


def sweep(tmp_path, name, *extra):
    out = tmp_path / name
    rc = cli.main(
        ["sweep", "--start", "400e6", "--stop", "3e9", "--points", "21", "--out", str(out), *extra]
    )
    return rc, out


def test_full_chain(fake, tmp_path):
    for name, g in (("open", 1.0), ("short", -1.0), ("load", 0.0)):
        STATE["gamma"] = g
        assert sweep(tmp_path, f"{name}.s2p")[0] == 0
    assert (
        cli.main(
            [
                "calibrate",
                "--open",
                str(tmp_path / "open.s2p"),
                "--short",
                str(tmp_path / "short.s2p"),
                "--load",
                str(tmp_path / "load.s2p"),
                "--thru",
                str(tmp_path / "load.s2p"),
                "--out",
                str(tmp_path / "x.cal"),
            ]
        )
        == 0
    )
    STATE["gamma"] = 0.3
    rc, out = sweep(tmp_path, "dut.s2p", "--cal", str(tmp_path / "x.cal"))
    assert rc == 0
    net = skrf.Network(str(out))
    np.testing.assert_allclose(net.s[:, 0, 0], 0.3, atol=1e-6)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["calibrated"] is True
    assert meta["calibration_file"].endswith("x.cal")


def test_raw_meta_not_calibrated(fake, tmp_path):
    STATE["gamma"] = 0.0
    _, out = sweep(tmp_path, "raw.s2p")
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["calibrated"] is False and meta["calibration_file"] is None


def test_cal_grid_mismatch_exit_1(fake, tmp_path, monkeypatch, capsys):
    for name, g in (("open", 1.0), ("short", -1.0), ("load", 0.0)):
        STATE["gamma"] = g
        sweep(tmp_path, f"{name}.s2p")
    p = tmp_path
    cli.main(
        [
            "calibrate",
            "--open",
            str(p / "open.s2p"),
            "--short",
            str(p / "short.s2p"),
            "--load",
            str(p / "load.s2p"),
            "--out",
            str(p / "x.cal"),
        ]
    )
    monkeypatch.setattr(FakeVNA, "sweep", lambda self: (F[:-1] + 5e6, np.ones((20, 2), complex)))
    rc, out = sweep(tmp_path, "dut.s2p", "--cal", str(p / "x.cal"))
    assert rc == 1
    assert "grille" in capsys.readouterr().err
    assert not out.exists()
