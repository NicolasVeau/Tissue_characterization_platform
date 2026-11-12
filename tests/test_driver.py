"""Tests du driver avec un faux port série émulant l'appareil (sans VNA)."""
import struct

import pytest
import serial

from vna_app.driver import litevna as L


class FakeSerial:
    """Émule le sous-ensemble du protocole utilisé par LiteVNA."""

    def __init__(self, stale=0, drop_after=None, regs=None):
        self.regs = {L.REG_DEVICE_VARIANT: 2, L.REG_PROTOCOL_VERSION: 1,
                     L.REG_HW_REVISION: 2, L.REG_FW_MAJOR: 2, L.REG_FW_MINOR: 2}
        self.regs.update(regs or {})
        self.stale = stale          # entrées résiduelles servies après le 1er restart
        self.drop_after = drop_after  # lève SerialException après n lectures
        self.rx = bytearray()
        self.fifo = []
        self.reads = 0
        self.points = self.avg = 0
        self.first_restart = True
        self.closed = False

    # -- API pyserial utilisée par le driver
    def reset_input_buffer(self):
        self.rx.clear()

    def close(self):
        self.closed = True

    def write(self, data):
        d = bytes(data)
        if d[:1] == bytes([L.OP_INDICATE]):
            self.rx += bytes([L.INDICATE_REPLY])
        elif d[:1] == bytes([L.OP_READ]):
            self.rx += bytes([self.regs[d[1]]])
        elif d[:1] == bytes([L.OP_READ2]):
            self.rx += int(self.regs[d[1]]).to_bytes(2, "little")
        elif d[:1] in (bytes([L.OP_WRITE2]), bytes([L.OP_WRITE])):
            size = 2 if d[0] == L.OP_WRITE2 else 1
            val = int.from_bytes(d[2:2 + size], "little")
            self.regs[d[1]] = val
            if d[1] == L.REG_SWEEP_POINTS:
                self.points = val
                self._restart()
            elif d[1] == L.REG_VALUES_PER_FREQ:
                self.avg = val
                self._restart()
            elif d[1] == L.REG_FIFO:
                self.fifo = []
        elif d[:1] == bytes([L.OP_READFIFO]):
            n = d[2]
            out, self.fifo = self.fifo[:n], self.fifo[n:]
            self.rx += b"".join(out)

    def read(self, n):
        self.reads += 1
        if self.drop_after is not None and self.reads > self.drop_after:
            raise serial.SerialException("device disconnected")
        out, self.rx = bytes(self.rx[:n]), self.rx[n:]
        return out

    # -- émulation
    def _entry(self, k):
        return struct.pack(L.FRAME_FMT, 1000, 0, 10 * (k + 1), 0, k, 0, k)

    def _restart(self):
        self.fifo = [self._entry(k) for k in range(self.points) for _ in range(self.avg or 1)]
        if self.first_restart and self.stale:
            self.fifo = [self._entry(k) for k in range(self.stale)] + self.fifo
        if self.points and self.avg:
            self.first_restart = False


def make(monkeypatch, **kw):
    fake = FakeSerial(**kw)
    monkeypatch.setattr(L.serial, "Serial", lambda *a, **k: fake)
    vna = L.LiteVNA("/dev/fake")
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    vna.connect()
    return vna, fake


def test_info(monkeypatch):
    vna, _ = make(monkeypatch)
    assert vna.device_info()["firmware"] == "2.2"


def test_wrong_variant_refused(monkeypatch):
    with pytest.raises(L.LiteVNAError):
        make(monkeypatch, regs={L.REG_DEVICE_VARIANT: 1})


def test_sweep_basic(monkeypatch):
    vna, _ = make(monkeypatch)
    f = vna.configure_sweep(400e6, 3e9, 201, 1)
    assert f[0] == 400e6 and f[-1] == 3e9
    freqs, s = vna.sweep()
    assert s.shape == (201, 2)
    assert s[5, 0] == pytest.approx(0.06)


def test_sweep_averaging(monkeypatch):
    vna, _ = make(monkeypatch)
    vna.configure_sweep(1e9, 2e9, 11, 4)
    _, s = vna.sweep()
    assert s.shape == (11, 2)


def test_stale_entries_trigger_retry(monkeypatch):
    # 63 résiduelles de l'ancien sweep : 1re tentative incomplète, 2e OK
    vna, fake = make(monkeypatch, stale=63)
    vna.configure_sweep(400e6, 3e9, 201, 1)
    fake.first_restart = True  # le résiduel arrive au restart de sweep()
    _, s = vna.sweep()
    assert s.shape == (201, 2)


def test_refused_params(monkeypatch):
    vna, fake = make(monkeypatch)
    orig = fake.write

    def write(data):
        orig(data)
        if bytes(data)[:2] == bytes([L.OP_WRITE2, L.REG_SWEEP_POINTS]):
            fake.regs[L.REG_SWEEP_POINTS] = 100  # l'appareil clippe

    fake.write = write
    with pytest.raises(L.LiteVNAError, match="refusés"):
        vna.configure_sweep(400e6, 3e9, 201, 1)


def test_disconnect_mid_sweep(monkeypatch):
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 201, 1)
    fake.drop_after = fake.reads  # coupe à la prochaine lecture
    with pytest.raises(L.LiteVNAError, match="série"):
        vna.sweep()


def test_timeout_when_no_data(monkeypatch):
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 11, 1)
    monkeypatch.setattr(fake, "_restart", lambda: None)  # rien ne revient
    t = iter(range(0, 1000, 5))
    monkeypatch.setattr(L.time, "monotonic", lambda: next(t))
    with pytest.raises(L.LiteVNATimeout):
        vna.sweep()


def test_not_connected():
    with pytest.raises(L.LiteVNAError):
        L.LiteVNA("/dev/fake").device_info()
