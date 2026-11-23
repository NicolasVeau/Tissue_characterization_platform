"""Tests du driver avec un faux port série émulant l'appareil (sans VNA)."""

import struct

import pytest
import serial

from vna_app.driver import litevna as L


class FakeSerial:
    """Émule le sous-ensemble du protocole utilisé par LiteVNA."""

    def __init__(self, stale=0, drop_after=None, regs=None):
        self.regs = {
            L.REG_DEVICE_VARIANT: 2,
            L.REG_PROTOCOL_VERSION: 1,
            L.REG_HW_REVISION: 2,
            L.REG_FW_MAJOR: 2,
            L.REG_FW_MINOR: 2,
        }
        self.regs.update(regs or {})
        self.stale = stale  # entrées résiduelles servies après le 1er restart
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
            val = int.from_bytes(d[2 : 2 + size], "little")
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
    def _entry(self, k, stale=False):
        rev0 = -777 if stale else 10 * (k + 1)  # un résiduel porte une valeur ANCIENNE (fausse)
        return struct.pack(L.FRAME_FMT, 1000, 0, rev0, 0, k, 0, k)

    def _restart(self):
        self.fifo = [self._entry(k) for k in range(self.points) for _ in range(self.avg or 1)]
        if self.first_restart and self.stale:
            old = [self._entry(k, True) for k in range(self.stale) for _ in range(self.avg or 1)]
            self.fifo = old + self.fifo
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


@pytest.mark.parametrize("stale", [1, 14, 63, 200])
@pytest.mark.parametrize("avg", [1, 3])
def test_stale_entries_discarded(monkeypatch, stale, avg):
    """Résiduels (ancien échantillon) jetés : seul le nouveau balayage est utilisé."""
    vna, fake = make(monkeypatch, stale=stale)
    vna.configure_sweep(400e6, 3e9, 201, avg)
    fake.first_restart = True  # le résiduel arrive au restart de sweep()
    _, s = vna.sweep(retries=0)
    assert s.shape == (201, 2)
    assert (s[:, 0].real > 0).all()  # aucun résiduel (-777) conservé
    for k in (0, 1, stale - 1, 5, 200):
        if 0 <= k < 201:
            assert s[k, 0] == pytest.approx(0.01 * (k + 1))


@pytest.mark.parametrize("n_stale", [1, 2, 3, 4, 5, 7, 11, 63])
def test_stale_partial_point_with_averaging(monkeypatch, n_stale):
    """Résiduel de n ENTRÉES (pas de points entiers) : il peut finir au milieu d'un point."""
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 201, 4)
    orig = fake._restart

    def restart():
        orig()
        fake.fifo = [fake._entry(i // 4, True) for i in range(n_stale)] + fake.fifo

    fake._restart = restart
    _, s = vna.sweep(retries=0)
    assert (s[:, 0].real > 0).all()  # aucun résiduel (-777) conservé
    assert s[0, 0] == pytest.approx(0.01)


def test_endless_bad_indices_hit_cap_then_error(monkeypatch):
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 11, 1)
    orig = fake.write

    def write(data):
        d = bytes(data)
        if d[:1] == bytes([L.OP_READFIFO]):  # appareil qui ne renvoie que l'index 0, sans fin
            fake.rx += b"".join(fake._entry(0) for _ in range(d[2]))
        else:
            orig(data)

    fake.write = write
    with pytest.raises(L.LiteVNAError, match="incomplet"):
        vna.sweep(retries=1)


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


# -- streaming ----------------------------------------------------------------
def _get(q, timeout=5):
    import queue as _q

    try:
        return q.get(timeout=timeout)
    except _q.Empty:  # pragma: no cover
        pytest.fail("rien reçu dans la queue")


def test_streaming_delivers_ordered_results(monkeypatch):
    vna, _ = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 21, 1)
    q = vna.start_streaming(maxsize=16)
    assert vna.is_streaming
    got = [_get(q) for _ in range(5)]
    vna.stop_streaming()
    assert not vna.is_streaming
    assert [r.seq for r in got] == [0, 1, 2, 3, 4]
    assert all(isinstance(r, L.SweepResult) and r.s_raw.shape == (21, 2) for r in got)
    assert got[0].freqs[0] == 400e6 and got[0].timestamp > 0


def test_streaming_slow_consumer_never_blocks_and_keeps_latest(monkeypatch):
    import threading as _th

    vna, _ = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 11, 1)
    q = vna.start_streaming(maxsize=2)
    _th.Event().wait(0.3)  # time.sleep est neutralisé par make() ; le consommateur dort
    vna.stop_streaming()
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert 1 <= len(items) <= 2
    assert items[-1].seq >= 2  # les plus anciens ont été écartés


def test_streaming_disconnect_puts_error_and_stops(monkeypatch):
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 21, 1)
    q = vna.start_streaming(maxsize=64)
    assert isinstance(_get(q), L.SweepResult)
    fake.drop_after = fake.reads  # coupe la liaison
    err = None
    for _ in range(200):
        item = _get(q)
        if isinstance(item, Exception):
            err = item
            break
    assert isinstance(err, L.LiteVNAError)
    vna._thread.join(5)
    assert not vna.is_streaming


def test_sync_calls_refused_while_streaming(monkeypatch):
    vna, _ = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 21, 1)
    vna.start_streaming()
    with pytest.raises(L.LiteVNAError, match="streaming"):
        vna.sweep()
    with pytest.raises(L.LiteVNAError, match="streaming"):
        vna.configure_sweep(1e9, 2e9, 11, 1)
    with pytest.raises(L.LiteVNAError, match="déjà"):
        vna.start_streaming()
    vna.stop_streaming()
    vna.stop_streaming()  # idempotent
    vna.configure_sweep(1e9, 2e9, 11, 1)  # de nouveau autorisé
    assert vna.sweep()[1].shape == (11, 2)


def test_streaming_requires_configure(monkeypatch):
    vna, _ = make(monkeypatch)
    with pytest.raises(L.LiteVNAError, match="configure_sweep"):
        vna.start_streaming()


def test_close_stops_streaming(monkeypatch):
    vna, fake = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 21, 1)
    vna.start_streaming()
    vna.close()
    assert not vna.is_streaming and fake.closed


def test_restart_streaming_after_reconfigure(monkeypatch):
    vna, _ = make(monkeypatch)
    vna.configure_sweep(400e6, 3e9, 21, 1)
    q = vna.start_streaming()
    _get(q)
    vna.stop_streaming()
    vna.configure_sweep(1e9, 2e9, 11, 1)
    q2 = vna.start_streaming()
    r = _get(q2)
    vna.stop_streaming()
    assert r.seq == 0 and r.s_raw.shape == (11, 2)


def test_set_power_roundtrip_on_sim():
    from vna_app.driver.sim import open_device

    with open_device("sim") as v:
        assert v.get_power() == (3, 1)
        v.set_power(adf4350=0)
        assert v.get_power() == (0, 1)
        v.set_power(si5351=2)
        assert v.get_power() == (0, 2)
        with pytest.raises(ValueError):
            v.set_power(adf4350=4)
