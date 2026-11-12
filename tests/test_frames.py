import random
import struct

import numpy as np
import pytest

from vna_app.driver.litevna import (
    FRAME_FMT,
    FRAME_SIZE,
    LiteVNAError,
    assemble_sweep,
    decode_frame,
    decode_frames,
)


def test_frame_size():
    assert FRAME_SIZE == 32


def test_decode_single():
    raw = struct.pack(FRAME_FMT, 1000, -2000, 300, 400, -500, 600, 42)
    fr = decode_frame(raw)
    assert fr.fwd0 == 1000 - 2000j
    assert fr.rev0 == 300 + 400j
    assert fr.rev1 == -500 + 600j
    assert fr.freq_index == 42


def test_decode_extreme_values_and_u16_index():
    raw = struct.pack(FRAME_FMT, 2**31 - 1, -2**31, 0, 0, 0, 0, 65534)
    fr = decode_frame(raw)
    assert fr.fwd0 == complex(2**31 - 1, -2**31)
    assert fr.freq_index == 65534


def test_decode_bad_length():
    with pytest.raises(ValueError):
        decode_frame(b"\x00" * 31)
    with pytest.raises(ValueError):
        decode_frames(b"\x00" * 33)


def test_s_params_no_averaging():
    raw = struct.pack(FRAME_FMT, 1000, 0, 100, 0, 500, 500, 0)
    s = assemble_sweep(decode_frames(raw), 1, 1)
    assert s.shape == (1, 2)
    assert s[0, 0] == pytest.approx(0.1)
    assert s[0, 1] == pytest.approx(0.5 + 0.5j)


def test_out_of_order_frames():
    n = 10
    raws = [struct.pack(FRAME_FMT, 1000, 0, 10 * k, 0, 0, 100 * k, k) for k in range(n)]
    random.Random(1).shuffle(raws)
    s = assemble_sweep(decode_frames(b"".join(raws)), n, 1)
    np.testing.assert_allclose(s[:, 0], 0.01 * np.arange(n))
    np.testing.assert_allclose(s[:, 1], 0.1j * np.arange(n))


def test_averaging_interleaved_and_shuffled():
    n, avg = 4, 3
    raws = []
    for k in range(n):
        for a in range(avg):
            # rev0Re moyenne = 100*(k+1) ; bruit symétrique (-a+1..)
            raws.append(struct.pack(FRAME_FMT, 1000, 0, 100 * (k + 1) + (a - 1) * 10, 0,
                                    0, 0, k))
    random.Random(7).shuffle(raws)
    s = assemble_sweep(decode_frames(b"".join(raws)), n, avg)
    np.testing.assert_allclose(s[:, 0], 0.1 * (np.arange(n) + 1))


def test_averaging_raw_complex_before_division():
    # fwd varie : <rev>/<fwd> = 200/1500, différent de moyenne(rev/fwd)
    raws = [struct.pack(FRAME_FMT, 1000, 0, 100, 0, 0, 0, 0),
            struct.pack(FRAME_FMT, 2000, 0, 300, 0, 0, 0, 0)]
    s = assemble_sweep(decode_frames(b"".join(raws)), 1, 2)
    assert s[0, 0] == pytest.approx(200 / 1500)


def test_missing_point_raises():
    raws = b"".join(struct.pack(FRAME_FMT, 1, 0, 1, 0, 1, 0, k) for k in (0, 2))
    with pytest.raises(LiteVNAError):
        assemble_sweep(decode_frames(raws), 3, 1)


def test_out_of_range_index_ignored_then_incomplete():
    raws = b"".join(struct.pack(FRAME_FMT, 1, 0, 1, 0, 1, 0, k) for k in (0, 9))
    with pytest.raises(LiteVNAError):
        assemble_sweep(decode_frames(raws), 2, 1)


def test_extra_duplicates_ignored():
    raws = b"".join(struct.pack(FRAME_FMT, 100, 0, v, 0, 0, 0, k)
                    for k, v in ((0, 10), (0, 99999), (1, 20)))
    s = assemble_sweep(decode_frames(raws), 2, 1)
    np.testing.assert_allclose(s[:, 0], [0.1, 0.2])
