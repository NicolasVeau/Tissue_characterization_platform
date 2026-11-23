"""Rend `gr_litevna` et `vna_app` importables ; ignore les tests si GNU Radio est absent."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("gnuradio", reason="GNU Radio non installé")
