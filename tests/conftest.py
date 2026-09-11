from __future__ import annotations

import pytest

from core.bitboard import Bitboard
from core.board_naive import NaiveBoard


@pytest.fixture(params=[NaiveBoard, Bitboard], ids=["naive", "bitboard"])
def board_class(request):
    """Every contract test runs once per board implementation."""
    return request.param
