"""The rules of Connect Four, and the contract every board implementation must satisfy.

This module contains no algorithms. It exists so that the naive reference board and the
bitboard can be tested against each other: a differential test is only meaningful if both
sides agree on what the operations mean.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import GameResult, Move, Player

ROWS = 6
COLS = 7
CONNECT = 4
CELLS = ROWS * COLS

# Row 0 is the BOTTOM row, column 0 is the leftmost. This matches the bitboard layout
# (bit index = col * 7 + row), so the two implementations can be compared cell by cell
# without any coordinate translation. Only the GUI flips rows, at draw time.

# The four win axes as (delta_row, delta_col): vertical, horizontal, diagonal up-right,
# diagonal up-left. Each is checked in both directions, so the opposites are redundant.
DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


@runtime_checkable
class Board(Protocol):
    """Operations every board representation provides, with identical semantics."""

    @property
    def player_to_move(self) -> Player: ...

    @property
    def moves_played(self) -> int: ...

    def cell(self, row: int, col: int) -> int:
        """EMPTY, Player.ONE or Player.TWO. Raises IndexError outside the board."""

    def can_play(self, col: Move) -> bool: ...

    def legal_moves(self) -> list[Move]: ...

    def play(self, col: Move) -> None:
        """Drop a disc for the player to move. Illegal or terminal-position calls raise."""

    def undo(self) -> None: ...

    def winner(self) -> Player | None: ...

    def is_full(self) -> bool: ...

    def is_terminal(self) -> bool: ...

    def result(self) -> GameResult: ...

    def copy(self) -> Board: ...
