"""Fast board representation: two integers and a handful of shifts.

Layout is 7 bits per column, bit index = col * 7 + row, row 0 at the bottom. Rows 0-5
are playable; row 6 is a sentinel that always stays zero. That sentinel is what stops a
shift from running off the top of one column into the bottom of the next, which is the
bitboard form of the index-wraparound bug the naive board guards against with bounds
checks. Here the layout makes the bug impossible instead of merely detecting it.

    col:      0    1    2    3    4    5    6
    row 6:    6   13   20   27   34   41   48   <- sentinel, always 0
    row 5:    5   12   19   26   33   40   47
    ...
    row 0:    0    7   14   21   28   35   42   <- bottom
"""

from __future__ import annotations

from typing import Iterable

from core.rules import CELLS, COLS, ROWS, render
from core.types import EMPTY, GameResult, Move, Player

_HEIGHT = ROWS + 1

# One step in each of the four win axes, as a distance in bit indices.
_UP = 1
_RIGHT = _HEIGHT
_UP_RIGHT = _HEIGHT + 1
_DOWN_RIGHT = _HEIGHT - 1

_BOTTOM_MASK = tuple(1 << (col * _HEIGHT) for col in range(COLS))
_TOP_MASK = tuple(1 << (col * _HEIGHT + ROWS - 1) for col in range(COLS))


class Bitboard:
    __slots__ = ("_position", "_mask", "_history", "_winner")

    def __init__(self) -> None:
        self._position = 0  # discs of the player to move
        self._mask = 0  # every disc on the board
        self._history: list[tuple[Move, int, int]] = []
        self._winner: Player | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_moves(cls, moves: Iterable[Move]) -> Bitboard:
        board = cls()
        for col in moves:
            board.play(col)
        return board

    @classmethod
    def from_move_string(cls, moves: str) -> Bitboard:
        """Parse the standard benchmark format: 1-based column digits, e.g. "4455"."""
        return cls.from_moves(int(digit) - 1 for digit in moves)

    def copy(self) -> Bitboard:
        clone = Bitboard.__new__(Bitboard)
        clone._position = self._position
        clone._mask = self._mask
        clone._history = self._history[:]
        clone._winner = self._winner
        return clone

    # ------------------------------------------------------------------ state

    @property
    def player_to_move(self) -> Player:
        return Player.ONE if len(self._history) % 2 == 0 else Player.TWO

    @property
    def moves_played(self) -> int:
        return len(self._history)

    @property
    def history(self) -> tuple[Move, ...]:
        return tuple(col for col, _, _ in self._history)

    def cell(self, row: int, col: int) -> int:
        if not (0 <= row < ROWS and 0 <= col < COLS):
            raise IndexError(f"Cell ({row}, {col}) is outside the {ROWS}x{COLS} board.")
        bit = 1 << (col * _HEIGHT + row)
        if not self._mask & bit:
            return EMPTY
        mover = self.player_to_move
        return mover if self._position & bit else mover.opponent

    # ------------------------------------------------------------------ moves

    def can_play(self, col: Move) -> bool:
        return 0 <= col < COLS and not self._mask & _TOP_MASK[col]

    def legal_moves(self) -> list[Move]:
        mask = self._mask
        return [col for col in range(COLS) if not mask & _TOP_MASK[col]]

    def play(self, col: Move) -> None:
        if self.is_terminal():
            raise ValueError("Cannot play: the game is already over.")
        if not self.can_play(col):
            raise ValueError(f"Illegal move: column {col} is not playable.")

        mover = self.player_to_move
        self._history.append((col, self._position, self._mask))
        self._position ^= self._mask  # the opponent's discs, who is now to move
        self._mask |= self._mask + _BOTTOM_MASK[col]  # carry lands on the lowest empty cell

        if self._alignment(self._position ^ self._mask):
            self._winner = mover

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Cannot undo: no moves have been played.")
        _, self._position, self._mask = self._history.pop()
        # play() refuses terminal positions, so the position we moved from had no winner.
        self._winner = None

    # ------------------------------------------------------------------ terminal states

    def winner(self) -> Player | None:
        return self._winner

    def is_full(self) -> bool:
        return len(self._history) == CELLS

    def is_terminal(self) -> bool:
        return self._winner is not None or len(self._history) == CELLS

    def result(self) -> GameResult:
        if self._winner is not None:
            return GameResult.win_for(self._winner)
        return GameResult.DRAW if self.is_full() else GameResult.IN_PROGRESS

    # ------------------------------------------------------------------ identity

    def key(self) -> int:
        """Collision-free name for this position -- see `core.rules.Board.key`.

        One addition. `_mask` is a run of ones from the bottom of every column, so adding
        it to `_position` sets the first free cell of each column and leaves the discs of
        the side to move below it, which is the whole state in one integer.
        """
        return self._position + self._mask

    # ------------------------------------------------------------------ win detection

    @staticmethod
    def _alignment(discs: int) -> bool:
        """True if one colour has four in a row. Two operations per axis, no bounds checks."""
        pairs = discs & (discs >> _UP)
        if pairs & (pairs >> 2 * _UP):
            return True
        pairs = discs & (discs >> _RIGHT)
        if pairs & (pairs >> 2 * _RIGHT):
            return True
        pairs = discs & (discs >> _UP_RIGHT)
        if pairs & (pairs >> 2 * _UP_RIGHT):
            return True
        pairs = discs & (discs >> _DOWN_RIGHT)
        return bool(pairs & (pairs >> 2 * _DOWN_RIGHT))

    # ------------------------------------------------------------------ display

    def __str__(self) -> str:
        return render(self)

    def __repr__(self) -> str:
        played = "".join(str(col + 1) for col, _, _ in self._history)
        return f"Bitboard.from_move_string({played!r})"
