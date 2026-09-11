"""Reference board implementation: a 6x7 list of lists.

Slow and deliberately unclever. Its job is to be obviously correct, so that the bitboard
can be proved right by agreeing with it (BUILD_PLAN, Phase 2).
"""

from __future__ import annotations

from typing import Iterable

from core.rules import CELLS, COLS, CONNECT, DIRECTIONS, ROWS
from core.types import EMPTY, GameResult, Move, Player


class NaiveBoard:
    __slots__ = ("_grid", "_heights", "_history", "_winner")

    def __init__(self) -> None:
        self._grid: list[list[int]] = [[EMPTY] * COLS for _ in range(ROWS)]
        self._heights: list[int] = [0] * COLS
        self._history: list[Move] = []
        self._winner: Player | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_moves(cls, moves: Iterable[Move]) -> NaiveBoard:
        board = cls()
        for col in moves:
            board.play(col)
        return board

    @classmethod
    def from_move_string(cls, moves: str) -> NaiveBoard:
        """Parse the standard benchmark format: 1-based column digits, e.g. "4455"."""
        return cls.from_moves(int(digit) - 1 for digit in moves)

    def copy(self) -> NaiveBoard:
        clone = NaiveBoard.__new__(NaiveBoard)
        clone._grid = [row[:] for row in self._grid]
        clone._heights = self._heights[:]
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
        return tuple(self._history)

    def cell(self, row: int, col: int) -> int:
        if not (0 <= row < ROWS and 0 <= col < COLS):
            raise IndexError(f"Cell ({row}, {col}) is outside the {ROWS}x{COLS} board.")
        return self._grid[row][col]

    # ------------------------------------------------------------------ moves

    def can_play(self, col: Move) -> bool:
        return 0 <= col < COLS and self._heights[col] < ROWS

    def legal_moves(self) -> list[Move]:
        return [col for col in range(COLS) if self._heights[col] < ROWS]

    def play(self, col: Move) -> None:
        if self.is_terminal():
            raise ValueError("Cannot play: the game is already over.")
        if not self.can_play(col):
            raise ValueError(f"Illegal move: column {col} is not playable.")

        player = self.player_to_move
        row = self._heights[col]
        self._grid[row][col] = player
        self._heights[col] += 1
        self._history.append(col)

        if self._wins_through(row, col, player):
            self._winner = player

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Cannot undo: no moves have been played.")
        col = self._history.pop()
        self._heights[col] -= 1
        self._grid[self._heights[col]][col] = EMPTY
        # play() refuses terminal positions, so the position we moved from had no winner.
        self._winner = None

    # ------------------------------------------------------------------ terminal states

    def winner(self) -> Player | None:
        return self._winner

    def is_full(self) -> bool:
        return len(self._history) == CELLS

    def is_terminal(self) -> bool:
        return self._winner is not None or self.is_full()

    def result(self) -> GameResult:
        if self._winner is not None:
            return GameResult.win_for(self._winner)
        return GameResult.DRAW if self.is_full() else GameResult.IN_PROGRESS

    # ------------------------------------------------------------------ win detection

    def _wins_through(self, row: int, col: int, player: int) -> bool:
        """Incremental check: only lines through the disc just played can be new wins."""
        for d_row, d_col in DIRECTIONS:
            run = 1 + self._run(row, col, d_row, d_col, player)
            run += self._run(row, col, -d_row, -d_col, player)
            if run >= CONNECT:
                return True
        return False

    def _run(self, row: int, col: int, d_row: int, d_col: int, player: int) -> int:
        count = 0
        row += d_row
        col += d_col
        # Bounds are checked before every read: in Python a negative index would silently
        # wrap to the opposite edge of the board instead of raising.
        while 0 <= row < ROWS and 0 <= col < COLS and self._grid[row][col] == player:
            count += 1
            row += d_row
            col += d_col
        return count

    def scan_for_winner(self) -> Player | None:
        """Independent full-board scan of all 69 four-cell windows. Test oracle only."""
        for row in range(ROWS):
            for col in range(COLS):
                player = self._grid[row][col]
                if player == EMPTY:
                    continue
                for d_row, d_col in DIRECTIONS:
                    end_row = row + (CONNECT - 1) * d_row
                    end_col = col + (CONNECT - 1) * d_col
                    if not (0 <= end_row < ROWS and 0 <= end_col < COLS):
                        continue
                    if all(
                        self._grid[row + step * d_row][col + step * d_col] == player
                        for step in range(1, CONNECT)
                    ):
                        return Player(player)
        return None

    # ------------------------------------------------------------------ display

    def __str__(self) -> str:
        symbols = {EMPTY: ".", Player.ONE: "X", Player.TWO: "O"}
        rows = [
            " ".join(symbols[self._grid[row][col]] for col in range(COLS))
            for row in reversed(range(ROWS))
        ]
        rows.append(" ".join(str(col + 1) for col in range(COLS)))
        return "\n".join(rows)

    def __repr__(self) -> str:
        played = "".join(str(col + 1) for col in self._history)
        return f"NaiveBoard.from_move_string({played!r})"
