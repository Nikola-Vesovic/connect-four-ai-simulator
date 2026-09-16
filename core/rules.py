"""The rules of Connect Four, and the contract every board implementation must satisfy.

This module contains no algorithms. It exists so that the naive reference board and the
bitboard can be tested against each other: a differential test is only meaningful if both
sides agree on what the operations mean.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import EMPTY, GameResult, Move, Player

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

# Bits per column in `Board.key`. One more than ROWS, so that the carry a full column
# produces has somewhere to land -- the same sentinel row that stops the bitboard's
# shifts from running out of one column into the next.
KEY_STRIDE = ROWS + 1


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

    def key(self) -> int:
        """A number that identifies this position, including whose turn it is.

        Deferred from Phase 1 to Phase 8, where it is first needed: a transposition
        table is a dictionary from positions to search results, so it needs a name for a
        position. The name has to distinguish everything a search result depends on --
        the discs of both colours *and* the side to move -- and nothing else.

        The scheme is the bitboard's, and both implementations produce the same number
        (there is a differential test). Per column, `key = own_discs + (2**height - 1)`,
        laid out `KEY_STRIDE` bits apart. **It is collision-free, not a hash**, which is
        worth stating because it is unusual: chess engines use Zobrist hashing and live
        with collisions. The argument is short. A column of height h contributes a value
        in `[2**h - 1, 2**(h+1) - 2]`, and those ranges are disjoint for h = 0..6, so the
        contribution determines both the height and the discs; the widest of them,
        126 for a full column, still fits in the seven bits the column owns, so no carry
        ever crosses into its neighbour. Two positions with the same key are therefore
        the same position.
        """


def render(board: Board) -> str:
    """Text view of any board, top row first, with 1-based column labels underneath."""
    symbols = {EMPTY: ".", Player.ONE: "X", Player.TWO: "O"}
    rows = [
        " ".join(symbols[board.cell(row, col)] for col in range(COLS))
        for row in reversed(range(ROWS))
    ]
    rows.append(" ".join(str(col + 1) for col in range(COLS)))
    return "\n".join(rows)


def winning_line(board: Board) -> tuple[tuple[int, int], ...] | None:
    """The four cells that won the game, as (row, col), or None if nobody has won.

    For display only. A search asks `board.winner()`, which is incremental and far
    cheaper; the cells themselves are not part of the contract, because one
    implementation knows them and the other would have to reconstruct them. Reading the
    board through `cell()` keeps this working on both without either having to change.
    """
    for col in range(COLS):
        for row in range(ROWS):
            occupant = board.cell(row, col)
            if occupant == EMPTY:
                continue
            for d_row, d_col in DIRECTIONS:
                cells = tuple(
                    (row + step * d_row, col + step * d_col) for step in range(CONNECT)
                )
                if all(0 <= r < ROWS and 0 <= c < COLS for r, c in cells) and all(
                    board.cell(r, c) == occupant for r, c in cells
                ):
                    return cells
    return None
