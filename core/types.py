from __future__ import annotations

from enum import IntEnum

# A move is the index of the column the disc is dropped into, 0 = leftmost.
Move = int

EMPTY = 0


class Player(IntEnum):
    ONE = 1
    TWO = 2

    @property
    def opponent(self) -> Player:
        return Player.TWO if self is Player.ONE else Player.ONE


class GameResult(IntEnum):
    IN_PROGRESS = 0
    ONE_WINS = 1
    TWO_WINS = 2
    DRAW = 3

    @staticmethod
    def win_for(player: Player) -> GameResult:
        return GameResult.ONE_WINS if player is Player.ONE else GameResult.TWO_WINS


class Score:
    """The single value scale every search in this project reports on.

    Scores are always from the point of view of the player to move at the root, which is
    what lets one number express "I am winning" and "I am losing" on the same axis.

    A win is not one value but a band. `win_in(plies)` subtracts the distance to the
    win, so of two wins the search prefers the shorter one, and of two losses the longer
    one -- a search without this happily shuffles pieces around while holding a mate it
    never plays, because every winning line scores the same. The same subtraction handles
    both sides: MAX maximises, so it takes 9999 over 9997 and -9995 over -9999, and MIN
    minimises, so it delivers mate as fast as MAX delays it.

    The band is bounded by the length of the game, and that bound is the point. A game
    lasts at most 42 plies, so no terminal score can be closer to zero than MIN_WIN.
    Every heuristic in this project must stay strictly inside that, or a search could
    prefer a promising-looking position over an actual win -- the warning on slide 44 of
    the sequential-games deck. `search.evaluation` checks its own bound against MIN_WIN
    at import time.
    """

    WIN = 10_000
    LOSS = -WIN
    DRAW = 0

    # ROWS * COLS, restated here because core.rules imports this module and not the
    # other way round. A test asserts the two agree.
    MAX_MATE_DISTANCE = 42

    MIN_WIN = WIN - MAX_MATE_DISTANCE  # the least valuable win still possible
    MAX_LOSS = LOSS + MAX_MATE_DISTANCE  # the least damaging loss still possible

    @staticmethod
    def win_in(plies: int) -> int:
        """Value of a win `plies` moves from the root, seen from the root player."""
        return Score.WIN - plies

    @staticmethod
    def loss_in(plies: int) -> int:
        """Value of a loss `plies` moves from the root, seen from the root player."""
        return Score.LOSS + plies
