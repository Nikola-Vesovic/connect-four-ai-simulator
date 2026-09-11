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
