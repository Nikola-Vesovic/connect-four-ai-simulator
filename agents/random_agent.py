"""The floor: a uniformly random legal column.

Everything else in this thesis is measured against this agent. It does no lookahead at
all, so its node count is zero by definition -- which is the point of a floor: it shows
what a given win rate is worth when nothing was paid for it.
"""

from __future__ import annotations

from agents.base import Agent
from core.rules import Board
from core.types import Move
from search.stats import Budget, SearchStats


class RandomAgent(Agent):
    name = "random"

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        return self._rng.choice(board.legal_moves())
