"""The seam every agent plugs into.

An agent is anything that can look at a position and name a column. The tournament runner,
the benchmark scripts and the GUI only ever see this interface, so a one-line random agent
and a full alpha-beta search with a transposition table are interchangeable to them.

`choose_move` is deliberately *not* the method subclasses override. It is a fixed wrapper
around `_select_move` that does the bookkeeping no agent should have to repeat, or be
trusted with:

  * it refuses to move in a finished position,
  * it hands the agent a **copy** of the board, so a buggy search cannot corrupt the game,
  * it times the call on a monotonic clock, so no agent can forget to or misreport,
  * it checks that the column that came back is actually legal,
  * it checks that the agent left the board as it found it -- an unbalanced play/undo pair
    is the single most common bug in the phases that follow, and this catches it at the
    seam instead of three phases later as a wrong number,
  * it fills in `stats.move`, so the returned move and the recorded move cannot disagree.

Why an ABC here when `core.rules.Board` is a Protocol: the two board representations are
independent implementations that have nothing to inherit, and the contract between them
exists so a differential test is meaningful. Agents are the opposite -- one family, all
ours, with real shared behaviour. They also mirror the departmental pygame framework,
where students subclass an `Agent` base class.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod

from core.rules import Board
from core.types import Move
from search.stats import UNLIMITED, Budget, SearchStats


class Agent(ABC):
    """Base class for every agent. Subclasses implement `_select_move` and nothing else."""

    name = "agent"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def new_game(self, seed: int | None = None) -> None:
        """Start a fresh game, optionally reseeding.

        The experiment runner calls this with a seed derived from the master seed before
        every game, which makes a match reproducible regardless of the order its games are
        played in. Agents that carry state between moves -- MCTS reusing its tree in
        Phase 11 -- override this to clear it.
        """
        if seed is not None:
            self._rng.seed(seed)

    def choose_move(
        self, state: Board, budget: Budget = UNLIMITED
    ) -> tuple[Move, SearchStats]:
        """Pick a legal column for the player to move, and report what it cost."""
        if state.is_terminal():
            raise ValueError(f"{self.name}: the game is over, there is nothing to play.")

        board = state.copy()
        stats = SearchStats()

        start = time.perf_counter()
        move = self._select_move(board, budget, stats)
        stats.elapsed = time.perf_counter() - start

        if board.moves_played != state.moves_played:
            raise RuntimeError(
                f"{self.name} returned with {board.moves_played - state.moves_played} "
                "moves left on the board: some play() has no matching undo()."
            )
        if move not in state.legal_moves():
            raise ValueError(f"{self.name} chose column {move!r}, which is not legal here.")

        stats.move = move
        return move, stats

    @abstractmethod
    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        """Decide on a column. Implemented by subclasses, never called directly.

        `board` is a private copy: play and undo on it freely, but leave it balanced.
        `budget` is a ceiling that may be ignored if the agent finishes under it.
        `stats` is to be filled in as the search runs; `elapsed` and `move` are not the
        agent's job.
        """
