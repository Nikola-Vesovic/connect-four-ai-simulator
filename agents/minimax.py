"""Minimax with a depth limit: the first real search in this project.

The shape is the pseudocode from the sequential-games deck (slides 6-8), with MAX and MIN
written out as two separate functions rather than folded into one. Folding them is Phase
6's job -- negamax -- and doing it later makes that refactor a measurable step instead of
a starting assumption.

MAX is always the player to move at the root. The recursion alternates, so there is one
invariant worth stating because it removes a parameter from every signature:

    a MAX node is entered right after MIN moved, so any winner found there is MIN.

That is why `_max_value` scores a finished game as a loss without asking who won, and
`_min_value` as a win. The root is exempt from the question entirely: `Agent.choose_move`
refuses to be called on a position that is already over.

What this agent is for is as much the measurement as the play. It has no pruning, so it
visits the entire tree to its depth, and that count is the baseline every later phase is
measured against -- alpha-beta, ordering and transposition tables are all judged by how
much of this tree they can avoid while returning exactly this value. Two consequences
follow from having no pruning, and both are tested rather than assumed: the node count
does not depend on the evaluation function, and it does not depend on move ordering.
"""

from __future__ import annotations

from agents.base import Agent
from core.rules import Board
from core.types import Move, Score
from search.evaluation import DEFAULT, Evaluation
from search.ordering import CENTER_FIRST
from search.stats import Budget, SearchStats

# One step outside the score scale, so the first real value always replaces it.
_WORSE_THAN_ANY = Score.LOSS - 1
_BETTER_THAN_ANY = Score.WIN + 1


class MinimaxAgent(Agent):
    def __init__(
        self,
        depth: int = 4,
        *,
        evaluation: Evaluation = DEFAULT,
        seed: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"Minimax needs at least one ply to pick a move, got {depth}.")
        super().__init__(seed)
        self.depth = depth
        self.evaluation = evaluation
        suffix = "" if evaluation is DEFAULT else f"-{evaluation.name}"
        self.name = f"minimax-d{depth}{suffix}"

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        # The budget is a ceiling, not an instruction: it can only make the search
        # shallower. It cannot make it zero, because a move still has to come back.
        depth = self.depth if budget.depth is None else max(1, min(self.depth, budget.depth))
        stats.depth = depth

        best_value = _WORSE_THAN_ANY
        best_move = None
        for col in CENTER_FIRST:
            if not board.can_play(col):
                continue
            board.play(col)
            stats.nodes += 1
            value = self._min_value(board, depth - 1, 1, stats)
            board.undo()
            if value > best_value:
                best_value, best_move = value, col

        stats.value = best_value
        return best_move

    def _max_value(self, board: Board, depth: int, ply: int, stats: SearchStats) -> int:
        """Best the root player can guarantee from here, with `depth` plies left."""
        if board.winner() is not None:  # MIN moved last, so MIN is the winner
            stats.leaves += 1
            return Score.loss_in(ply)
        if board.is_full():
            stats.leaves += 1
            return Score.DRAW
        if depth <= 0:
            stats.leaves += 1
            return self.evaluation(board, board.player_to_move)

        best = _WORSE_THAN_ANY
        for col in CENTER_FIRST:
            if not board.can_play(col):
                continue
            board.play(col)
            stats.nodes += 1
            value = self._min_value(board, depth - 1, ply + 1, stats)
            board.undo()
            if value > best:
                best = value
        return best

    def _min_value(self, board: Board, depth: int, ply: int, stats: SearchStats) -> int:
        """Worst the opponent can hold the root player to, with `depth` plies left."""
        if board.winner() is not None:  # MAX moved last, so MAX is the winner
            stats.leaves += 1
            return Score.win_in(ply)
        if board.is_full():
            stats.leaves += 1
            return Score.DRAW
        if depth <= 0:
            stats.leaves += 1
            return self.evaluation(board, board.player_to_move.opponent)

        best = _BETTER_THAN_ANY
        for col in CENTER_FIRST:
            if not board.can_play(col):
                continue
            board.play(col)
            stats.nodes += 1
            value = self._max_value(board, depth - 1, ply + 1, stats)
            board.undo()
            if value < best:
                best = value
        return best
