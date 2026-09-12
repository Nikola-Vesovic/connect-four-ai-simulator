"""One rung above random: never miss a win in one, never hand one over.

The policy, in order:

  1. if some column wins immediately, play it;
  2. otherwise prefer a column after which the opponent has no immediate win;
  3. break every tie by playing as close to the centre as possible.

Step 2 is "block an immediate loss" stated as a one-reply lookahead rather than as a
rule, for two reasons. The rule version -- "if the opponent wins in column c, play c" --
has a hole: occupying their winning cell can hand them a new win directly above it. And
the board contract has no way to ask where the opponent would win, since `play` always
moves for whoever is to move; the lookahead gets the answer from `play`/`undo` for free.

The honest consequence is that this is a two-ply minimax with a trivial evaluation, so
Phase 5's depth-2 minimax should roughly tie with it and only pull ahead from depth 3.
That is a result to measure, not a problem to hide.

Deterministic: given a position it always returns the same column, so its strength is
never confounded by its own randomness.
"""

from __future__ import annotations

from agents.base import Agent
from core.rules import Board
from core.types import Move, Score
from search.ordering import CENTER_FIRST
from search.stats import Budget, SearchStats


class GreedyAgent(Agent):
    name = "greedy"

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        stats.depth = 2  # fixed horizon: our move, then the opponent's reply
        me = board.player_to_move
        legal = set(board.legal_moves())
        # Centre order, so the first column that qualifies is also the tie-break winner.
        candidates = [col for col in CENTER_FIRST if col in legal]

        # Winning takes priority over every other column, so it has to be settled for the
        # whole board before any safety check runs. Doing it as a separate first pass also
        # keeps the usual cost at 15 positions instead of 56: the second pass stops at the
        # first column that qualifies, while a fused loop would have to price them all.
        for col in candidates:
            board.play(col)
            stats.nodes += 1
            won = board.winner() is me
            board.undo()
            if won:
                stats.value = Score.win_in(1)
                return col

        for col in candidates:
            board.play(col)
            stats.nodes += 1
            safe = not self._opponent_wins_next(board, stats)
            board.undo()
            if safe:
                return col  # value stays None: surviving one reply is not an assessment

        # Every move hands over an immediate win. Within its two-ply horizon this is the
        # one other thing greedy knows for certain, and before Phase 5 it had no way to
        # say so -- it reported the same blank value here as in a winning position.
        stats.value = Score.loss_in(2)
        return candidates[0]  # go down in the centre

    @staticmethod
    def _opponent_wins_next(board: Board, stats: SearchStats) -> bool:
        """Can whoever is to move now win with their next disc?"""
        if board.is_terminal():  # our move filled the board: there is no reply
            return False
        opponent = board.player_to_move
        for col in board.legal_moves():
            board.play(col)
            stats.nodes += 1
            wins = board.winner() is opponent
            board.undo()
            if wins:
                return True
        return False
