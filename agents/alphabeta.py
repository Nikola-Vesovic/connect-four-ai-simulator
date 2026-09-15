"""Negamax, with alpha-beta pruning that can be turned off.

Two changes to Phase 5, and the order matters: the first is a pure reshaping that must not
move a single number, and the second is the one that does the work.

**Negamax** (deck slides 81-84). Phase 5 keeps every value on one axis, the root player's,
and needs two functions to do it: MAX takes the largest child, MIN the smallest. Negamax
keeps the value from the point of view of *whoever is to move at that node* instead, and
then there is only one rule:

    value(node) = max over children of -value(child)

because the game is zero-sum, so a position worth +v to the opponent is worth -v to me.
The identity underneath is max(a, b) = -min(-a, -b): negation reverses order, so the
smallest of the negatives is the negative of the largest. MIN has not disappeared, it has
been absorbed into the minus sign that the parent applies when it reads a child's answer.

What that does to Phase 5's two functions is worth seeing, because it explains why the
refactor is small. `_max_value` scored a finished game as `loss_in(ply)` without asking
who won, on the invariant that a MAX node is entered right after MIN moved. In negamax
that stops being a fact about MAX nodes and becomes the rule for every node: you are
always looking at a position the opponent just moved into, so a winner here is always the
opponent. The `_max_value` branches survive word for word and the `_min_value` ones are
deleted.

Negamax is legal only if the value scale is exactly antisymmetric -- the same position
must be worth +v to one player and -v to the other. Both halves of the scale satisfy it
and both are tested rather than assumed: the evaluation by construction, and the mate
distance because `loss_in(p) = LOSS + p = -(WIN - p) = -win_in(p)` exactly. Mate distance
therefore survives the sign flips untouched, which is the one thing this refactor could
plausibly have broken.

**Alpha-beta** (slides 51-56, 114-115). Two numbers travel down the tree: `alpha` is the
best the side to move has already secured somewhere else, `beta` is the best the opponent
has already secured. A node that finds a move worth `beta` or more is a node the opponent
will never allow, so the rest of its moves are not examined. The root value is unchanged;
only the number of positions visited is.

`pruning=False` is not a second code path. Plain negamax *is* alpha-beta with a window no
value can escape: the window starts one step outside the score scale in both directions,
and if `alpha` is never raised it stays there at every node, so `alpha >= beta` can never
fire. One flag on one assignment, and the two modes cannot drift apart. Having the
unpruned mode at all is BUILD_PLAN design principle 2 -- an optimisation whose
contribution cannot be measured does not belong in this project -- and here it has a
second job: it must return exactly the node count Phase 5's minimax returns, which is what
proves the negamax step changed the shape of the code and nothing else.

Fail-soft, not fail-hard: a node returns the best value it actually found, even when that
is outside the window it was given, rather than clamping to the window. With a full window
at the root the two conventions cannot differ in the value they report, and with no
transposition table they do not differ in the nodes they visit either (measured in Phase
6); the reason to prefer fail-soft is Phase 8, where the tighter bound is what gets stored.

**From this phase on, a node count is meaningless without saying which evaluation and
which move ordering produced it.** Minimax has no cutoffs, so its tree cannot depend on
either; alpha-beta's tree depends on both, because what it can prune depends on what it
has already found and on the order it found it in.

**Phase 7 turns the ordering into a parameter.** It was `search.ordering.CENTER_FIRST`,
a module constant, for the whole of Phase 6 so that phase had one thing to vary; it is now
a `search.ordering.Ordering` value on the agent, exactly as the evaluation already was.
The search asks it for the columns at every node and tells it about every cutoff. Nothing
else here changes -- the ordering cannot see a score, so it cannot affect one.
"""

from __future__ import annotations

from agents.base import Agent
from core.rules import Board
from core.types import Move, Score
from search.evaluation import DEFAULT, Evaluation
from search.ordering import DYNAMIC, MoveOrder, Ordering
from search.stats import Budget, SearchStats

# One step outside the score scale in both directions. As a search window this is one no
# value can escape, which is exactly what "pruning switched off" has to mean.
_BEYOND_LOSS = Score.LOSS - 1
_BEYOND_WIN = Score.WIN + 1


class NegamaxAgent(Agent):
    def __init__(
        self,
        depth: int = 4,
        *,
        pruning: bool = True,
        evaluation: Evaluation = DEFAULT,
        ordering: Ordering = DYNAMIC,
        seed: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"Negamax needs at least one ply to pick a move, got {depth}.")
        super().__init__(seed)
        self.depth = depth
        self.pruning = pruning
        self.evaluation = evaluation
        self.ordering = ordering
        # Only a random base order consumes it, and it is kept rather than drawn from
        # `self._rng` so that the same position always produces the same tree.
        self._ordering_seed = seed
        suffix = "" if evaluation is DEFAULT else f"-{evaluation.name}"
        if ordering != DYNAMIC:
            suffix += f"-{ordering.name}"
        self.name = f"{'alphabeta' if pruning else 'negamax'}-d{depth}{suffix}"

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        # The budget is a ceiling, not an instruction: it can only make the search
        # shallower. It cannot make it zero, because a move still has to come back.
        depth = self.depth if budget.depth is None else max(1, min(self.depth, budget.depth))
        stats.depth = depth
        order = self.ordering.start(depth, self._ordering_seed)

        alpha, beta = _BEYOND_LOSS, _BEYOND_WIN
        best_value = _BEYOND_LOSS
        best_move = None
        # The killer and history tables are still empty here, so this is the base order
        # whatever the flags say -- which is why ordering never changes the move played.
        for col in order.columns(board, 0):
            board.play(col)
            stats.nodes += 1
            value = -self._negamax(board, order, depth - 1, 1, -beta, -alpha, stats)
            board.undo()
            if value > best_value:
                best_value, best_move = value, col
                if self.pruning:
                    alpha = best_value

        stats.value = best_value
        return best_move

    def _negamax(
        self,
        board: Board,
        order: MoveOrder,
        depth: int,
        ply: int,
        alpha: int,
        beta: int,
        stats: SearchStats,
    ) -> int:
        """Value of this position to the player to move, with `depth` plies left.

        The window is `(alpha, beta)` from that same player's point of view, which is why
        the recursive call passes `(-beta, -alpha)` and negates what comes back: the child
        is the opponent's node, and every bound has to be read from his side of the axis.
        """
        if board.winner() is not None:  # the opponent just moved into this, so he won
            stats.leaves += 1
            return Score.loss_in(ply)
        if board.is_full():
            stats.leaves += 1
            return Score.DRAW
        if depth <= 0:
            stats.leaves += 1
            return self.evaluation(board, board.player_to_move)

        best = _BEYOND_LOSS
        columns = order.columns(board, ply)
        promoted = order.promoted  # bound now: recursing will overwrite it
        for index, col in enumerate(columns):
            board.play(col)
            stats.nodes += 1
            value = -self._negamax(board, order, depth - 1, ply + 1, -beta, -alpha, stats)
            board.undo()
            if value > best:
                best = value
                if self.pruning and best > alpha:
                    alpha = best
                    if alpha >= beta:
                        # One per node that stopped early, not one per move skipped.
                        stats.cutoffs += 1
                        stats.cut_index_total += index + 1
                        if index == 0:
                            stats.first_cutoffs += 1
                        if index < promoted:
                            stats.killer_cutoffs += 1
                        order.cutoff(board, col, ply, depth)
                        break
        return best
