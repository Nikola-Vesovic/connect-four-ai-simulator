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

**Phase 8 adds the transposition table**, the third thing on the same pattern. Alpha-beta
and move ordering both work on one tree; the table notices that the tree contains the same
position many times over and searches it once. Three points about how it is wired in here,
because each of them is a decision rather than a detail:

* **A node is answered from the table only when the entry settles the current window.**
  An entry is not necessarily a value: a node that cut off computed a lower bound, a node
  where nothing reached alpha computed an upper bound. `EXACT` answers anything; `LOWER`
  answers only if it is at or above beta; `UPPER` only if it is at or below alpha. Getting
  that wrong is the classic transposition-table bug, and unlike most bugs in this project
  it would change the answer rather than the speed.
* **Horizon nodes are never probed.** At `depth <= 0` the search returns the static
  evaluation, and it must keep doing so: a stored value from a deeper search is a *better*
  number, and substituting it there would quietly turn a depth-N search into something
  else. Design principle 3 is that an optimisation changes how fast the answer arrives and
  never what it is, and that includes not improving it.
* **The root never probes.** It needs a move rather than a value, and leaving it out keeps
  the root's move order the base order, which is what preserves the property tested since
  Phase 6: the optimisation checkboxes cannot change which column gets played.
"""

from __future__ import annotations

from agents.base import Agent
from core.rules import Board
from core.types import Move, Score
from search.evaluation import DEFAULT, Evaluation
from search.ordering import DYNAMIC, MoveOrder, Ordering
from search.stats import Budget, SearchStats
from search.transposition import (
    EXACT,
    LOWER,
    TABLE,
    UPPER,
    Transposition,
    from_entry,
    to_entry,
)

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
        transposition: Transposition = TABLE,
        seed: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"Negamax needs at least one ply to pick a move, got {depth}.")
        super().__init__(seed)
        self.depth = depth
        self.pruning = pruning
        self.evaluation = evaluation
        self.ordering = ordering
        self.transposition = transposition
        self._table = transposition.start() if transposition.persistent else None
        # Only a random base order consumes it, and it is kept rather than drawn from
        # `self._rng` so that the same position always produces the same tree.
        self._ordering_seed = seed
        suffix = "" if evaluation is DEFAULT else f"-{evaluation.name}"
        if ordering != DYNAMIC:
            suffix += f"-{ordering.name}"
        if transposition != TABLE:
            suffix += f"-{transposition.name}"
        self.name = f"{'alphabeta' if pruning else 'negamax'}-d{depth}{suffix}"

    def new_game(self, seed: int | None = None) -> None:
        """A persistent table is the one piece of state that outlives a search, so it is
        also the one piece that a new game has to throw away."""
        super().new_game(seed)
        if self._table is not None:
            self._table.clear()

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        # The budget is a ceiling, not an instruction: it can only make the search
        # shallower. It cannot make it zero, because a move still has to come back.
        depth = self.depth if budget.depth is None else max(1, min(self.depth, budget.depth))
        stats.depth = depth
        order = self.ordering.start(depth, self._ordering_seed)
        table = self._table if self.transposition.persistent else self.transposition.start()
        if table is not None:
            table.reset_counters()

        alpha, beta = _BEYOND_LOSS, _BEYOND_WIN
        best_value = _BEYOND_LOSS
        best_move = None
        # The killer slots, the history counts and the table are all empty here, and the
        # root deliberately does not probe, so this is the base order whatever the flags
        # say -- which is why none of this can change the move played.
        for col in order.columns(board, 0):
            board.play(col)
            stats.nodes += 1
            value = -self._negamax(board, order, table, depth - 1, 1, -beta, -alpha, stats)
            board.undo()
            if value > best_value:
                best_value, best_move = value, col
                if self.pruning:
                    alpha = best_value

        stats.value = best_value
        stats.take_table(table)
        return best_move

    def _negamax(
        self,
        board: Board,
        order: MoveOrder,
        table,
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

        tt_move = -1
        key = 0
        if table is not None:
            key = board.key()
            found = table.probe(key)
            if found is not None:
                stored_depth, flag, value, tt_move = found
                # `>=`, not `>`: an entry computed with at least as much depth left as we
                # have now is at least as trustworthy as the search we were about to run.
                if self.transposition.reads and stored_depth >= depth:
                    value = from_entry(value, ply)
                    if (
                        flag == EXACT
                        or (flag == LOWER and value >= beta)
                        or (flag == UPPER and value <= alpha)
                    ):
                        stats.tt_returns += 1
                        return value
                if not self.transposition.reads:
                    tt_move = -1  # maintained but not acted on: the control experiment

        original_alpha = alpha
        best = _BEYOND_LOSS
        best_col = -1
        columns = order.columns(board, ply, tt_move)
        promoted = order.promoted  # bound now: recursing will overwrite it
        from_table = order.tt_promoted
        for index, col in enumerate(columns):
            board.play(col)
            stats.nodes += 1
            value = -self._negamax(board, order, table, depth - 1, ply + 1, -beta, -alpha, stats)
            board.undo()
            if value > best:
                best = value
                best_col = col
                if self.pruning and best > alpha:
                    alpha = best
                    if alpha >= beta:
                        # One per node that stopped early, not one per move skipped.
                        stats.cutoffs += 1
                        stats.cut_index_total += index + 1
                        if index == 0:
                            stats.first_cutoffs += 1
                        if index < from_table:
                            stats.tt_move_cutoffs += 1
                        elif index < promoted:
                            stats.killer_cutoffs += 1
                        order.cutoff(board, col, ply, depth)
                        break

        if table is not None:
            # Which of the three kinds of answer this node produced. Fail-soft is what
            # makes the bounds worth keeping: `best` is the value actually found rather
            # than the edge of the window it was compared against, so both bounds are
            # tighter than a fail-hard search would have been able to store.
            if best <= original_alpha:
                flag = UPPER
            elif best >= beta:
                flag = LOWER
            else:
                flag = EXACT
            table.store(key, depth, flag, to_entry(best, ply), best_col)
        return best
