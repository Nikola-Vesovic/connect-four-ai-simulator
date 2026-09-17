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

**Phase 9 adds iterative deepening**, and it is the first thing in this file that breaks
that last sentence. See `search/deepening.py` for why the algorithm exists; three things
about how it is wired in here:

* **The answer is the last iteration that finished.** A budget can stop the search in the
  middle of an iteration, and that iteration is thrown away whole -- root score, best move
  and all. Keeping the best of the three root moves it managed to look at would be a value
  no depth ever computed.
* **The abort is an exception, and the cost of being able to abort is one integer
  comparison per node.** `stats.nodes >= self._next_check` is `inf` when no budget is
  live, so a depth-budgeted search pays a single predictable branch. When it does fire,
  the node limit is checked exactly and the clock only every `_CLOCK_INTERVAL` nodes,
  because reading a clock is the expensive half. The unwinding exception skips the
  transposition store at the bottom of `_negamax`, which is exactly right -- `best` is
  partial there -- and the root puts the board back by undoing to the ply it started at.
* **Iteration 1 is uninterruptible**, so a legal move comes back however small the budget.

**Phase 10 adds NegaScout**, the fifth switch and the smallest change of the five: one
expression for the window a child is given, and one re-search when the scout was wrong.
See `search/scouting.py` for why the algorithm exists. Three things about how it is wired
in here:

* **The first move of every node, root included, keeps the full window.** A null window
  can only ever answer "better than alpha or not", and something has to establish what
  alpha is worth in the first place.
* **A re-search is charged a node.** It visits the same position a second time and the
  whole subtree under it, and every position inside that subtree is counted again anyway,
  so not counting the re-searched move itself would be the one inconsistency. The effect
  is to make the gain this phase reports an under-statement rather than an over-statement.
* **It is ignored entirely when `pruning` is off**, because a null window around an alpha
  that is never raised is a window below the score scale. That is also what keeps the
  `--verify --agent negamax` node-count gate untouched by this phase.
"""

from __future__ import annotations

import time

from agents.base import Agent
from core.rules import Board
from core.types import Move, Score
from search.deepening import DIRECT, Deepening
from search.evaluation import DEFAULT, Evaluation
from search.ordering import DYNAMIC, MoveOrder, Ordering
from search.scouting import SCOUT, Scout
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

# How often a budgeted search is allowed to read the clock, in nodes. A node budget is
# checked exactly -- it is an integer comparison -- but `perf_counter` is not free, and
# this is what bounds the overshoot: once the deadline passes, the search runs at most
# this many more nodes before noticing. The number was chosen by measuring rather than
# guessed: 2,048 nodes overshot a tenth of a second by 15 ms, and this does not.
_CLOCK_INTERVAL = 64

_NEVER = float("inf")


class _OutOfBudget(Exception):
    """Raised at the node where the budget ran out, caught by the iteration loop.

    Carries nothing: everything the aborted iteration computed is discarded by
    definition, so there is nothing for it to carry.
    """


class NegamaxAgent(Agent):
    def __init__(
        self,
        depth: int = 4,
        *,
        pruning: bool = True,
        evaluation: Evaluation = DEFAULT,
        ordering: Ordering = DYNAMIC,
        transposition: Transposition = TABLE,
        deepening: Deepening = DIRECT,
        scouting: Scout = SCOUT,
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
        self.deepening = deepening
        self.scouting = scouting
        self._table = transposition.start() if transposition.persistent else None
        # Only a random base order consumes it, and it is kept rather than drawn from
        # `self._rng` so that the same position always produces the same tree.
        self._ordering_seed = seed
        # The live budget, for the duration of one search. `_next_check` is the whole cost
        # of being interruptible: one comparison per node against a number that is
        # infinite whenever there is nothing to enforce.
        self._next_check: float = _NEVER
        self._node_limit: float = _NEVER
        self._deadline: float = _NEVER
        suffix = "" if evaluation is DEFAULT else f"-{evaluation.name}"
        if ordering != DYNAMIC:
            suffix += f"-{ordering.name}"
        if transposition != TABLE:
            suffix += f"-{transposition.name}"
        if deepening != DIRECT:
            suffix += f"-{deepening.name}"
        if scouting != SCOUT:
            suffix += f"-{scouting.name}"
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
        stats.depth_cap = depth
        # Created once per MOVE and used by every iteration of it. That is the whole of
        # "shared between iterations, thrown away between moves" -- Phase 8 measured why
        # the second half has to stay true (BUILD_PLAN, thesis point P21).
        table = self._table if self.transposition.persistent else self.transposition.start()
        if table is not None:
            table.reset_counters()

        if not self.deepening.enabled:
            stats.depth = depth
            stats.iterations = 1
            order = self.ordering.start(depth, self._ordering_seed)
            move, value = self._search_root(board, order, table, depth, -1, stats)
            stats.value = value
            stats.take_table(table)
            return move

        return self._deepen(board, table, depth, budget, stats)

    # ------------------------------------------------------------ iterative deepening

    def _deepen(
        self, board: Board, table, cap: int, budget: Budget, stats: SearchStats
    ) -> Move:
        """Search 1, 2, 3... plies until the budget runs out; answer from the last one
        that finished."""
        deepening = self.deepening
        self._node_limit = _NEVER if budget.nodes is None else budget.nodes
        self._deadline = (
            _NEVER if budget.time is None else time.perf_counter() + budget.time
        )
        root_plies = board.moves_played
        shared = self.ordering.start(cap, self._ordering_seed) if deepening.share_order else None

        answer: tuple[Move, int, int] | None = None  # move, value, depth it came from
        latest: tuple[Move, int, int] | None = None  # the same, ignoring the parity rule
        pv_move = -1

        for depth in range(1, cap + 1):
            # Iteration 1 is never interrupted: without it there would be no move at all.
            if depth == 1:
                self._next_check = _NEVER
            else:
                if stats.nodes >= self._node_limit or time.perf_counter() >= self._deadline:
                    break
                self._next_check = min(self._node_limit, stats.nodes + _CLOCK_INTERVAL)

            order = shared if shared is not None else self.ordering.start(
                depth, self._ordering_seed
            )
            try:
                move, value = self._search_root(
                    board, order, table, depth, pv_move if deepening.root_pv else -1, stats
                )
            except _OutOfBudget:
                # Everything this iteration found is discarded, including whatever the
                # board still has on it.
                while board.moves_played > root_plies:
                    board.undo()
                stats.aborted = True
                break

            stats.iterations += 1
            latest = (move, value, depth)
            pv_move = move
            if deepening.answers(depth):
                answer = latest

        self._next_check = _NEVER
        # `latest` is the fallback for `parity="even"` before the first even iteration has
        # finished; depth 1 always completes, so one of the two is always set.
        move, value, depth = answer if answer is not None else latest
        stats.depth = depth
        stats.value = value
        stats.take_table(table)
        return move

    # --------------------------------------------------------------------- the search

    def _search_root(
        self,
        board: Board,
        order: MoveOrder,
        table,
        depth: int,
        pv_move: Move,
        stats: SearchStats,
    ) -> tuple[Move, int]:
        """One search of the whole tree to `depth`, returning the best move and its value.

        The killer slots at ply 0 are empty whatever happens -- a cutoff is only ever
        recorded inside `_negamax` -- and the root deliberately does not probe the table,
        so the only two things that can reorder it are `pv_move` and, once it has been
        carried over from an earlier iteration, the history table. Both are Phase 9's, and
        both are switches.

        Phase 10 scouts here too, and this is where its central claim lives: the first
        move is searched with the full window and fixes `alpha` at the exact value, every
        later move is asked only whether it beats that, and a move that fails low comes
        back at most `alpha` -- which never clears the **strict** improvement the line
        below requires. So the column that gets played is the one alpha-beta played.
        """
        alpha, beta = _BEYOND_LOSS, _BEYOND_WIN
        best_value = _BEYOND_LOSS
        best_move = None
        scouting = self.pruning and self.scouting.scouts(depth)
        for index, col in enumerate(order.columns(board, 0, pv_move)):
            board.play(col)
            stats.nodes += 1
            scouted = scouting and index > 0 and alpha + 1 < beta
            value = -self._negamax(
                board, order, table, depth - 1, 1,
                -(alpha + 1) if scouted else -beta, -alpha, stats,
            )
            if scouted:
                stats.scout_searches += 1
                if alpha < value < beta:
                    stats.researches += 1
                    stats.nodes += 1
                    value = -self._negamax(
                        board, order, table, depth - 1, 1, -beta,
                        -(value if self.scouting.narrow else alpha), stats,
                    )
            board.undo()
            if value > best_value:
                best_value, best_move = value, col
                if self.pruning:
                    alpha = best_value
        return best_move, best_value

    def _out_of_budget(self, stats: SearchStats) -> None:
        """Called only when `stats.nodes` has reached the next checkpoint."""
        if stats.nodes >= self._node_limit or time.perf_counter() >= self._deadline:
            raise _OutOfBudget
        self._next_check = min(self._node_limit, stats.nodes + _CLOCK_INTERVAL)

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
        if stats.nodes >= self._next_check:  # infinite unless a budget is being enforced
            self._out_of_budget(stats)
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
        scouting = self.pruning and self.scouting.scouts(depth)
        narrow = self.scouting.narrow
        for index, col in enumerate(columns):
            board.play(col)
            stats.nodes += 1
            # The first move is asked what this node is worth and needs the real window
            # to answer. Every later one is only asked whether it beats `alpha`, and
            # `(alpha, alpha + 1)` is the narrowest window that can put that question --
            # no integer value sits strictly inside it, so the search cannot come back
            # undecided. With scouting off this expression is `beta` at every move, which
            # is Phase 9's search unchanged.
            scouted = scouting and index > 0 and alpha + 1 < beta
            value = -self._negamax(
                board, order, table, depth - 1, ply + 1,
                -(alpha + 1) if scouted else -beta, -alpha, stats,
            )
            if scouted:
                stats.scout_searches += 1
                if alpha < value < beta:
                    # The scout was wrong: this move may be better, and a null window
                    # cannot say by how much. `value` is a fail-soft *lower* bound on the
                    # move's true value, so re-searching from it rather than from `alpha`
                    # cannot lose the answer -- see `search/scouting.py`.
                    stats.researches += 1
                    stats.nodes += 1
                    value = -self._negamax(
                        board, order, table, depth - 1, ply + 1, -beta,
                        -(value if narrow else alpha), stats,
                    )
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
