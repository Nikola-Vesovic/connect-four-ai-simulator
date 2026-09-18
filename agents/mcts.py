"""Monte Carlo tree search with UCT. No neural networks anywhere.

The second of the two approaches the mentor approved, and the only algorithm in this
project that is not in the course deck. See `search/playout.py` for what the four steps
are for, why design principle 3 does not apply to this file, and what replaces the frozen
value gate. This module is how it is wired up.

**The tree holds moves and counters, never boards.** One board is played down during
selection and undone on the way back up, which is the same play/undo pair every search in
this project uses and costs two integer operations per ply on the bitboard. It also means
the `Agent` seam's balance check -- the one that has caught the commonest bug of Phases
6-10 -- is watching a search that does thousands of plays and undos per move, which is
where it earns its keep.

**A child is created at the moment it is first visited**, never before. That is not a
detail: UCB1 divides by a child's visit count, so the usual implementation has to special
case children with no visits and invent a value for them. Here there are none -- a node
exists only if it has been simulated at least once -- and the "order the unvisited children
are tried in" question becomes an explicit list (`Playout.order`) instead of a hidden
consequence of how the formula was patched.

**A terminal node is never rolled out.** Its value is known exactly, so it is backed up
exactly, every time it is selected. That is plain UCT rather than an enhancement: rolling
out a finished game would be sampling something that has no randomness left in it.
Propagating that certainty *upwards* is the enhancement, and it is `Playout.solver`.

**What the counters mean, because MCTS does not count what alpha-beta counts.** An
iteration of this algorithm is not a node of a tree search: it walks a path, adds one
node, and plays a whole game out. So two units are reported and every comparison says
which one it used:

* `simulations` -- complete select / expand / simulate / backpropagate cycles. This is the
  budget a user sets and the unit the MCTS literature quotes.
* `nodes` -- **positions reached by making a move**, which is the unit every agent in this
  project has counted since Phase 4: tree descents, expansions, rollout plies and, with
  greedy rollouts, the probe positions the policy looks at. It is the only unit in which
  MCTS and alpha-beta can be put on the same axis at all.

`depth` is the deepest the *tree* got, not the rollouts, and it is a result in the sense
Phase 9 made the word mean. `leaves` is the number of simulations that ended at a terminal
node already in the tree and therefore needed no rollout at all.
"""

from __future__ import annotations

import math
import time

from agents.base import Agent
from agents.greedy_agent import GreedyAgent
from core.rules import COLS, Board
from core.types import Move, Score
from search.ordering import CENTER_FIRST, LEFT_TO_RIGHT
from search.playout import UCT, Playout
from search.stats import UNLIMITED, Budget, SearchStats

_NEVER = float("inf")

# Rewards live on [0, 1] so the exploration constant means what it means in the
# literature. The flip a parent applies to a child's result is the reflection of this
# interval, which is what negamax does with a minus sign.
WIN = 1.0
DRAW = 0.5
LOSS = 0.0


class _Node:
    """One move in the tree, and what the simulations through it have said about it.

    `reward` is accumulated **from the point of view of the player who played `move`**,
    not of the player to move here. That is Q(s, a) -- the value of the move rather than
    of the position -- and it is what lets a parent read a child's mean straight out of it
    with no negation, leaving exactly one place where the flip happens.

    `proof` is `None` while the node's value is only an average. A terminal node gets its
    exact value here when it is created, whatever the settings; anything above a terminal
    node gets one only with `Playout.solver` on.
    """

    __slots__ = ("move", "parent", "children", "untried", "visits", "reward", "proof")

    def __init__(self, move: Move, parent: "_Node | None", untried: list[Move],
                 proof: float | None) -> None:
        self.move = move
        self.parent = parent
        self.children: list[_Node] = []
        self.untried = untried
        self.visits = 0
        self.reward = 0.0
        self.proof = proof

    @property
    def mean(self) -> float:
        return self.reward / self.visits if self.visits else 0.0


class MctsAgent(Agent):
    def __init__(
        self,
        iterations: int = 1000,
        *,
        playout: Playout = UCT,
        seed: int | None = None,
    ) -> None:
        if iterations < 1:
            raise ValueError(f"MCTS needs at least one simulation, got {iterations}.")
        super().__init__(seed)
        self.iterations = iterations
        self.playout = playout
        self._base = CENTER_FIRST if playout.order == "center" else LEFT_TO_RIGHT
        # The rollout policy, reused rather than reimplemented: this is Phase 4's greedy
        # agent, audited there against an independent implementation of its own spec.
        self._greedy = GreedyAgent() if playout.rollout == "greedy" else None
        # One scratch record for the whole life of the agent. The greedy policy fills in
        # a `SearchStats` as it probes, and its counters must not land in the real one --
        # it writes `depth` and `value` too. Only the node delta is taken.
        self._scratch = SearchStats()
        self._root: _Node | None = None
        self._root_history: tuple[Move, ...] = ()
        # Value equality, not identity -- Phase 7's rule: a spec built freshly from the
        # window's checkboxes must report the same name as the shared constant it equals.
        suffix = "" if playout == UCT else f"-{playout.name}"
        self.name = f"mcts-{iterations}{suffix}"

    def new_game(self, seed: int | None = None) -> None:
        """The tree is the one piece of state that outlives a search, so a new game is
        where it has to go. This is the hook `Agent.new_game` was given in Phase 4 for
        exactly this phase."""
        super().new_game(seed)
        self._root = None
        self._root_history = ()

    # ------------------------------------------------------------------------ the search

    def _select_move(self, board: Board, budget: Budget, stats: SearchStats) -> Move:
        node_limit = _NEVER if budget.nodes is None else budget.nodes
        deadline = _NEVER if budget.time is None else time.perf_counter() + budget.time
        root = self._root_for(board, stats)

        for index in range(self.iterations):
            # The first simulation is uninterruptible, for Phase 9's reason: without it
            # the root has no children and there is no move to give back.
            if index and (stats.nodes >= node_limit or time.perf_counter() >= deadline):
                break
            if root.proof is not None:  # the position is solved; more sampling adds nothing
                break
            self._simulate(board, root, stats)
            stats.simulations += 1

        best = self._choose(root)
        self._report(root, best, stats)
        return best.move

    def _simulate(self, board: Board, root: _Node, stats: SearchStats) -> None:
        """One select / expand / simulate / backpropagate cycle. Leaves the board as it
        found it."""
        node = root
        depth = 0

        # SELECTION -- down through nodes that are fully expanded and not yet decided.
        while node.proof is None and not node.untried:
            node = self._best_child(node)
            board.play(node.move)
            stats.nodes += 1
            depth += 1

        # EXPANSION -- one child, and only if there is something left to learn here.
        if node.proof is None:
            node = self._expand(node, board, stats)
            depth += 1

        # SIMULATION -- a terminal node already knows the answer, so it is never played out.
        if node.proof is not None:
            reward = node.proof
            stats.leaves += 1
        else:
            reward = self._rollout(board, stats)
            stats.rollouts += 1

        if depth > stats.depth:
            stats.depth = depth

        # BACKPROPAGATION -- up the same path, undoing the board and **flipping the reward
        # at every level**: a game that is a win for the player who moved into this node
        # is a loss for the one who moved into its parent. This is negamax's minus sign
        # written for rewards on [0, 1].
        solver = self.playout.solver
        while node is not root:
            node.visits += 1
            node.reward += reward
            board.undo()
            if solver:
                self._prove(node.parent)
            reward = WIN - reward
            node = node.parent
        root.visits += 1

    def _expand(self, node: _Node, board: Board, stats: SearchStats) -> _Node:
        """Add the next untried child and move onto it. The order the seven children are
        first tried in is `Playout.order`, centre-first by default (thesis point P3)."""
        col = node.untried.pop(0)
        board.play(col)
        stats.nodes += 1
        child = _Node(col, node, self._legal(board), self._terminal_value(board))
        node.children.append(child)
        return child

    def _best_child(self, node: _Node) -> _Node:
        """UCB1: what the simulations say, plus what is still unknown.

        No division by zero is possible and none is guarded against: a child exists only
        because a simulation created it, so every `visits` here is at least one.
        """
        c = self.playout.exploration
        log_parent = math.log(node.visits)
        best = None
        best_score = -_NEVER
        for child in node.children:
            score = child.reward / child.visits + c * math.sqrt(log_parent / child.visits)
            if score > best_score:
                best_score, best = score, child
        return best

    def _rollout(self, board: Board, stats: SearchStats) -> float:
        """Play the position out and report the result to whoever moved into it."""
        mover = board.player_to_move.opponent
        plies = 0
        if self._greedy is None:
            choice = self._rng.choice
            while not board.is_terminal():
                board.play(choice(board.legal_moves()))
                plies += 1
        else:
            scratch = self._scratch
            select = self._greedy._select_move
            while not board.is_terminal():
                before = scratch.nodes
                board.play(select(board, UNLIMITED, scratch))
                stats.nodes += scratch.nodes - before
                plies += 1
        stats.nodes += plies

        winner = board.winner()
        reward = DRAW if winner is None else (WIN if winner is mover else LOSS)
        for _ in range(plies):
            board.undo()
        return reward

    # ------------------------------------------------------------------------ the solver

    @staticmethod
    def _prove(node: _Node) -> None:
        """Turn an average into a certainty where the children allow it.

        A child that is a proven win for the side to move here settles the node at once --
        that move will be played. Otherwise every child has to exist and be proven before
        anything can be said, and then the node is worth the best of them, flipped.

        **A proof does nothing to the UCB1 formula, and that is a decision rather than an
        omission.** Selection still ranks children by mean and doubt; what changes is that
        a proven node is never descended into or rolled out, so its exact value is backed
        up on every visit and its mean converges to 0 or 1 instead of wandering. UCB1 then
        abandons a proven loss and settles on a proven win of its own accord. The one place
        a proof is consulted directly is the root's final answer, where an average must not
        be allowed to outrank a certainty.
        """
        if node is None or node.proof is not None:
            return
        best = LOSS
        complete = not node.untried
        for child in node.children:
            if child.proof is None:
                complete = False
            elif child.proof > best:
                best = child.proof
                if best == WIN:
                    node.proof = LOSS
                    return
        if complete and node.children:
            node.proof = WIN - best

    @staticmethod
    def _terminal_value(board: Board) -> float | None:
        """The exact reward of a finished position, to the player who just moved."""
        if board.winner() is not None:  # only the player who just moved can have won
            return WIN
        return DRAW if board.is_full() else None

    # ------------------------------------------------------------------------ the answer

    def _choose(self, root: _Node) -> _Node:
        """Which child to actually play.

        Most visited by default, not highest mean. UCB1 spends its visits where it
        believes, so a visit count is an average over the whole search, while a mean over
        three simulations is noise -- and the small budgets are exactly the case an
        interactive program runs. `root="mean"` is the switch, and the two are measured
        against each other.

        Ties fall to the child created first, which is the centre-first expansion order,
        so the answer is deterministic without a random tie-break.
        """
        by_mean = self.playout.root == "mean"
        solver = self.playout.solver
        best = None
        best_key = None
        for child in root.children:
            score = child.mean if by_mean else float(child.visits)
            # With the solver on, a proven win outranks anything an average can say and a
            # proven loss is played only if there is nothing else.
            rank = 1
            if solver and child.proof is not None:
                rank = 2 if child.proof == WIN else (0 if child.proof == LOSS else 1)
            key = (rank, score)
            if best_key is None or key > best_key:
                best_key, best = key, child
        return best

    def _report(self, root: _Node, best: _Node, stats: SearchStats) -> None:
        counts = [0] * COLS
        for child in root.children:
            counts[child.move] = child.visits
        stats.visits = tuple(counts)
        stats.root_visits = best.visits
        stats.root_reward = best.reward
        stats.tree_nodes = self._count(root)
        if self.playout.solver and best.proof is not None:
            stats.value = (
                Score.WIN if best.proof == WIN
                else Score.LOSS if best.proof == LOSS
                else Score.DRAW
            )

    @staticmethod
    def _count(node: _Node) -> int:
        total = 1
        stack = list(node.children)
        while stack:
            current = stack.pop()
            total += 1
            stack.extend(current.children)
        return total

    # --------------------------------------------------------------------- the tree root

    def _legal(self, board: Board) -> list[Move]:
        if board.is_terminal():
            return []
        legal = set(board.legal_moves())
        return [col for col in self._base if col in legal]

    def _root_for(self, board: Board, stats: SearchStats) -> _Node:
        """The tree this search starts from: the surviving subtree, or a fresh node.

        Reuse walks down from the previous root by the moves that have been played since,
        which is read off `board.history` -- the same attribute `experiments.match` reads
        to record an opening. If anything does not line up (a new game, a board that is
        not a continuation, a reply the search never expanded) the tree is simply rebuilt,
        because a wrong subtree would be a wrong search and a fresh one is merely a slower.
        """
        if self.playout.reuse and self._root is not None:
            played = tuple(board.history)
            start = len(self._root_history)
            if len(played) > start and played[:start] == self._root_history:
                node = self._root
                for col in played[start:]:
                    node = next((c for c in node.children if c.move == col), None)
                    if node is None:
                        break
                if node is not None:
                    node.parent = None
                    stats.tree_reused = node.visits
                    self._root = node
                    self._root_history = played
                    return node

        root = _Node(-1, None, self._legal(board), None)
        self._root = root
        self._root_history = tuple(board.history)
        return root
