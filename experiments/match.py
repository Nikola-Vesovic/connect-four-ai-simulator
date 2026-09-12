"""Headless agent-vs-agent runner: the minimum needed to produce the Phase 4 gate numbers.

Connect Four is not symmetric. Phase 2 measured that with random play the first player
already wins 55.4% of games, so a win rate quoted without saying who moved first is not a
statement about skill. Every result here is therefore split by seat, and the two agents
swap seats on alternate games.

Reproducibility: one master seed produces a fixed list of per-game seed pairs, and each
game depends on nothing except its own pair. A re-run gives identical outcomes, and would
still do so if the games were ever played out of order or in parallel. Wall-clock times
are the one thing that does not reproduce, so `fingerprint` leaves them out.

    python -m experiments.match --a greedy --b random --games 10000 --seed 2026
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence

from agents.base import Agent
from agents.greedy_agent import GreedyAgent
from agents.minimax import MinimaxAgent
from agents.random_agent import RandomAgent
from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from core.types import GameResult, Move
from search.stats import UNLIMITED, Budget

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "minimax": MinimaxAgent,
}
BOARDS = {"bitboard": Bitboard, "naive": NaiveBoard}


@dataclass(frozen=True, slots=True)
class GameRecord:
    result: GameResult  # ONE_WINS means the agent that moved first won
    plies: int
    first_moves: int  # plies the agent actually chose; an opening belongs to neither
    second_moves: int
    first_nodes: int
    second_nodes: int
    first_seconds: float
    second_seconds: float


@dataclass(slots=True)
class Tally:
    """Wins, draws and losses from one agent's point of view."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Chess scoring: a win is 1, a draw a half. 0.5 means evenly matched."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0


@dataclass(slots=True)
class MatchResult:
    agent_a: str
    agent_b: str
    seed: int
    a_first: Tally = field(default_factory=Tally)  # games A started, scored from A's side
    b_first: Tally = field(default_factory=Tally)  # games B started, scored from A's side
    plies: int = 0
    moves_a: int = 0
    moves_b: int = 0
    nodes_a: int = 0
    nodes_b: int = 0
    seconds_a: float = 0.0
    seconds_b: float = 0.0

    @property
    def games(self) -> int:
        return self.a_first.games + self.b_first.games

    @property
    def overall(self) -> Tally:
        return Tally(
            self.a_first.wins + self.b_first.wins,
            self.a_first.draws + self.b_first.draws,
            self.a_first.losses + self.b_first.losses,
        )

    @property
    def mean_plies(self) -> float:
        return self.plies / self.games if self.games else 0.0

    @property
    def first_mover_wins(self) -> int:
        """Games won by whoever moved first, either agent.

        Both tallies are scored from A's side, so in the games B started it is A's *losses*
        that were first-mover wins. Worth a named property: reading it the other way is an
        easy mistake, and it silently turns a 55% first-mover edge into a 47% one.
        """
        return self.a_first.wins + self.b_first.losses

    @property
    def second_mover_wins(self) -> int:
        return self.a_first.losses + self.b_first.wins

    def fingerprint(self) -> tuple:
        """Everything about this run that a re-run from the same seed must reproduce."""
        return (
            self.agent_a,
            self.agent_b,
            self.seed,
            (self.a_first.wins, self.a_first.draws, self.a_first.losses),
            (self.b_first.wins, self.b_first.draws, self.b_first.losses),
            self.plies,
            self.nodes_a,
            self.nodes_b,
        )

    def digest(self) -> str:
        """Short stable hash of the fingerprint. `hash()` would not do: CPython randomises
        string hashing per process, so it differs between runs of the same experiment."""
        return hashlib.sha256(repr(self.fingerprint()).encode()).hexdigest()[:8]


def play_game(
    first: Agent,
    second: Agent,
    *,
    board_class=Bitboard,
    budget: Budget = UNLIMITED,
    opening: Sequence[Move] = (),
) -> GameRecord:
    """One game. `first` gets Player.ONE, which on a fresh board is whoever moves first.

    `opening` is played before either agent is consulted, and counts towards the game's
    length. It is how two deterministic agents can be measured against each other at all
    (see `play_match`); with the default empty opening nothing about a game changes.
    """
    board = board_class()
    for col in opening:
        board.play(col)
    seats = (first, second)
    moves = [0, 0]
    nodes = [0, 0]
    seconds = [0.0, 0.0]

    while not board.is_terminal():
        seat = board.moves_played % 2
        move, stats = seats[seat].choose_move(board, budget)
        moves[seat] += 1
        nodes[seat] += stats.nodes
        seconds[seat] += stats.elapsed
        board.play(move)

    return GameRecord(board.result(), board.moves_played, *moves, *nodes, *seconds)


def game_seeds(seed: int, games: int) -> list[tuple[int, int]]:
    """The seed pair each game is played from. Fixed by the master seed alone, so a game's
    outcome never depends on which games ran before it."""
    rng = random.Random(seed)
    return [(rng.randrange(2**63), rng.randrange(2**63)) for _ in range(games)]


def random_opening(seed: int, plies: int, board_class=Bitboard) -> tuple[Move, ...]:
    """A uniformly random legal opening of `plies` moves that does not end the game."""
    rng = random.Random(f"opening-{seed}")
    while True:
        board = board_class()
        for _ in range(plies):
            board.play(rng.choice(board.legal_moves()))
            if board.is_terminal():
                break
        if not board.is_terminal():
            return tuple(board.history)


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    games: int,
    seed: int,
    *,
    board_class=Bitboard,
    budget: Budget = UNLIMITED,
    opening_plies: int = 0,
) -> MatchResult:
    """Play `games` games, swapping seats every game, and tally them from A's side.

    `opening_plies` exists because two deterministic agents play the same game every
    time: without it, a thousand games between minimax and greedy are one game reported
    a thousand times. A few random plies before either agent is consulted turn the match
    back into a sample. Consecutive games share their opening, so the two agents meet
    each position once from each seat and the seat split stays a paired comparison
    rather than two independent ones.
    """
    match = MatchResult(agent_a.name, agent_b.name, seed)
    seeds = game_seeds(seed, games)

    for index, (seed_a, seed_b) in enumerate(seeds):
        agent_a.new_game(seed_a)
        agent_b.new_game(seed_b)
        a_first = index % 2 == 0
        first, second = (agent_a, agent_b) if a_first else (agent_b, agent_a)
        opening = (
            random_opening(seeds[index - index % 2][0], opening_plies, board_class)
            if opening_plies
            else ()
        )
        game = play_game(
            first, second, board_class=board_class, budget=budget, opening=opening
        )

        tally = match.a_first if a_first else match.b_first
        if game.result is GameResult.DRAW:
            tally.draws += 1
        elif (game.result is GameResult.ONE_WINS) == a_first:
            tally.wins += 1
        else:
            tally.losses += 1

        match.plies += game.plies
        if a_first:
            match.moves_a += game.first_moves
            match.moves_b += game.second_moves
            match.nodes_a += game.first_nodes
            match.nodes_b += game.second_nodes
            match.seconds_a += game.first_seconds
            match.seconds_b += game.second_seconds
        else:
            match.moves_b += game.first_moves
            match.moves_a += game.second_moves
            match.nodes_b += game.first_nodes
            match.nodes_a += game.second_nodes
            match.seconds_b += game.first_seconds
            match.seconds_a += game.second_seconds

    return match


def format_report(match: MatchResult) -> str:
    def row(label: str, tally: Tally) -> str:
        return (
            f"{label:<22}{tally.wins:>8}{tally.draws:>8}{tally.losses:>8}"
            f"{tally.score:>10.1%}"
        )

    def cost(moves: int, nodes: int, seconds: float) -> str:
        if not moves:
            return "no moves"
        return f"{nodes / moves:>7.1f} nodes/move, {1000 * seconds / moves:>6.3f} ms/move"

    lines = [
        f"A = {match.agent_a}   B = {match.agent_b}   "
        f"games = {match.games}   seed = {match.seed}",
        "",
        f"{'':<22}{'A wins':>8}{'draws':>8}{'B wins':>8}{'A score':>10}",
        row("A moves first", match.a_first),
        row("B moves first", match.b_first),
        row("overall", match.overall),
        "",
        f"mean game length   {match.mean_plies:.1f} plies",
        f"first mover won    {match.first_mover_wins / match.games:.1%} "
        f"(random-vs-random baseline: 55.4%)",
        f"A ({match.agent_a}): {cost(match.moves_a, match.nodes_a, match.seconds_a)}",
        f"B ({match.agent_b}): {cost(match.moves_b, match.nodes_b, match.seconds_b)}",
        f"fingerprint  {match.digest()}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a seeded match between two agents.")
    parser.add_argument("--a", default="greedy", choices=sorted(AGENTS))
    parser.add_argument("--b", default="random", choices=sorted(AGENTS))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--board", default="bitboard", choices=sorted(BOARDS))
    parser.add_argument("--opening-plies", type=int, default=0)
    args = parser.parse_args()

    match = play_match(
        AGENTS[args.a](),
        AGENTS[args.b](),
        games=args.games,
        seed=args.seed,
        board_class=BOARDS[args.board],
        opening_plies=args.opening_plies,
    )
    print(format_report(match))


if __name__ == "__main__":
    main()
