"""Phase 13: scoring an agent against the objectively correct move instead of an opponent.

**No course algorithm.** What it contains is the one measurement in this thesis that does
not compare two agents. Thesis point P2 is the argument for it in one line: greedy's 98.6%
against random measures partly how good greedy is and partly how bad random is at
punishing. A win rate against an opponent is a blunt instrument; the true value of a
position is not.

    python -m experiments.benchmark --verify --positions 1000 --workers 8
    python -m experiments.benchmark --score  --positions 1000,500 --workers 8
    python -m experiments.benchmark --check 8 --sets end-easy --positions 24

Three steps, in this order, and step 1 is not skippable.

1. **Prove the oracle.** Our own full-depth search is compared against Pascal Pons's
   published scores. It is a two-way validation: the artifact validates our solver and our
   solver validates the artifact in one run.
2. **Use the oracle to value every child.** The file gives the value of a *position*, not
   of each move, so it cannot judge a move directly. `Oracle.move_values` closes that gap.
3. **Score each agent** on optimal-move rate, blunder rate and mean value loss.

**The three metrics are three different things and conflating them is the easiest mistake
in the phase, so they are defined here before anything computes them.**

* **Optimal** means the move is in the set that preserves the true value. There is usually
  more than one such move, so it is never "the best move".
* A **blunder** changes the theoretical OUTCOME -- win to draw, win to loss, draw to loss.
  Playing a win in 11 instead of a win in 5 is *suboptimal and not a blunder*.
* **Mean value loss** is on the exact scale, so inside one outcome class it is literally
  the number of extra plies; across classes it is about 10,000 or 20,000. The mean over
  everything is therefore dominated by blunders, and it is reported decomposed: the mean
  *ply* loss over moves that did not blunder, beside the blunder rate.

**Two populations are reported separately rather than averaged, and the reason is P2
applied in advance.** A position where every legal move preserves the value cannot tell
two agents apart, and a metric a broken agent still passes is not a metric -- so the
**discriminating** count is reported and every rate is given over that subset as well. And
a **lost** position cannot contain a blunder at all, because every move loses; a 0.0%
blunder rate there is a property of the population and not a strength. Everything is
therefore also split by whether the side to move is winning, drawing or losing, which is
this phase's analogue of P1's seat split.

**Parallel by position, not by game.** `experiments/runner.py` shards the games of a
pairing; this shards positions, and the work unit is a single position because the cost of
one varies by orders of magnitude inside one set. The four things that made Phase 12's gate
work are kept: a module-level picklable work function, the `spawn` context, a work unit
that is a pure function of its index -- including the seed every stochastic agent is given
-- and a digest compared at one worker and at eight.

**The input is untrusted.** The files arrived over plain HTTP, because the site's HTTPS
certificate does not match its host (`data/testsets/gamesolver/SOURCE.txt`). `load_set`
therefore parses strictly and raises: every move string must replay into a legal,
never-terminal game, and every score must be on Pons's scale. Nothing is silently skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import math
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from agents.oracle import (
    DEFAULT_BUDGET,
    DRAW,
    LOSS,
    MAX_PONS,
    WIN,
    Oracle,
    Unsettled,
    outcome_of,
    to_pons,
)
from core.bitboard import Bitboard
from core.rules import CELLS, COLS
from core.types import Score
from experiments.runner import TimedMatchRefused, TournamentConfig
from experiments.tournament import AgentSpec
from search.stats import Budget

ROOT = Path(__file__).resolve().parent.parent
TESTSETS = ROOT / "data" / "testsets" / "gamesolver"
RESULTS = ROOT / "data" / "results"
LADDER = ROOT / "data" / "configs" / "ladder.json"

# In the order SOURCE.txt lists them, which is also cheapest first. The two marked
# solvable are the two a Python full-depth search can finish; see `REACH` below.
SETS: tuple[tuple[str, str], ...] = (
    ("end-easy", "Test_L3_R1"),
    ("middle-easy", "Test_L2_R1"),
    ("middle-medium", "Test_L2_R2"),
    ("begin-easy", "Test_L1_R1"),
    ("begin-medium", "Test_L1_R2"),
    ("begin-hard", "Test_L1_R3"),
)
FILES = dict(SETS)
ORDINAL = {label: index for index, (label, _) in enumerate(SETS)}

# The two sets this phase SCORES agents on. Scoring needs the true value of every child,
# which is about seven solves, so a set can be provable and still not be scoreable -- and
# Begin-Easy is exactly that case.
SCOREABLE = ("end-easy", "middle-easy")
SOLVABLE = SCOREABLE  # the name Phase 13 was drafted with, kept so nothing else breaks

# **Measured 2026-09-19, and the first version of this table was WRONG.** It said Begin-Easy
# was out of reach, which is true of an *exhaustive* solve and false of a *proof*: a value
# in the win or loss band is exact however early the search stopped, so a bounded prover
# settles all thirty Begin-Easy positions tried. Only a draw needs the whole tree.
REACH = {
    "end-easy": "1000/1000 settled at 1,000,000 nodes, 772 nodes and 19.7 ms each",
    "middle-easy": "500/500 settled at 1,000,000 nodes, 151,703 nodes and 3.2 s each",
    "middle-medium": "21/30 settled at 2,000,000 nodes, 21.7 s each",
    "begin-easy": (
        "30/30 settled at 2,000,000 nodes, 52.9 s each -- provable but NOT scoreable, "
        "because valuing every child is about seven times that"
    ),
    "begin-medium": "not settled at 2,000,000 nodes: reached depth 15-21 of 28-32",
    "begin-hard": "not settled at 2,000,000 nodes: reached depth 14-17 of 33-37",
}

OUTCOME_NAMES = {WIN: "win", DRAW: "draw", LOSS: "loss"}


class BadPosition(ValueError):
    """Raised when a line of a downloaded test set is not a position we can trust."""


# -------------------------------------------------------------------------- the loader


@dataclass(frozen=True)
class Position:
    label: str
    index: int
    moves: str
    score: int  # published, on Pons's scale
    stones: int

    def board(self) -> Bitboard:
        return Bitboard.from_move_string(self.moves)


def parse_position(label: str, number: int, line: str) -> Position:
    """One line of a downloaded file, validated rather than believed.

    A line is accepted only if it has exactly two fields, the move string is made of
    columns 1-7, replaying it never plays into a full column and never continues past a
    finished game, the position that results is still in progress, and the score is on
    Pons's scale. Anything else raises: a benchmark that quietly drops the lines it does
    not understand is a benchmark over an unknown sample.
    """
    if not line.strip():
        raise BadPosition(f"{label} line {number}: blank line in a position file.")
    parts = line.split()
    if len(parts) != 2:
        raise BadPosition(f"{label} line {number}: expected '<moves> <score>', got {line!r}.")
    moves, raw = parts
    try:
        score = int(raw)
    except ValueError:
        raise BadPosition(f"{label} line {number}: score {raw!r} is not an integer.") from None
    if not -MAX_PONS <= score <= MAX_PONS:
        raise BadPosition(
            f"{label} line {number}: score {score} is outside Pons's scale "
            f"[-{MAX_PONS}, {MAX_PONS}]."
        )
    if not moves or len(moves) > CELLS:
        raise BadPosition(f"{label} line {number}: {len(moves)} moves is not a game.")
    board = Bitboard()
    for ply, character in enumerate(moves, start=1):
        if character not in "1234567":
            raise BadPosition(f"{label} line {number}: {character!r} is not a column.")
        column = int(character) - 1
        if board.is_terminal():
            raise BadPosition(f"{label} line {number}: ply {ply} continues a finished game.")
        if not board.can_play(column):
            raise BadPosition(f"{label} line {number}: ply {ply} plays a full column.")
        board.play(column)
    if board.is_terminal():
        raise BadPosition(f"{label} line {number}: the position is already over.")
    return Position(label, number - 1, moves, score, len(moves))


def load_set(label: str, limit: int | None = None) -> tuple[Position, ...]:
    """Parse one file strictly. Every failure raises; nothing is skipped."""
    if label not in FILES:
        raise BadPosition(f"unknown test set {label!r}; known are {sorted(FILES)}.")
    path = TESTSETS / FILES[label]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing optional benchmark input {path.name}. "
            "Run: python -m experiments.download_testsets"
        ) from error
    positions = []
    for number, line in enumerate(lines, start=1):
        positions.append(parse_position(label, number, line))
        if limit is not None and len(positions) >= limit:
            break
    return tuple(positions)


# --------------------------------------------------------------------- the work units


def position_seed(master: int, label: str, index: int) -> int:
    """A seed that is a pure function of the position's identity and nothing else.

    Several agents carry state and some are stochastic, so without this the score would
    depend on which worker happened to get which position -- MCTS with random rollouts is
    the case that catches it. `hash()` is deliberately not involved: string hashing is
    randomised per process, and a spawned worker is a different process.
    """
    return master * 1_000_003 + ORDINAL[label] * 100_003 + index


@dataclass(frozen=True)
class VerifyTask:
    """Step 1: solve one position and compare it with the published score."""

    position: Position
    oracle_nodes: int = DEFAULT_BUDGET.nodes


@dataclass(frozen=True)
class ScoreTask:
    """Steps 2 and 3: value every child, then ask every agent for its move."""

    position: Position
    agents: tuple[AgentSpec, ...]
    budget: Budget
    seed: int
    oracle_nodes: int = DEFAULT_BUDGET.nodes


@dataclass(frozen=True)
class AgentMove:
    label: str
    move: int
    optimal: bool
    blunder: bool
    value_loss: int
    nodes: int
    seconds: float


@dataclass(frozen=True)
class Verdict:
    """One position, fully measured. Everything the digest reads is an integer."""

    label: str
    index: int
    stones: int
    legal: int
    published: int
    ours: int
    value: int
    outcome: int
    optimal_count: int
    discriminating: bool
    blunder_available: bool
    oracle_nodes: int
    oracle_seconds: float
    moves: tuple[AgentMove, ...] = ()
    settled: bool = True

    @property
    def agrees(self) -> bool:
        return self.ours == self.published

    @property
    def agrees_on_outcome(self) -> bool:
        return (self.ours > 0) - (self.ours < 0) == (self.published > 0) - (self.published < 0)


def _unsettled(position: Position, nodes: int, seconds: float) -> Verdict:
    """A position the oracle could not prove. Kept as a record rather than dropped, so the
    number excluded is reported beside every rate instead of quietly shrinking the sample."""
    return Verdict(
        label=position.label,
        index=position.index,
        stones=position.stones,
        legal=0,
        published=position.score,
        ours=0,
        value=0,
        outcome=0,
        optimal_count=0,
        discriminating=False,
        blunder_available=False,
        oracle_nodes=nodes,
        oracle_seconds=seconds,
        settled=False,
    )


def verify_position(task: VerifyTask) -> Verdict:
    """Module level and picklable, which is what `spawn` needs: a worker re-imports this
    module and looks the function up by name."""
    position = task.position
    board = position.board()
    oracle = Oracle(Budget(nodes=task.oracle_nodes))
    started = time.perf_counter()
    try:
        value, stats = oracle.solve(board)
    except Unsettled:
        return _unsettled(position, task.oracle_nodes, time.perf_counter() - started)
    seconds = time.perf_counter() - started
    return Verdict(
        label=position.label,
        index=position.index,
        stones=position.stones,
        legal=len(board.legal_moves()),
        published=position.score,
        ours=to_pons(value, position.stones),
        value=value,
        outcome=outcome_of(value),
        optimal_count=0,
        discriminating=False,
        blunder_available=False,
        oracle_nodes=stats.nodes,
        oracle_seconds=seconds,
    )


def score_position(task: ScoreTask) -> Verdict:
    """The same, plus every child valued and every agent asked.

    The position's own value is derived as the maximum over its children rather than
    solved separately, so the comparison against the published score comes for free here
    too -- and it is a sharper check than step 1's, because it would also catch the
    off-by-one that makes a child's mate distance a ply short.

    A position whose children cannot all be proved has no ground truth, so no agent is
    asked about it at all: scoring a move against a heuristic guess would be scoring it
    against an opinion.
    """
    position = task.position
    board = position.board()
    oracle = Oracle(Budget(nodes=task.oracle_nodes))
    started = time.perf_counter()
    try:
        table = oracle.move_values(board)
    except Unsettled:
        return _unsettled(position, task.oracle_nodes, time.perf_counter() - started)
    seconds = time.perf_counter() - started

    seed = position_seed(task.seed, position.label, position.index)
    moves = []
    for spec in task.agents:
        agent = spec.build()
        agent.new_game(seed)
        move, stats = agent.choose_move(board, task.budget)
        moves.append(
            AgentMove(
                label=spec.label,
                move=move,
                optimal=table.is_optimal(move),
                blunder=table.is_blunder(move),
                value_loss=table.loss(move),
                nodes=stats.nodes,
                seconds=stats.elapsed,
            )
        )

    return Verdict(
        label=position.label,
        index=position.index,
        stones=position.stones,
        legal=len(table.values),
        published=position.score,
        ours=to_pons(table.value, position.stones),
        value=table.value,
        outcome=outcome_of(table.value),
        optimal_count=len(table.optimal),
        discriminating=table.discriminating,
        blunder_available=table.blunder_available,
        oracle_nodes=table.nodes,
        oracle_seconds=seconds,
        moves=tuple(moves),
    )


# ------------------------------------------------------------------------------ the run


def digest_of(verdicts: Sequence[Verdict]) -> str:
    """The artifact the reproducibility gate compares.

    Wall-clock seconds are excluded by construction, for the reason `MatchResult.digest`
    gives: they are the one thing that cannot reproduce. Everything else here is an
    integer or a flag, so merging in any order gives the same answer -- the verdicts are
    nevertheless sorted, so the report is stable too.
    """
    parts = []
    for verdict in sorted(verdicts, key=lambda v: (ORDINAL[v.label], v.index)):
        parts.append(
            f"{verdict.label}/{verdict.index}/{int(verdict.settled)}/{verdict.value}/"
            f"{verdict.ours}/{verdict.published}/{verdict.optimal_count}/"
            f"{int(verdict.discriminating)}/{int(verdict.blunder_available)}"
        )
        for move in verdict.moves:
            parts.append(
                f"{move.label}={move.move}/{int(move.optimal)}/{int(move.blunder)}/"
                f"{move.value_loss}/{move.nodes}"
            )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def execute(tasks: Sequence, worker: Callable, workers: int, progress: bool = True) -> tuple[list[Verdict], float]:
    """Run the tasks, in this process at `workers <= 1` and in spawned children above it.

    `workers <= 1` runs with no pool at all, which is what every phase before Phase 12 did,
    so the speedup curve compares against the real previous state of the project and the
    reproducibility gate is the strong form of its claim: no multiprocessing at all against
    eight fresh interpreters.

    One position is one task. Phase 12 could use a contiguous block because every game of a
    pairing costs about the same; here the cost of a position varies by orders of magnitude
    inside one set, so a block would leave one worker holding the tail.
    """
    started = time.perf_counter()
    if workers <= 1:
        verdicts = [worker(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        verdicts = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            for done, verdict in enumerate(pool.map(worker, tasks, chunksize=1), start=1):
                verdicts.append(verdict)
                if progress and (done % 50 == 0 or done == len(tasks)):
                    print(f"  [{done:>4}/{len(tasks)}]", flush=True)
    return verdicts, time.perf_counter() - started


def verify(
    labels: Sequence[str],
    counts: Mapping[str, int],
    workers: int = 1,
    progress: bool = True,
    oracle_nodes: int = DEFAULT_BUDGET.nodes,
) -> tuple[list[Verdict], float]:
    tasks = [
        VerifyTask(position, oracle_nodes)
        for label in labels
        for position in load_set(label, counts.get(label))
    ]
    return execute(tasks, verify_position, workers, progress)


def score(
    labels: Sequence[str],
    counts: Mapping[str, int],
    agents: Sequence[AgentSpec],
    budget: Budget,
    seed: int = 2026,
    workers: int = 1,
    progress: bool = True,
    oracle_nodes: int = DEFAULT_BUDGET.nodes,
) -> tuple[list[Verdict], float]:
    if workers > 1 and budget.time is not None:
        raise TimedMatchRefused(
            f"the benchmark was asked for a wall-clock budget ({budget}) and {workers} "
            "workers. A timed search under load measures the load (P24). Either express "
            "the budget in nodes, or run it with --workers 1."
        )
    tasks = [
        ScoreTask(position, tuple(agents), budget, seed, oracle_nodes)
        for label in labels
        for position in load_set(label, counts.get(label))
    ]
    return execute(tasks, score_position, workers, progress)


# ------------------------------------------------------------------------- aggregation


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _sigma(part: int, whole: int) -> float:
    if whole < 2:
        return 0.0
    rate = part / whole
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / whole)


def oracle_rows(verdicts: Sequence[Verdict], workers: int, seconds: float) -> list[dict]:
    """Step 1's table: what the published scores and our solver said about each other.

    Every rate is over the positions the oracle **settled**, and the number it could not
    settle is a column rather than a footnote: they were excluded from the sample, so the
    reader has to be able to see how big the exclusion was.
    """
    rows = []
    for label in sorted({v.label for v in verdicts}, key=ORDINAL.__getitem__):
        everything = [v for v in verdicts if v.label == label]
        group = [v for v in everything if v.settled]
        exact = sum(1 for v in group if v.agrees)
        outcome = sum(1 for v in group if v.agrees_on_outcome)
        rows.append(
            {
                "set": label,
                "file": FILES[label],
                "attempted": len(everything),
                "settled": len(group),
                "unsettled": len(everything) - len(group),
                "min_stones": min(v.stones for v in everything),
                "max_stones": max(v.stones for v in everything),
                "outcome_agree": outcome,
                "outcome_rate": round(_rate(outcome, len(group)), 6),
                "exact_agree": exact,
                "exact_rate": round(_rate(exact, len(group)), 6),
                "wins": sum(1 for v in group if v.outcome == WIN),
                "draws": sum(1 for v in group if v.outcome == DRAW),
                "losses": sum(1 for v in group if v.outcome == LOSS),
                "mean_nodes": round(sum(v.oracle_nodes for v in group) / max(len(group), 1), 1),
                "mean_ms": round(1000 * sum(v.oracle_seconds for v in group) / max(len(group), 1), 2),
                "workers": workers,
                "seconds": round(seconds, 1),
                "digest": digest_of(everything),
            }
        )
    return rows


def _agent_metrics(group: Sequence[Verdict], label: str) -> dict:
    """The three metrics plus the two populations, for one agent over one set of positions."""
    pairs = [(v, next(m for m in v.moves if m.label == label)) for v in group]
    if not pairs:
        return {}
    optimal = sum(1 for _, m in pairs if m.optimal)
    discriminating = [(v, m) for v, m in pairs if v.discriminating]
    optimal_disc = sum(1 for _, m in discriminating if m.optimal)
    exposed = [(v, m) for v, m in pairs if v.blunder_available]
    blunders = sum(1 for _, m in pairs if m.blunder)
    sound = [m for _, m in pairs if not m.blunder]
    return {
        "positions": len(pairs),
        "optimal": optimal,
        "optimal_rate": round(_rate(optimal, len(pairs)), 4),
        "optimal_sigma": round(_sigma(optimal, len(pairs)), 4),
        "discriminating": len(discriminating),
        "optimal_rate_discriminating": round(_rate(optimal_disc, len(discriminating)), 4),
        "blunder_available": len(exposed),
        "blunders": blunders,
        "blunder_rate": round(_rate(blunders, len(pairs)), 4),
        "blunder_rate_exposed": round(_rate(blunders, len(exposed)), 4),
        "mean_ply_loss": round(sum(m.value_loss for m in sound) / len(sound), 3) if sound else 0.0,
        "mean_value_loss": round(sum(m.value_loss for _, m in pairs) / len(pairs), 1),
        "nodes_per_move": round(sum(m.nodes for _, m in pairs) / len(pairs), 1),
        "ms_per_move": round(1000 * sum(m.seconds for _, m in pairs) / len(pairs), 3),
    }


def agent_rows(
    verdicts: Sequence[Verdict],
    agents: Sequence[AgentSpec],
    budget: Budget,
    workers: int,
) -> list[dict]:
    rows = []
    for label in sorted({v.label for v in verdicts}, key=ORDINAL.__getitem__):
        group = [v for v in verdicts if v.label == label and v.settled]
        if not group:
            continue
        for spec in agents:
            rows.append(
                {
                    "set": label,
                    "agent": spec.label,
                    "agent_name": spec.agent_name,
                    "budget": str(budget),
                    "workers": workers,
                    "mean_legal": round(sum(v.legal for v in group) / len(group), 2),
                    **_agent_metrics(group, spec.label),
                }
            )
    return rows


def outcome_rows(
    verdicts: Sequence[Verdict], agents: Sequence[AgentSpec], budget: Budget
) -> list[dict]:
    """The same metrics split by whether the side to move is winning, drawing or losing.

    This phase's P1. A lost position cannot contain a blunder, so pooling the three would
    dilute the blunder rate with a population in which it is structurally zero.
    """
    rows = []
    for label in sorted({v.label for v in verdicts}, key=ORDINAL.__getitem__):
        for outcome in (WIN, DRAW, LOSS):
            group = [
                v for v in verdicts if v.label == label and v.settled and v.outcome == outcome
            ]
            if not group:
                continue
            for spec in agents:
                rows.append(
                    {
                        "set": label,
                        "outcome": OUTCOME_NAMES[outcome],
                        "agent": spec.label,
                        "budget": str(budget),
                        **_agent_metrics(group, spec.label),
                    }
                )
    return rows


# -------------------------------------------------------------------------- reporting


def format_oracle(rows: Sequence[Mapping]) -> str:
    lines = [
        f"{'set':<15}{'attempted':>10}{'settled':>9}{'stones':>10}{'outcome':>10}"
        f"{'exact':>10}{'nodes/pos':>14}{'ms/pos':>10}{'W/D/L':>16}",
    ]
    for row in rows:
        stones = f"{row['min_stones']}-{row['max_stones']}"
        wdl = f"{row['wins']}/{row['draws']}/{row['losses']}"
        settled = f"{row['settled']} ({_rate(row['settled'], row['attempted']):.0%})"
        lines.append(
            f"{row['set']:<15}{row['attempted']:>10}{settled:>14}{stones:>10}"
            f"{row['outcome_rate']:>9.1%}{row['exact_rate']:>10.1%}"
            f"{row['mean_nodes']:>14,.0f}{row['mean_ms']:>10.1f}{wdl:>16}"
        )
    return "\n".join(lines)


def format_agents(rows: Sequence[Mapping]) -> str:
    lines = [
        f"{'set':<14}{'agent':<11}{'optimal':>10}{'+-':>7}{'on discr.':>11}"
        f"{'blunder':>10}{'when possible':>15}{'ply loss':>10}{'nodes':>9}{'ms':>8}",
    ]
    current = None
    for row in rows:
        if row["set"] != current:
            current = row["set"]
            lines.append("")
        lines.append(
            f"{row['set']:<14}{row['agent']:<11}{row['optimal_rate']:>9.1%}"
            f"{row['optimal_sigma']:>7.1%}{row['optimal_rate_discriminating']:>11.1%}"
            f"{row['blunder_rate']:>10.1%}{row['blunder_rate_exposed']:>15.1%}"
            f"{row['mean_ply_loss']:>10.2f}{row['nodes_per_move']:>9,.0f}"
            f"{row['ms_per_move']:>8.1f}"
        )
    return "\n".join(lines)


def format_populations(verdicts: Sequence[Verdict]) -> str:
    """How much of each set actually measures anything -- reported before any rate is."""
    lines = [
        f"{'set':<15}{'settled':>10}{'unsettled':>11}{'discriminating':>16}"
        f"{'blunder possible':>18}{'optimal moves':>15}{'W/D/L':>16}",
    ]
    for label in sorted({v.label for v in verdicts}, key=ORDINAL.__getitem__):
        everything = [v for v in verdicts if v.label == label]
        group = [v for v in everything if v.settled]
        if not group:
            continue
        discriminating = sum(1 for v in group if v.discriminating)
        exposed = sum(1 for v in group if v.blunder_available)
        wdl = (
            f"{sum(1 for v in group if v.outcome == WIN)}/"
            f"{sum(1 for v in group if v.outcome == DRAW)}/"
            f"{sum(1 for v in group if v.outcome == LOSS)}"
        )
        lines.append(
            f"{label:<15}{len(group):>10}{len(everything) - len(group):>11}"
            f"{discriminating:>10} {_rate(discriminating, len(group)):>5.1%}"
            f"{exposed:>12} {_rate(exposed, len(group)):>5.1%}"
            f"{sum(v.optimal_count for v in group) / len(group):>15.2f}{wdl:>16}"
        )
    return "\n".join(lines)


def write_csv(rows: Sequence[Mapping], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


# -------------------------------------------------------------------------------- CLI


def parse_counts(labels: Sequence[str], raw: str) -> dict[str, int]:
    """`--positions 1000` applies to every set; `--positions 1000,500` is per set."""
    parts = [int(piece) for piece in raw.split(",")]
    if len(parts) == 1:
        parts *= len(labels)
    if len(parts) != len(labels):
        raise ValueError(f"--positions has {len(parts)} values for {len(labels)} sets.")
    return dict(zip(labels, parts))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 13 perfect-solver benchmark.")
    parser.add_argument("--verify", action="store_true", help="step 1: prove the oracle")
    parser.add_argument("--score", action="store_true", help="steps 2 and 3: score the agents")
    parser.add_argument("--check", type=int, default=0, help="compare the digest at 1 and N workers")
    parser.add_argument("--sets", default=",".join(SOLVABLE))
    parser.add_argument("--positions", default="40")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", default=str(LADDER))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--oracle-nodes", type=int, default=DEFAULT_BUDGET.nodes)
    parser.add_argument("--write", action="store_true", help="write the CSVs")
    args = parser.parse_args(argv)

    labels = [label.strip() for label in args.sets.split(",") if label.strip()]
    for label in labels:
        if label not in SCOREABLE:
            print(f"NOTE: {label} -- {REACH[label]}")
    counts = parse_counts(labels, args.positions)
    config = TournamentConfig.load(args.config)
    seed = args.seed if args.seed is not None else config.seed

    if args.check:
        tasks_serial, seconds_serial = score(
            labels, counts, config.agents, config.budget, seed, workers=1, progress=False,
            oracle_nodes=args.oracle_nodes,
        )
        tasks_parallel, seconds_parallel = score(
            labels, counts, config.agents, config.budget, seed, workers=args.check,
            progress=False, oracle_nodes=args.oracle_nodes,
        )
        one, many = digest_of(tasks_serial), digest_of(tasks_parallel)
        print(f"1 worker  {seconds_serial:>7.1f} s   digest {one}")
        print(f"{args.check} workers {seconds_parallel:>7.1f} s   digest {many}")
        print("IDENTICAL" if one == many else "*** DIFFERENT ***")
        return

    if args.verify:
        verdicts, seconds = verify(labels, counts, args.workers, oracle_nodes=args.oracle_nodes)
        rows = oracle_rows(verdicts, args.workers, seconds)
        print()
        print(format_oracle(rows))
        settled = [v for v in verdicts if v.settled]
        print(
            f"\n{len(verdicts)} positions attempted at an oracle budget of "
            f"{args.oracle_nodes:,} nodes; {len(settled)} settled; {seconds:.1f} s at "
            f"{args.workers} worker(s); digest {digest_of(verdicts)}"
        )
        disagreements = [v for v in settled if not v.agrees]
        for verdict in disagreements[:10]:
            print(f"  DISAGREE {verdict.label}/{verdict.index}: published "
                  f"{verdict.published}, ours {verdict.ours}")
        print(f"{len(disagreements)} disagreement(s).")
        if args.write:
            write_csv(rows, RESULTS / "oracle_agreement.csv")

    if args.score:
        verdicts, seconds = score(
            labels, counts, config.agents, config.budget, seed, args.workers,
            oracle_nodes=args.oracle_nodes,
        )
        settled = [v for v in verdicts if v.settled]
        mismatched = [v for v in settled if not v.agrees]
        print()
        print(format_populations(verdicts))
        rows = agent_rows(verdicts, config.agents, config.budget, args.workers)
        print()
        print(format_agents(rows))
        print(
            f"\n{len(verdicts)} positions attempted, {len(settled)} settled at an oracle "
            f"budget of {args.oracle_nodes:,} nodes, x {len(config.agents)} agents in "
            f"{seconds:.1f} s at {args.workers} worker(s); budget {config.budget} "
            f"({config.budget_unit}); seed {seed}; digest {digest_of(verdicts)}"
        )
        print(
            f"root value from the children agrees with the published score on "
            f"{len(settled) - len(mismatched)}/{len(settled)}"
        )
        for verdict in mismatched[:10]:
            print(f"  DISAGREE {verdict.label}/{verdict.index}: published "
                  f"{verdict.published}, ours {verdict.ours}")
        if args.write:
            write_csv(rows, RESULTS / "benchmark_by_agent.csv")
            write_csv(
                outcome_rows(verdicts, config.agents, config.budget),
                RESULTS / "benchmark_by_outcome.csv",
            )


if __name__ == "__main__":
    main()
