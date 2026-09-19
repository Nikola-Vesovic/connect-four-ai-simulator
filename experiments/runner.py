"""Config-driven tournament execution: one JSON file in, CSV out, across cores.

    python -m experiments.runner --config data/configs/ladder.json --workers 8
    python -m experiments.runner --config data/configs/speedup.json --speedup 1,2,4,8
    python -m experiments.runner --config data/configs/speedup.json --check 8

Phase 12 contains **no course algorithm**. It exists because measurement time was the
binding constraint on the two phases that follow: Phase 9's 540-game match pair took
about thirty-six minutes serially, and Phase 11's ablation was sized down to fit, which
is why four of its six switches came back inside 1.2 sigma and could not be separated.

**A timed match is refused in parallel rather than warned about.** A wall-clock budget
under load measures the load. This is not a theoretical worry: Phase 9 ran a 540-game
match under a 0.07 s clock that said 44.3% and its node-budgeted twin that said 48.5%,
and the second was the controlled one -- the opponent simply happened to run 19% slower
than in calibration, so the agent being measured got 13% less clock (thesis point P24).
Eight of those matches sharing four performance cores would not have a control at all.
So `--workers 1` will run a timed config and record that it does not reproduce; anything
above 1 raises.

**Why `--workers 1` is the serial baseline and not one worker process.** It runs the
games in this process with no pool at all, which is exactly what every phase before this
one did. That makes the speedup curve a comparison against the real previous state of
the project, and it makes the reproducibility gate the strongest form of its claim:
byte-identical output between no multiprocessing at all and eight spawned children.

macOS starts children with `spawn`, so a child is a fresh interpreter that re-imports
this module. Everything here is therefore import-safe, the entry point is behind
`if __name__ == "__main__"`, and `tests/test_session.py` asserts in a fresh interpreter
that importing it leaves pygame out of `sys.modules` -- which is the same assertion as
"a worker process is pygame-free", because a spawned worker *is* a fresh interpreter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from experiments.match import MatchResult, random_opening, standard_error
from experiments.tournament import (
    AgentSpec,
    Pairing,
    PairingOutcome,
    Shard,
    bootstrap_elo,
    elo_difference,
    elo_difference_error,
    fit_elo,
    first_mover_tally,
    merge,
    play_shard,
    resolvable_margin,
    round_robin,
    seat_records,
    shards_for,
)
from core.bitboard import Bitboard
from search.stats import Budget

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
CONFIGS = ROOT / "data" / "configs"


class TimedMatchRefused(ValueError):
    """Raised when a config asks for a wall-clock budget and more than one worker."""


@dataclass(frozen=True)
class TournamentConfig:
    name: str
    agents: tuple[AgentSpec, ...]
    games: int
    seed: int = 2026
    opening_plies: int = 4
    budget: Budget = Budget()
    board: str = "bitboard"
    elo_prior: float = 2.0
    elo_resamples: int = 1000
    note: str = ""

    @classmethod
    def load(cls, path: Path | str) -> "TournamentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        budget = data.get("budget") or {}
        unknown = set(budget) - {"nodes", "time", "depth"}
        if unknown:
            raise ValueError(f"unknown budget axes {sorted(unknown)} in {path}.")
        return cls(
            name=data["name"],
            agents=tuple(AgentSpec.from_json(entry) for entry in data["agents"]),
            games=int(data["games"]),
            seed=int(data.get("seed", 2026)),
            opening_plies=int(data.get("opening_plies", 4)),
            budget=Budget(**budget),
            board=data.get("board", "bitboard"),
            elo_prior=float(data.get("elo_prior", 2.0)),
            elo_resamples=int(data.get("elo_resamples", 1000)),
            note=data.get("note", ""),
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(spec.label for spec in self.agents)

    @property
    def budget_unit(self) -> str:
        """The unit the budget was expressed in, which thesis point P29 requires beside
        every number: an MCTS simulation is not an alpha-beta node, and a clock is not
        either."""
        if self.budget.time is not None:
            return "time"
        if self.budget.nodes is not None:
            return "nodes"
        if self.budget.depth is not None:
            return "depth"
        return "unlimited"

    def pairings(self) -> list[Pairing]:
        return [
            Pairing(a, b, self.games, self.seed, self.opening_plies, self.budget, self.board)
            for a, b in round_robin(self.agents)
        ]


def check_parallelisable(config: TournamentConfig, workers: int) -> None:
    if workers > 1 and config.budget.time is not None:
        raise TimedMatchRefused(
            f"config {config.name!r} has a wall-clock budget ({config.budget}) and asks "
            f"for {workers} workers.\n"
            "A timed match under load measures the load, not the agents. Phase 9 ran the "
            "same 540 games under a clock (44.3%) and under a node budget (48.5%) and "
            "only the second was a controlled experiment.\n"
            "Either express the budget in nodes, or run it with --workers 1."
        )


# ------------------------------------------------------------------------------ the run


@dataclass
class TournamentRun:
    config: TournamentConfig
    results: dict[tuple[str, str], MatchResult]
    workers: int
    shards: int
    seconds: float
    outcomes: list[PairingOutcome] = field(default_factory=list)

    def digest(self) -> str:
        """The artifact the reproducibility gate compares.

        A sha256 over the pairing digests in pairing order, for the reason
        `MatchResult.digest` gives: `hash()` randomises string hashing per process, so it
        would differ between two runs of the same experiment -- and between a parent and
        its own workers.
        """
        parts = [self.results[key].digest() for key in sorted(self.results)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def run(
    config: TournamentConfig,
    workers: int = 1,
    *,
    shards_per_worker: int = 8,
    progress: bool = True,
) -> TournamentRun:
    """Play the whole round robin and return it merged, split by seat, with timings."""
    check_parallelisable(config, workers)
    pairings = config.pairings()
    per_pairing = 1 if workers <= 1 else max(1, (workers * shards_per_worker) // len(pairings) + 1)
    shards = shards_for(pairings, per_pairing)

    started = time.perf_counter()
    if workers <= 1:
        parts = {shard.order: play_shard(shard) for shard in shards}
    else:
        parts = {}
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as pool:
            futures = {pool.submit(play_shard, shard): shard for shard in shards}
            for done in concurrent.futures.as_completed(futures):
                shard = futures[done]
                parts[shard.order] = done.result()
                if progress:
                    print(
                        f"  [{len(parts):>3}/{len(shards)}] "
                        f"{shard.pairing.a.label} vs {shard.pairing.b.label} "
                        f"games {shard.first}-{shard.last}",
                        flush=True,
                    )
    seconds = time.perf_counter() - started

    # Merged in shard order, never in completion order. Everything the digest reports is
    # an integer and integer addition is associative, so the digest would survive either;
    # the seconds would not, and a number that moves with the scheduler is worth avoiding.
    results: dict[tuple[str, str], MatchResult] = {}
    grouped: dict[tuple[str, str], list[MatchResult]] = {}
    for shard in shards:
        grouped.setdefault(shard.pairing.key, []).append(parts[shard.order])
    for key, group in grouped.items():
        results[key] = merge(group)

    outcomes = [
        PairingOutcome(a, b, m.overall.wins, m.overall.draws, m.overall.losses)
        for (a, b), m in results.items()
    ]
    return TournamentRun(config, results, workers, len(shards), seconds, outcomes)


# ----------------------------------------------------------------------------- reporting


def pairing_rows(run: TournamentRun) -> list[dict]:
    """One row per pairing, split by seat.

    `workers` is a column and not a footnote: the score columns are load-independent
    because the budget is in nodes, but `a_ms_per_move` and `b_ms_per_move` were measured
    with that many searches sharing four performance cores, so they are **not** comparable
    to a serial run. Thesis point P10 in its Phase 12 form -- every number reported with
    what produced it, and for a wall clock that now includes how loaded the machine was.
    """
    config = run.config
    rows = []
    for (a, b), match in run.results.items():
        overall = match.overall
        rows.append(
            {
                "a": a,
                "b": b,
                "a_agent": match.agent_a,
                "b_agent": match.agent_b,
                "budget": str(config.budget),
                "budget_unit": config.budget_unit,
                "workers": run.workers,
                "games": match.games,
                "opening_plies": match.opening_plies,
                "a_first_score": round(match.a_first.score, 4),
                "a_first_sigma": round(standard_error(match.a_first), 4),
                "a_second_score": round(match.b_first.score, 4),
                "a_second_sigma": round(standard_error(match.b_first), 4),
                "score": round(overall.score, 4),
                "sigma": round(standard_error(overall), 4),
                "wins": overall.wins,
                "draws": overall.draws,
                "losses": overall.losses,
                "first_mover_score": round(
                    match.first_mover_wins / match.games if match.games else 0.0, 4
                ),
                "mean_plies": round(match.mean_plies, 2),
                "a_nodes_per_move": round(match.nodes_a / match.moves_a, 1) if match.moves_a else "",
                "b_nodes_per_move": round(match.nodes_b / match.moves_b, 1) if match.moves_b else "",
                "a_ms_per_move": round(1000 * match.seconds_a / match.moves_a, 3) if match.moves_a else "",
                "b_ms_per_move": round(1000 * match.seconds_b / match.moves_b, 3) if match.moves_b else "",
                "elo_diff": round(elo_difference(overall.score), 1),
                "elo_diff_sigma": round(elo_difference_error(overall), 1),
                "digest": match.digest(),
            }
        )
    return rows


def elo_rows(run: TournamentRun) -> list[dict]:
    config = run.config
    labels = list(config.labels)
    ratings = fit_elo(run.outcomes, labels, prior=config.elo_prior)
    bands = bootstrap_elo(
        run.outcomes,
        labels,
        prior=config.elo_prior,
        resamples=config.elo_resamples,
        seed=config.seed,
    )
    records = seat_records(run.results, labels)
    rows = []
    for label in sorted(labels, key=lambda name: (-ratings[name], name)):
        record = records[label]
        overall = record.overall
        rows.append(
            {
                "label": label,
                "agent": next(s for s in config.agents if s.label == label).agent_name,
                "budget": str(config.budget),
                "budget_unit": config.budget_unit,
                "games": overall.games,
                "elo": round(ratings[label], 1),
                "elo_low": round(bands[label][0], 1),
                "elo_high": round(bands[label][1], 1),
                "score": round(overall.score, 4),
                "sigma": round(standard_error(overall), 4),
                "score_first": round(record.first.score, 4),
                "sigma_first": round(standard_error(record.first), 4),
                "score_second": round(record.second.score, 4),
                "sigma_second": round(standard_error(record.second), 4),
                "wins": overall.wins,
                "draws": overall.draws,
                "losses": overall.losses,
            }
        )
    return rows


def format_report(run: TournamentRun, elo: Sequence[Mapping]) -> str:
    config = run.config
    games = sum(m.games for m in run.results.values())
    first = first_mover_tally(run.results.values())
    lines = [
        f"tournament {config.name}   {len(config.agents)} agents, "
        f"{len(run.results)} pairings x {config.games} games = {games} games",
        f"budget {config.budget} ({config.budget_unit})   seed {config.seed}   "
        f"opening {config.opening_plies} random plies   board {config.board}",
        f"{run.workers} worker(s), {run.shards} shards, {run.seconds:.1f} s "
        f"({games / run.seconds:.2f} games/s)   digest {run.digest()}",
        f"a pairing of {config.games} games resolves "
        f"{resolvable_margin(config.games):.1f} points at 2 sigma",
        "",
        f"{'agent':<12}{'elo':>8}{'95% band':>18}{'score':>9}{'first':>9}{'second':>9}"
        f"{'W-D-L':>18}",
    ]
    for row in elo:
        band = f"[{row['elo_low']:+.0f}, {row['elo_high']:+.0f}]"
        record = f"{row['wins']}-{row['draws']}-{row['losses']}"
        lines.append(
            f"{row['label']:<12}{row['elo']:>+8.0f}{band:>18}"
            f"{row['score']:>8.1%} {row['score_first']:>8.1%} {row['score_second']:>8.1%}"
            f"{record:>18}"
        )
    lines += [
        "",
        f"{'pairing':<26}{'A score':>10}{'sigma':>8}{'A 1st':>8}{'A 2nd':>8}"
        f"{'elo gap':>10}{'draws':>8}",
    ]
    for row in pairing_rows(run):
        gap = row["elo_diff"]
        shown = f"{gap:+.0f}" if math.isfinite(gap) else ("+inf" if gap > 0 else "-inf")
        name = f"{row['a']} vs {row['b']}"
        lines.append(
            f"{name:<26}{row['score']:>9.1%} "
            f"{row['sigma']:>7.1%}{row['a_first_score']:>8.1%}{row['a_second_score']:>8.1%}"
            f"{shown:>10}{row['draws'] / row['games']:>8.1%}"
        )
    lines += [
        "",
        f"first mover over the whole tournament: {first.score:.1%} "
        f"+- {standard_error(first):.1%} over {first.games} games "
        "(random-vs-random baseline 55.4%)",
    ]
    if config.budget_unit == "time":
        lines.append(
            "\nWARNING: this tournament was budgeted by a wall clock, so it does not "
            "reproduce.\nIts digest is a record of one run, not a contract (P24)."
        )
    return "\n".join(lines)


def write_csv(rows: Sequence[Mapping], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


# ------------------------------------------------------------- the parallelism, measured


def boundary_cost(config: TournamentConfig, workers: int) -> dict:
    """What crossing the process boundary costs, measured rather than assumed.

    The probe is a real shard with an empty range of games, so it pays exactly what a
    working shard pays apart from the games: spawn a fresh interpreter, re-import the
    project, build both agents from their specs, and pickle a `MatchResult` back. No test
    scaffolding is involved -- an empty shard is a legal shard.
    """
    pairing = config.pairings()[0]
    probe = Shard(pairing, 0, 0, 0)
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as pool:
        list(pool.map(play_shard, [probe] * workers))
    elapsed = time.perf_counter() - started
    return {
        "workers": workers,
        "startup_seconds": round(elapsed, 3),
        "shard_bytes": len(pickle.dumps(probe)),
        "result_bytes": len(pickle.dumps(play_shard(probe))),
    }


def speedup_sweep(
    config: TournamentConfig,
    worker_counts: Sequence[int],
    *,
    shards_per_worker: int = 8,
    granularity: Sequence[int] = (),
) -> list[dict]:
    """One variable at a time. Everything else is held fixed, including the seed.

    The experiment doubles as the reproducibility gate: every run must produce the same
    digest, and that is asserted here rather than eyeballed. Eight cores is not eight
    times -- this machine reports eight, but `sysctl` says four performance cores and
    four efficiency cores, and it is a fanless Air. Reporting the bend is the point.

    `granularity` runs the second control: the same worker count with the work cut into
    more or fewer shards. If load imbalance were what is missing from the speedup, this
    is the knob that would show it.
    """
    rows = []
    baseline = None
    digest = None
    games = len(config.pairings()) * config.games

    def record(variable: str, workers: int, per_worker: int) -> None:
        nonlocal baseline, digest
        finished = run(config, workers, shards_per_worker=per_worker, progress=False)
        if digest is None:
            digest, baseline = finished.digest(), finished.seconds
        elif finished.digest() != digest:
            raise AssertionError(
                f"{workers} workers x {per_worker} shards gave digest "
                f"{finished.digest()}, the baseline gave {digest}: not reproducible."
            )
        rows.append(
            {
                "config": config.name,
                "variable": variable,
                "games": games,
                "budget": str(config.budget),
                "budget_unit": config.budget_unit,
                "workers": workers,
                "shards_per_worker": per_worker,
                "shards": finished.shards,
                "seconds": round(finished.seconds, 2),
                "games_per_second": round(games / finished.seconds, 2),
                "speedup": round(baseline / finished.seconds, 3),
                "efficiency": round(baseline / finished.seconds / workers, 3),
                "digest": finished.digest(),
            }
        )
        row = rows[-1]
        print(
            f"{variable:<18}{workers:>2} workers {row['shards']:>3} shards  "
            f"{row['seconds']:>7.2f}s  {row['games_per_second']:>6.2f} games/s  "
            f"speedup {row['speedup']:>5.2f}x  efficiency {row['efficiency']:>5.1%}  "
            f"digest {row['digest']}",
            flush=True,
        )

    for workers in worker_counts:
        record("workers", workers, shards_per_worker)
    for per_worker in granularity:
        record("shards_per_worker", max(worker_counts), per_worker)
    return rows


def depth_reached(config: TournamentConfig) -> list[dict]:
    """How deep each agent actually gets on the tournament's budget.

    The tournament says which agent is stronger; this says *why*, and without it the
    headline of Phase 12 is a table rather than an explanation. An optimisation given a
    fixed number of positions cannot spend them on more positions -- it can only spend
    them on more plies, and a ply is quantised, so a saving that does not cross a ply
    boundary buys nothing at all.

    The sample is the one every cost table in `session.py` uses: the empty board plus four
    random openings at each of 2, 4, 6, 8, 10 and 12 plies, seed 2026.
    """
    boards = [Bitboard()]
    for plies in (2, 4, 6, 8, 10, 12):
        for index in range(4):
            board = Bitboard()
            for column in random_opening(2026 + 100 * plies + index, plies):
                board.play(column)
            boards.append(board)

    rows = []
    for spec in config.agents:
        agent = spec.build()
        depths, nodes, seconds = [], [], []
        for board in boards:
            _, stats = agent.choose_move(board, config.budget)
            depths.append(stats.depth)
            nodes.append(stats.nodes)
            seconds.append(stats.elapsed)
        rows.append(
            {
                "label": spec.label,
                "agent": agent.name,
                "budget": str(config.budget),
                "budget_unit": config.budget_unit,
                "positions": len(boards),
                "mean_depth": round(sum(depths) / len(depths), 2),
                "min_depth": min(depths),
                "max_depth": max(depths),
                "mean_nodes": round(sum(nodes) / len(nodes), 1),
                "mean_ms": round(1000 * sum(seconds) / len(seconds), 2),
            }
        )
        row = rows[-1]
        print(
            f"{row['label']:<10} {row['agent']:<48} depth {row['mean_depth']:>5.2f} "
            f"({row['min_depth']}-{row['max_depth']})  {row['mean_nodes']:>8,.0f} nodes  "
            f"{row['mean_ms']:>7.2f} ms",
            flush=True,
        )
    return rows


# ----------------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven tournament.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shards-per-worker", type=int, default=8)
    parser.add_argument("--speedup", default="", help="comma-separated worker counts")
    parser.add_argument("--granularity", default="", help="comma-separated shard counts")
    parser.add_argument("--check", type=int, default=0, help="worker count to compare with 1")
    parser.add_argument("--depths", action="store_true",
                        help="how deep each agent gets on the config's budget")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    path = Path(args.config)
    if not path.exists():
        path = CONFIGS / args.config
    config = TournamentConfig.load(path)

    if args.check:
        serial = run(config, 1, progress=False)
        parallel = run(config, args.check, shards_per_worker=args.shards_per_worker,
                       progress=False)
        agree = serial.digest() == parallel.digest()
        print(f"1 worker    {serial.digest()}  {serial.seconds:.1f}s")
        print(f"{args.check} workers   {parallel.digest()}  {parallel.seconds:.1f}s")
        print("REPRODUCIBLE" if agree else "DIFFERENT -- the run is not reproducible")
        raise SystemExit(0 if agree else 1)

    if args.depths:
        rows = depth_reached(config)
        if not args.no_write:
            write_csv(rows, RESULTS / f"tournament_{config.name}_depth.csv")
        return

    if args.speedup:
        counts = [int(part) for part in args.speedup.split(",")]
        grain = [int(part) for part in args.granularity.split(",") if part]
        rows = speedup_sweep(
            config, counts, shards_per_worker=args.shards_per_worker, granularity=grain
        )
        overhead = boundary_cost(config, max(counts))
        print(
            f"boundary: pool of {overhead['workers']} up and back in "
            f"{overhead['startup_seconds']}s, shard {overhead['shard_bytes']} B in, "
            f"result {overhead['result_bytes']} B out"
        )
        for row in rows:
            row.update(
                shard_bytes=overhead["shard_bytes"],
                result_bytes=overhead["result_bytes"],
                pool_startup_seconds=overhead["startup_seconds"],
            )
        if not args.no_write:
            write_csv(rows, RESULTS / "parallel_speedup.csv")
        return

    finished = run(config, args.workers, shards_per_worker=args.shards_per_worker)
    # The pairings are written before the Elo is fitted, and that ordering is deliberate:
    # the games are the expensive part (tens of minutes) and the bootstrap is a few
    # minutes of arithmetic that can be redone from this file at any time. Losing the
    # games to a slow fit would be a bad way to find that out.
    if not args.no_write:
        write_csv(pairing_rows(finished), RESULTS / f"tournament_{config.name}.csv")
    table = elo_rows(finished)
    print()
    print(format_report(finished, table))
    if not args.no_write:
        write_csv(table, RESULTS / f"tournament_{config.name}_elo.csv")


if __name__ == "__main__":
    main()
