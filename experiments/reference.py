"""The frozen minimax values every later search has to reproduce.

BUILD_PLAN design principle 3: alpha-beta, move ordering, transposition tables and
negascout change how fast an answer arrives, never what the answer is. That gives one
brutal correctness invariant for Phases 6-10, but only if the answers exist first, in a
form nobody can quietly adjust. This module produces them.

A record is one (position, depth) pair and the value plain minimax returned for it. The
position is a 1-based move string, the standard benchmark format, so the same file can be
read by anything later. The value is the only thing under contract: the chosen move is
stored too, but two searches may legitimately disagree about which of several equally
good moves to play, so it is informational, and so is the node count.

The values depend on the score scale and on the evaluation weights, so the file carries a
signature of both. Change a weight and the signature stops matching, loudly, instead of
leaving a stale file that quietly fails a Phase 6 gate for the wrong reason.

    python -m experiments.reference --generate
    python -m experiments.reference --verify
    python -m experiments.reference --verify --agent alphabeta
    python -m experiments.reference --verify --agent alphabeta --ordering center_first
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

from agents.alphabeta import NegamaxAgent
from agents.base import Agent
from agents.minimax import MinimaxAgent
from core.bitboard import Bitboard
from core.types import Score
from search import evaluation, ordering
from search.evaluation import DEFAULT

PATH = Path(__file__).resolve().parent.parent / "data" / "testsets" / "minimax_reference.json"

# Which search to replay the file with. The value is under contract for all of them; the
# node count only for the two that have no cutoffs, which is why it is carried alongside.
SEARCHES: dict[str, tuple[Callable[..., Agent], bool]] = {
    "minimax": (MinimaxAgent, True),
    "negamax": (functools.partial(NegamaxAgent, pruning=False), True),
    "alphabeta": (functools.partial(NegamaxAgent, pruning=True), False),
}

# Every named ordering has to reproduce every value. Phase 7 added the flag so that the
# Phase 6 baseline stays replayable rather than becoming an old commit.
ORDERINGS: dict[str, ordering.Ordering] = {
    spec.name: spec
    for spec in (ordering.DYNAMIC, ordering.STATIC, ordering.LEFT, ordering.RANDOM)
}

# 200 positions get the cheap depths; the expensive ones go to a prefix of the same list,
# so a deeper entry always has the shallower entries for the same position beside it.
DEPTH_PLAN: tuple[tuple[int, int], ...] = (
    (1, 200),
    (2, 200),
    (3, 200),
    (4, 200),
    (5, 200),
    (6, 40),
    (7, 10),
)


def signature() -> dict:
    """Everything a stored value depends on. If this changes, the file is stale."""
    return {
        "win": Score.WIN,
        "max_mate_distance": Score.MAX_MATE_DISTANCE,
        "evaluation": DEFAULT.name,
        "three": evaluation.THREE,
        "two": evaluation.TWO,
        "center": evaluation.CENTER,
        "parity": evaluation.PARITY,
    }


def sample_positions(count: int, seed: int, max_plies: int = 34) -> list[str]:
    """Distinct non-terminal positions spread across the stages of a game.

    The empty board is always first. The rest are random playouts cut at a random ply,
    which is the same generator the property tests use -- positions a real game would
    reach are the ones a search bug has to be caught in.
    """
    rng = random.Random(seed)
    seen: dict[str, None] = {"": None}
    while len(seen) < count:
        board = Bitboard()
        for _ in range(rng.randrange(1, max_plies)):
            if board.is_terminal():
                break
            board.play(rng.choice(board.legal_moves()))
        if board.is_terminal():
            continue
        seen.setdefault("".join(str(col + 1) for col in board.history), None)
    return list(seen)


def generate(seed: int = 2026, verbose: bool = True) -> dict:
    positions = sample_positions(max(count for _, count in DEPTH_PLAN), seed)
    records = []
    started = time.perf_counter()

    for depth, count in DEPTH_PLAN:
        agent = MinimaxAgent(depth, evaluation=DEFAULT)
        level = time.perf_counter()
        for moves in positions[:count]:
            board = Bitboard.from_move_string(moves)
            move, stats = agent.choose_move(board)
            records.append(
                {
                    "moves": moves,
                    "plies": len(moves),
                    "depth": depth,
                    "value": stats.value,
                    "move": move,
                    "nodes": stats.nodes,
                    "leaves": stats.leaves,
                }
            )
        if verbose:
            print(
                f"depth {depth}: {count} positions, "
                f"{time.perf_counter() - level:6.1f}s",
                flush=True,
            )

    data = {
        "metadata": {
            "generated_by": "experiments.reference, plain depth-limited minimax",
            "seed": seed,
            "positions": len(positions),
            "records": len(records),
            "depth_plan": [list(pair) for pair in DEPTH_PLAN],
            "signature": signature(),
            "seconds": round(time.perf_counter() - started, 1),
        },
        "records": records,
    }
    return data


def load(path: Path = PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(data: dict, path: Path = PATH) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
        handle.write("\n")


def verify(
    data: dict,
    *,
    agent_factory: Callable[[int], Agent] = MinimaxAgent,
    max_depth: int | None = None,
    check_nodes: bool = False,
) -> tuple[int, list[dict]]:
    """Replay the file with a search and compare. Returns (checked, disagreements).

    Only the **value** is compared by default, because only the value is under contract:
    alpha-beta visits a different set of nodes on purpose, and two correct searches may
    prefer different equally-good moves. `check_nodes` adds the node count, which is worth
    having when replaying with plain minimax itself -- it has no cutoffs, so its count is
    deterministic and makes a sharper regression net.

    `agent_factory` takes a depth and returns an agent, so Phase 6 onwards can point this
    at negamax, alpha-beta or negascout without touching anything here.
    """
    if data["metadata"]["signature"] != signature():
        raise ValueError(
            "The reference set was generated with a different score scale or different "
            "evaluation weights. Regenerate it, and re-run the Phase 6 gate against the "
            "new values -- do not compare across signatures."
        )
    agents: dict[int, Agent] = {}
    checked = 0
    wrong = []
    for record in data["records"]:
        if max_depth is not None and record["depth"] > max_depth:
            continue
        agent = agents.setdefault(record["depth"], agent_factory(record["depth"]))
        _, stats = agent.choose_move(Bitboard.from_move_string(record["moves"]))
        checked += 1
        if stats.value != record["value"] or (check_nodes and stats.nodes != record["nodes"]):
            wrong.append({**record, "got_value": stats.value, "got_nodes": stats.nodes})
    return checked, wrong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--agent", default="minimax", choices=sorted(SEARCHES))
    parser.add_argument("--ordering", default=None, choices=sorted(ORDERINGS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    if args.generate:
        data = generate(seed=args.seed)
        save(data)
        meta = data["metadata"]
        print(
            f"wrote {PATH} -- {meta['records']} records over {meta['positions']} "
            f"positions in {meta['seconds']}s"
        )
    if args.verify:
        data = load()
        factory, check_nodes = SEARCHES[args.agent]
        if args.ordering is not None:
            if args.agent == "minimax":
                parser.error("minimax has no cutoffs, so it has no use for an ordering.")
            factory = functools.partial(
                factory, ordering=ORDERINGS[args.ordering], seed=args.seed
            )
        started = time.perf_counter()
        checked, wrong = verify(
            data,
            agent_factory=factory,
            max_depth=args.max_depth,
            check_nodes=check_nodes,
        )
        contract = "value and nodes" if check_nodes else "value only"
        label = args.agent if args.ordering is None else f"{args.agent}/{args.ordering}"
        print(
            f"{label}: checked {checked} records ({contract}), "
            f"{len(wrong)} disagreements in {time.perf_counter() - started:.0f}s"
        )
        for row in wrong[:10]:
            print(" ", row)


if __name__ == "__main__":
    main()
