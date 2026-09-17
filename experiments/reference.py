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
    python -m experiments.reference --verify --agent alphabeta --table no_table
    python -m experiments.reference --verify --agent alphabeta --scout no_scout
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
from search import deepening, evaluation, ordering, scouting, transposition
from search.evaluation import DEFAULT
from search.transposition import NO_TABLE

PATH = Path(__file__).resolve().parent.parent / "data" / "testsets" / "minimax_reference.json"

# Which search to replay the file with. The value is under contract for all of them; the
# node count only for the two that have no cutoffs, which is why it is carried alongside.
#
# `negamax` pins the transposition table OFF. Its contract includes the node count, and
# from Phase 8 the table removes nodes even with pruning switched off -- that is exactly
# what a table does, and it is measured in `experiments.scaling`, but it would destroy the
# leaf-for-leaf comparison this entry exists for. Passing `--table` explicitly overrides
# it and drops back to comparing values only.
SEARCHES: dict[str, tuple[Callable[..., Agent], bool]] = {
    "minimax": (MinimaxAgent, True),
    "negamax": (functools.partial(NegamaxAgent, pruning=False, transposition=NO_TABLE), True),
    "alphabeta": (functools.partial(NegamaxAgent, pruning=True), False),
}

# Every named ordering has to reproduce every value. Phase 7 added the flag so that the
# Phase 6 baseline stays replayable rather than becoming an old commit.
ORDERINGS: dict[str, ordering.Ordering] = {
    spec.name: spec
    for spec in (ordering.DYNAMIC, ordering.STATIC, ordering.LEFT, ordering.RANDOM)
}

# Same idea for Phase 8. `no_table` is Phase 7's search, and every one of these has to
# come back with the same 1,050 values as the others.
TABLES: dict[str, transposition.Transposition] = {
    spec.name: spec
    for spec in (
        transposition.TABLE,
        transposition.NO_TABLE,
        transposition.ARRAY,
        transposition.WRITE_ONLY,
        transposition.PERSISTENT,
    )
}

# Same idea for Phase 9, and here the flag earns its keep twice over. Iterative deepening
# is the first setting in this project that may return a different one of several equally
# good moves, so the value contract is the *only* thing left holding it -- and that is
# what this file has enforced since Phase 5. `even` answers from depth d-1 when d is odd,
# so it is deliberately not offered here: it would be checked against a question it
# declined to answer. `experiments.scaling --deepening` compares it at the depth it
# actually answered from.
DEEPENINGS: dict[str, deepening.Deepening] = {
    spec.name: spec for spec in (deepening.DIRECT, deepening.NAIVE, deepening.ITERATIVE)
}

# Phase 10. Every one of these has to return the same 1,050 values, and the pair
# `negascout`/`negascout+wide` is the one that matters most: the re-search window is the
# classic bug of this algorithm, and the two conventions differ in nodes and in nothing
# else. `no_scout` is Phase 9's search, kept replayable for the same reason `no_table`
# and `center_first` were.
SCOUTS: dict[str, scouting.Scout] = {
    spec.name: spec
    for spec in (scouting.SCOUT, scouting.NO_SCOUT, scouting.WIDE, scouting.DEEP)
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
    parser.add_argument("--table", default=None, choices=sorted(TABLES))
    parser.add_argument("--deepening", default=None, choices=sorted(DEEPENINGS))
    parser.add_argument("--scout", default=None, choices=sorted(SCOUTS))
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
        if args.table is not None:
            if args.agent == "minimax":
                parser.error("minimax is the baseline the table is measured against.")
            factory = functools.partial(factory, transposition=TABLES[args.table])
            # A table changes which nodes are visited, so only the value stays comparable.
            check_nodes = check_nodes and not TABLES[args.table].enabled
        if args.deepening is not None:
            if args.agent == "minimax":
                parser.error("minimax searches one fixed depth; it has nothing to iterate.")
            spec = DEEPENINGS[args.deepening]
            factory = functools.partial(factory, deepening=spec)
            # Iterating searches depths 1..d, so the node count is a different number by
            # construction. The value is not, and that is the whole gate.
            check_nodes = check_nodes and not spec.enabled
        if args.scout is not None:
            if args.agent == "minimax":
                parser.error("minimax has no window to narrow; a null window needs alpha-beta.")
            factory = functools.partial(factory, scouting=SCOUTS[args.scout])
            # Deliberately NOT dropped to values-only. The only search here that carries a
            # node-count contract is `negamax`, which is pinned to `pruning=False`, and a
            # null window around an alpha that is never raised is meaningless -- so the
            # agent ignores this setting entirely there. Leaving the node comparison in
            # turns that sentence into part of the gate.
        started = time.perf_counter()
        checked, wrong = verify(
            data,
            agent_factory=factory,
            max_depth=args.max_depth,
            check_nodes=check_nodes,
        )
        contract = "value and nodes" if check_nodes else "value only"
        label = args.agent
        if args.ordering is not None:
            label += f"/{args.ordering}"
        if args.table is not None:
            label += f"/{args.table}"
        if args.deepening is not None:
            label += f"/{args.deepening}"
        if args.scout is not None:
            label += f"/{args.scout}"
        print(
            f"{label}: checked {checked} records ({contract}), "
            f"{len(wrong)} disagreements in {time.perf_counter() - started:.0f}s"
        )
        for row in wrong[:10]:
            print(" ", row)


if __name__ == "__main__":
    main()
