"""Round robin over many agents, Elo with confidence intervals, and the unit of work that
makes a tournament parallelisable.

Phase 12 contains **no course algorithm**. What it contains is the thing that makes
Phases 13 and 14 affordable: Phase 9's 540-game match pair took about thirty-six minutes
and could not share the CPU, and Phase 11's tournament was sized down to fit thirty-five
minutes of serial running, which is why four of its six switches came back inside 1.2
sigma and could not be separated.

`experiments/match.py` stays the definition of a match. This module orchestrates it.

**Why splitting a match across cores is legitimate here, stated as an argument rather
than assumed.** A game in `play_match` is a pure function of its index: `game_seeds`
fixes the seed pair from the master seed alone, the seating is `index % 2`, and the
opening comes from the pair's own seed. The only state that could leak from one game to
the next is state that outlives a search -- MCTS's reused tree and a persistent
transposition table -- and both are thrown away by `Agent.new_game(seed)`, which
`play_match` calls before every game. That hook was added in Phase 4 for exactly this.
So a shard is not an approximation of the serial run: it is the serial run, restricted.

**Nothing but data crosses the process boundary.** An agent is never pickled and never
shared: what travels is an `AgentSpec`, which is the JSON description of an agent, and
the child builds a fresh one. Sharing one agent object across processes would be wrong
(two searches writing one transposition table) and pickling one would be worse (a live
search tree copied per shard). The return trip is a `MatchResult`, which is counters.

**Reproducibility is independent of the worker count by construction, not by luck.**
Every counter that `MatchResult.fingerprint` reports is an integer, and integer addition
is associative, so merging shards in any order gives the same digest. Wall-clock seconds
are the one thing that does not reproduce, which is why `fingerprint` never included
them -- shards are still merged in index order so that even those are stable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from agents.base import Agent
from agents.greedy_agent import GreedyAgent
from agents.mcts import MctsAgent
from agents.minimax import MinimaxAgent
from agents.random_agent import RandomAgent
from agents.alphabeta import NegamaxAgent
from core.bitboard import Bitboard
from core.board_naive import NaiveBoard
from experiments.match import MatchResult, Tally, play_match, standard_error
from search.deepening import DIRECT, EVEN, ITERATIVE, NAIVE
from search.evaluation import DEFAULT, ZERO
from search.ordering import DYNAMIC, LEFT, RANDOM as RANDOM_ORDER, STATIC
from search.playout import LEFT_FIRST, MEAN_ROOT, REUSE, SOLVER, UCT, RANDOM as RANDOM_PLAYOUT
from search.scouting import DEEP, NO_SCOUT, SCOUT, WIDE
from search.stats import UNLIMITED, Budget
from search.transposition import ARRAY, NO_TABLE, PERSISTENT, TABLE, WRITE_ONLY

BOARDS = {"bitboard": Bitboard, "naive": NaiveBoard}

# Every switchable spec in the project, keyed by the name it already carries in the
# committed CSVs. A config file and a results file therefore speak one vocabulary, and a
# spec cannot be renamed in one place without the other noticing.
EVALUATIONS = {spec.name: spec for spec in (DEFAULT, ZERO)}
ORDERINGS = {spec.name: spec for spec in (STATIC, DYNAMIC, LEFT, RANDOM_ORDER)}
TABLES = {spec.name: spec for spec in (TABLE, NO_TABLE, ARRAY, WRITE_ONLY, PERSISTENT)}
DEEPENINGS = {spec.name: spec for spec in (DIRECT, ITERATIVE, NAIVE, EVEN)}
SCOUTS = {spec.name: spec for spec in (NO_SCOUT, SCOUT, WIDE, DEEP)}
PLAYOUTS = {
    spec.name: spec
    for spec in (UCT, RANDOM_PLAYOUT, REUSE, SOLVER, MEAN_ROOT, LEFT_FIRST)
}


# --------------------------------------------------------------------------- the agents


def _build_random(*, seed: int | None = None) -> Agent:
    return RandomAgent(seed=seed)


def _build_greedy(*, seed: int | None = None) -> Agent:
    return GreedyAgent(seed=seed)


def _build_minimax(*, depth: int = 4, evaluation: str = "full") -> Agent:
    return MinimaxAgent(depth, evaluation=EVALUATIONS[evaluation])


def _build_negamax(
    *,
    depth: int = 4,
    pruning: bool = True,
    evaluation: str = "full",
    ordering: str = "center_first+killers+history",
    transposition: str = "dict",
    deepening: str = "direct",
    scouting: str = "negascout",
    seed: int | None = None,
) -> Agent:
    """Alpha-beta is `pruning: true`, which is the default.

    Phase 3 refused to make alpha-beta a fifth entry in the picker for this reason and
    the config format keeps the same rule: it is a flag on one search, not an algorithm
    of its own, and the agent's own `name` says which it came out as.
    """
    return NegamaxAgent(
        depth,
        pruning=pruning,
        evaluation=EVALUATIONS[evaluation],
        ordering=ORDERINGS[ordering],
        transposition=TABLES[transposition],
        deepening=DEEPENINGS[deepening],
        scouting=SCOUTS[scouting],
        seed=seed,
    )


def _build_mcts(
    *, iterations: int = 1000, playout: str = "uct_greedy", seed: int | None = None
) -> Agent:
    return MctsAgent(iterations, playout=PLAYOUTS[playout], seed=seed)


BUILDERS = {
    "random": _build_random,
    "greedy": _build_greedy,
    "minimax": _build_minimax,
    "negamax": _build_negamax,
    "mcts": _build_mcts,
}


@dataclass(frozen=True)
class AgentSpec:
    """A description of an agent, not an agent.

    This is the thing that crosses the process boundary. It is JSON on both sides, so a
    worker builds its own agent from scratch and no live search state is ever copied.
    """

    label: str
    algorithm: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.algorithm not in BUILDERS:
            raise ValueError(
                f"{self.label}: unknown algorithm {self.algorithm!r}; "
                f"known are {sorted(BUILDERS)}."
            )

    def build(self) -> Agent:
        return BUILDERS[self.algorithm](**dict(self.params))

    @property
    def agent_name(self) -> str:
        """What the agent calls itself -- the string every CSV in this project reports a
        node count against (thesis point P10). Built once and thrown away."""
        return self.build().name

    def to_json(self) -> dict:
        return {"label": self.label, "algorithm": self.algorithm, **dict(self.params)}

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "AgentSpec":
        params = {k: v for k, v in data.items() if k not in ("label", "algorithm")}
        return cls(str(data["label"]), str(data["algorithm"]), params)


# ------------------------------------------------------------------ pairings and shards


@dataclass(frozen=True)
class Pairing:
    a: AgentSpec
    b: AgentSpec
    games: int
    seed: int
    opening_plies: int
    budget: Budget
    board: str = "bitboard"

    @property
    def key(self) -> tuple[str, str]:
        return (self.a.label, self.b.label)


@dataclass(frozen=True)
class Shard:
    """A contiguous block of one pairing's games, and the whole payload of a worker."""

    pairing: Pairing
    order: int  # position in the deterministic merge order
    first: int
    last: int  # exclusive

    @property
    def indices(self) -> range:
        return range(self.first, self.last)


def round_robin(specs: Sequence[AgentSpec]) -> list[tuple[AgentSpec, AgentSpec]]:
    """Every unordered pair once. Seats alternate inside a pairing, so playing a pair
    twice in both orders would only duplicate games, not balance anything."""
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"agent labels must be unique, got {labels}.")
    return [
        (specs[i], specs[j]) for i in range(len(specs)) for j in range(i + 1, len(specs))
    ]


def shard_bounds(games: int, shards: int) -> list[tuple[int, int]]:
    """Split `games` into contiguous blocks whose boundaries fall on **even** indices.

    Games 2k and 2k+1 share an opening and are the two seatings of it, which is what
    makes the seat split a paired comparison rather than two independent samples
    (`play_match`, Phase 5). Splitting between them would not change any number -- each
    game is independent -- but it would mean no shard is a valid sub-match on its own,
    and a shard's own digest would stop being meaningful.
    """
    if games < 0:
        raise ValueError(f"a match cannot have {games} games.")
    pairs = (games + 1) // 2
    shards = max(1, min(shards, pairs))
    bounds = []
    for index in range(shards):
        first = 2 * (index * pairs // shards)
        last = min(2 * ((index + 1) * pairs // shards), games)
        if first < last:
            bounds.append((first, last))
    return bounds


def shards_for(pairings: Sequence[Pairing], per_pairing: int) -> list[Shard]:
    """Flatten a tournament into work units, numbered in the order they must be merged."""
    shards = []
    for pairing in pairings:
        for first, last in shard_bounds(pairing.games, per_pairing):
            shards.append(Shard(pairing, len(shards), first, last))
    return shards


def play_shard(shard: Shard) -> MatchResult:
    """The function a worker process runs. Module level and picklable, which is what
    `spawn` needs: the child re-imports this module and looks the function up by name."""
    pairing = shard.pairing
    return play_match(
        pairing.a.build(),
        pairing.b.build(),
        games=pairing.games,
        seed=pairing.seed,
        board_class=BOARDS[pairing.board],
        budget=pairing.budget,
        opening_plies=pairing.opening_plies,
        indices=shard.indices,
    )


def merge(results: Sequence[MatchResult]) -> MatchResult:
    """Add up shards of one pairing. Order-independent for everything in the digest."""
    if not results:
        raise ValueError("nothing to merge.")
    head = results[0]
    total = MatchResult(head.agent_a, head.agent_b, head.seed, head.opening_plies)
    for part in results:
        if (part.agent_a, part.agent_b, part.seed, part.opening_plies) != (
            total.agent_a,
            total.agent_b,
            total.seed,
            total.opening_plies,
        ):
            raise ValueError("refusing to merge shards of different matches.")
        for here, there in ((total.a_first, part.a_first), (total.b_first, part.b_first)):
            here.wins += there.wins
            here.draws += there.draws
            here.losses += there.losses
        total.plies += part.plies
        total.moves_a += part.moves_a
        total.moves_b += part.moves_b
        total.nodes_a += part.nodes_a
        total.nodes_b += part.nodes_b
        total.seconds_a += part.seconds_a
        total.seconds_b += part.seconds_b
    return total


# ------------------------------------------------------------------------------- the Elo


def expected_score(difference: float) -> float:
    """The Elo curve: what a rating difference predicts the stronger side scores."""
    return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))


def elo_difference(score: float) -> float:
    """The rating gap a score implies. Infinite at 0 and 1, which is the whole reason
    the tournament fit needs a prior."""
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_difference_error(tally: Tally) -> float:
    """One sigma on a single pairing's rating gap, by the delta method.

    `d(elo)/d(score) = 400 / (ln 10 * s * (1 - s))`, and the sigma on the score comes
    from `standard_error`, which uses the real variance of a win/draw/loss outcome rather
    than p(1-p). Undefined at a clean sweep, reported as infinite rather than as a large
    number that looks like a measurement.
    """
    score = tally.score
    if tally.games < 2 or score <= 0.0 or score >= 1.0:
        return math.inf
    slope = 400.0 / (math.log(10.0) * score * (1.0 - score))
    return slope * standard_error(tally)


@dataclass(frozen=True)
class PairingOutcome:
    """One pairing reduced to what the Elo fit needs."""

    a: str
    b: str
    wins: int  # from A's side
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points_a(self) -> float:
        return self.wins + 0.5 * self.draws


def fit_elo(
    outcomes: Sequence[PairingOutcome],
    labels: Sequence[str],
    *,
    prior: float = 2.0,
    tolerance: float = 1e-9,
    max_iterations: int = 100_000,
) -> dict[str, float]:
    """Maximum-likelihood Elo over the whole round robin, not one number per pairing.

    **Why a fit and not a conversion.** A win rate converts to a rating *difference*
    against one opponent. Elo is a single number per agent that has to explain every
    pairing at once, and the only honest way to get it from a round robin is to ask which
    set of ratings makes the observed results most likely. Here that is the Bradley-Terry
    model -- `P(a beats b) = g_a / (g_a + g_b)` with `g = 10^(R/400)` -- solved by the
    standard minorisation-maximisation iteration, which is monotone and needs no
    derivatives. Sweeps are done in the given label order and updated in place, so the
    answer is deterministic to the last digit.

    **Draws are counted as half a point to each side**, the same convention as
    `Tally.score`. Say so in the write-up rather than leave it implied: the model fits the
    *score*, so it cannot express that two agents draw each other often. With strong
    agents on four-ply openings this project sees around 14% draws (P18), so the
    simplification is doing real work and is not free.

    **The prior is not decoration.** An agent that scores 0% against every opponent has no
    finite maximum-likelihood rating -- the likelihood keeps rising as its rating falls.
    `prior` virtual games, scored as half-points against an imaginary opponent at rating
    zero, bound every rating. The cost is a small pull towards the middle: two prior games
    against two hundred real ones is about a 1% compression of the scale.
    """
    if prior <= 0.0:
        raise ValueError(
            "a tournament fit needs a positive prior: an agent that never scores has no "
            "finite rating without one."
        )
    points = {label: 0.0 for label in labels}
    opponents: dict[str, list[tuple[str, int]]] = {label: [] for label in labels}
    for outcome in outcomes:
        if not outcome.games:
            continue
        points[outcome.a] += outcome.points_a
        points[outcome.b] += outcome.games - outcome.points_a
        opponents[outcome.a].append((outcome.b, outcome.games))
        opponents[outcome.b].append((outcome.a, outcome.games))

    gamma = {label: 1.0 for label in labels}
    for _ in range(max_iterations):
        largest = 0.0
        for label in labels:
            won = points[label] + prior / 2.0
            spread = prior / (gamma[label] + 1.0)
            for other, games in opponents[label]:
                spread += games / (gamma[label] + gamma[other])
            updated = won / spread
            largest = max(largest, abs(400.0 * math.log10(updated / gamma[label])))
            gamma[label] = updated
        if largest < tolerance:
            break
    # Elo is only defined up to an additive constant, so the level is a presentation
    # choice. Centring on zero leaves every difference untouched and makes the table
    # readable; the prior's job was regularisation, not the choice of origin.
    ratings = {label: 400.0 * math.log10(value) for label, value in gamma.items()}
    offset = sum(ratings.values()) / len(ratings)
    return {label: value - offset for label, value in ratings.items()}


def _binomial(rng: random.Random, trials: int, probability: float) -> int:
    sampler = getattr(rng, "binomialvariate", None)
    if sampler is not None:
        return sampler(trials, probability)
    if trials < 0 or not 0.0 <= probability <= 1.0:
        raise ValueError("binomial sampling requires trials >= 0 and 0 <= probability <= 1")
    return sum(rng.random() < probability for _ in range(trials))


def bootstrap_elo(
    outcomes: Sequence[PairingOutcome],
    labels: Sequence[str],
    *,
    prior: float = 2.0,
    resamples: int = 1000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Confidence intervals for the fitted ratings, by a seeded parametric bootstrap.

    **What the interval means, stated precisely because a decorative one would not be
    worth printing.** Each pairing's observed win/draw/loss frequencies are taken as the
    true probabilities, the whole tournament is re-drawn from them `resamples` times, and
    the fit is re-run on each. The reported band is the central `confidence` range of
    those refits. It measures *the sampling noise of a tournament this size* -- how much
    the table would move if the same agents played the same number of games again. It is
    not a statement about the agents' strength in any wider setting, and it inherits
    every assumption of the fit, including draws counted as halves.

    Seeded, so repeated runs on the same Python version reproduce exactly.

    **Its one blind spot, written down rather than discovered later.** An agent that has
    not won a single game is resampled from `p = 0`, wins none in every resample, and so
    gets a band far narrower than its real uncertainty: a clean sweep over n games is
    consistent with a true win rate up to about 3/n, and the bootstrap reports none of
    that. What keeps such a rating finite is the prior, not the interval.
    """
    rng = random.Random(f"elo-{seed}")
    samples: dict[str, list[float]] = {label: [] for label in labels}
    for _ in range(resamples):
        drawn = []
        for outcome in outcomes:
            games = outcome.games
            if not games:
                continue
            win_rate = outcome.wins / games
            wins = _binomial(rng, games, win_rate)
            remaining = games - wins
            draws = (
                _binomial(rng, remaining, outcome.draws / (games - outcome.wins))
                if remaining and outcome.wins < games
                else 0
            )
            drawn.append(
                PairingOutcome(outcome.a, outcome.b, wins, draws, remaining - draws)
            )
        for label, rating in fit_elo(drawn, labels, prior=prior).items():
            samples[label].append(rating)
    low = (1.0 - confidence) / 2.0
    return {
        label: (_percentile(sorted(values), low), _percentile(sorted(values), 1.0 - low))
        for label, values in samples.items()
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    below, above = math.floor(position), math.ceil(position)
    if below == above:
        return ordered[below]
    return ordered[below] * (above - position) + ordered[above] * (position - below)


# ------------------------------------------------------------------------- the seat split


@dataclass
class SeatRecord:
    """One agent's results, kept split by who moved first for as long as possible.

    Thesis point P1: an aggregate win rate mixes the agent's skill with the advantage of
    moving first, which two uniformly random players already show is worth 55.4%-44.3%.
    A tournament table that collapses that has stopped measuring strength.
    """

    label: str
    first: Tally = field(default_factory=Tally)
    second: Tally = field(default_factory=Tally)

    @property
    def overall(self) -> Tally:
        return Tally(
            self.first.wins + self.second.wins,
            self.first.draws + self.second.draws,
            self.first.losses + self.second.losses,
        )


def seat_records(
    results: Mapping[tuple[str, str], MatchResult], labels: Sequence[str]
) -> dict[str, SeatRecord]:
    """Split every pairing by seat and file it under both agents.

    `MatchResult` scores both of its tallies from A's side, so B's view of them is the
    mirror. Reading that the wrong way round is the trap Phase 4 named: it silently turns
    a 55.6% first-mover edge into 47.4%.
    """
    records = {label: SeatRecord(label) for label in labels}
    for (a, b), match in results.items():
        for tally, target in (
            (match.a_first, records[a].first),
            (match.b_first, records[a].second),
            (match.b_first.mirrored(), records[b].first),
            (match.a_first.mirrored(), records[b].second),
        ):
            target.wins += tally.wins
            target.draws += tally.draws
            target.losses += tally.losses
    return records


def first_mover_tally(results: Iterable[MatchResult]) -> Tally:
    """Every game in the tournament, scored for whoever moved first."""
    total = Tally()
    for match in results:
        for tally in (match.a_first, match.b_first.mirrored()):
            total.wins += tally.wins
            total.draws += tally.draws
            total.losses += tally.losses
    return total


def resolvable_margin(games: int, sigmas: float = 2.0) -> float:
    """Percentage points a pairing of this size can separate, for a drawless even match.

    The sizing arithmetic this project keeps needing, written down once: the per-game
    score variance of an even drawless pairing is 0.25, so one sigma is `50/sqrt(n)`
    points. Draws only shrink it. P24 sized 540 games at 4.3 points and P30 needed about
    1,100 for 3; both fall straight out of this.
    """
    if games < 1:
        return math.inf
    return sigmas * 50.0 / math.sqrt(games)
