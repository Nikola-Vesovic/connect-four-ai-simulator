"""Phase 13: the oracle, the conversion, and the three metrics.

Two things are being tested and they are different in kind.

**The conversion and the metrics are arithmetic**, and they are tested exhaustively or on
hand-built values, because a `MoveValues` is just a mapping from a column to an exact score
and needs no board at all. That is deliberate: the definitions of *optimal*, *blunder* and
*value loss* are the easiest thing in the phase to conflate, so they are pinned in one test
that spells out all three on the same position.

**The oracle is a claim about the world**, and the only thing that can check it is the
published benchmark. `test_the_published_scores_and_our_solver_agree_exactly` is the gate
of the phase in miniature: the full run is 2,000 positions, this is a sample sized to stay
inside a test suite, and it fails on outcome *or* mate distance.

The slow tests are the ones that solve. End-Easy is the cheapest set -- about 7 ms for one
full-depth solve and about 50 ms to value every child of a position -- so every sample here
is small and stated.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import random
from unittest.mock import Mock

import pytest

from agents.oracle import (
    DEFAULT_BUDGET,
    DRAW,
    LOSS,
    MAX_PONS,
    PONS_BASE,
    WIN,
    MoveValues,
    Oracle,
    Unsettled,
    from_pons,
    one_ply_further,
    outcome_of,
    to_pons,
)
from agents.alphabeta import NegamaxAgent
from core.bitboard import Bitboard
from core.rules import CELLS
from core.types import Score
from experiments import benchmark, download_testsets
from experiments.benchmark import (
    SETS,
    SOLVABLE,
    AgentMove,
    BadPosition,
    ScoreTask,
    Verdict,
    VerifyTask,
    agent_rows,
    digest_of,
    execute,
    load_set,
    oracle_rows,
    outcome_rows,
    parse_position,
    position_seed,
    score,
    score_position,
    verify_position,
)
from experiments.runner import TimedMatchRefused, TournamentConfig
from experiments.tournament import AgentSpec
from search.stats import Budget

LADDER = TournamentConfig.load("data/configs/ladder.json")


# ------------------------------------------------------------------ the two conventions


def test_the_score_conversion_round_trips_over_every_reachable_distance():
    """`ceil` is not injective, so `from_pons` looks ambiguous and is not.

    The side to move places a stone on plies 1, 3, 5..., so a win from its point of view
    ends on an odd ply and a loss on an even one. Of the two distances that satisfy the
    ceiling, exactly one has the required parity. Checked over every (stones, plies) pair
    a real game can produce rather than over a sample.
    """
    checked = 0
    for stones in range(CELLS):
        for plies in range(1, CELLS - stones + 1):
            winner_stones = (stones + plies + 1) // 2
            if not 4 <= winner_stones <= CELLS // 2:
                continue
            value = Score.win_in(plies) if plies % 2 else Score.loss_in(plies)
            score_ = to_pons(value, stones)
            assert -MAX_PONS <= score_ <= MAX_PONS
            assert from_pons(score_, stones) == value, (stones, plies, value, score_)
            checked += 1
    assert checked == 882  # the measured number, not an invented floor


def test_a_draw_is_zero_on_both_scales():
    assert to_pons(Score.DRAW, 20) == 0
    assert from_pons(0, 20) == Score.DRAW


def test_the_scale_is_bounded_by_the_shortest_possible_win():
    """22 minus four stones: a win cannot be worth more than 18 because the winner has to
    place four discs to make a line."""
    assert PONS_BASE == CELLS // 2 + 1 == 22
    assert MAX_PONS == 18
    assert to_pons(Score.win_in(7), 0) == 18  # seven plies, winner has played four
    with pytest.raises(ValueError):
        from_pons(MAX_PONS + 1, 10)


def test_a_child_value_is_one_ply_further_from_the_root_than_the_child_thinks():
    """The other piece of arithmetic in the phase, and the one that changes no outcome and
    every mate distance -- which is precisely why only the published scores catch it."""
    assert one_ply_further(Score.win_in(3)) == Score.win_in(4)
    assert one_ply_further(Score.loss_in(2)) == Score.loss_in(3)
    assert one_ply_further(Score.DRAW) == Score.DRAW


# ---------------------------------------------------- what the three metrics mean

def hand_built() -> MoveValues:
    """One position with all four cases in it: the best win, a slower win, a loss and a
    draw. No board -- these are definitions, not searches."""
    return MoveValues(
        {0: Score.win_in(5), 1: Score.win_in(11), 2: Score.DRAW, 3: Score.loss_in(4)},
        nodes=0,
        seconds=0.0,
    )


def test_optimal_is_a_set_of_moves_and_never_the_best_move():
    values = MoveValues({0: Score.win_in(5), 1: Score.win_in(5), 2: Score.DRAW}, 0, 0.0)
    assert values.optimal == (0, 1)
    assert values.is_optimal(0) and values.is_optimal(1)
    assert values.discriminating


def test_a_slower_win_is_suboptimal_and_is_not_a_blunder():
    """The single easiest mistake in the phase, pinned. A win in 11 instead of a win in 5
    gives away six plies and no part of the outcome."""
    values = hand_built()
    assert values.value == Score.win_in(5)
    assert values.optimal == (0,)
    assert not values.is_optimal(1)
    assert not values.is_blunder(1)
    assert values.loss(1) == 6  # plies, because both moves are in the same band


def test_a_blunder_is_a_change_of_outcome_and_costs_a_whole_band():
    values = hand_built()
    assert values.is_blunder(2) and values.is_blunder(3)
    assert values.loss(2) == Score.win_in(5)  # about 10,000: one class
    assert values.loss(3) > Score.WIN  # about 20,000: two classes
    assert values.blunder_available


def test_a_lost_position_cannot_contain_a_blunder_at_all():
    """Every move loses, so "optimal" can only mean losing as slowly as possible. A 0.0%
    blunder rate over such a population is a property of the population."""
    values = MoveValues(
        {0: Score.loss_in(10), 1: Score.loss_in(4), 2: Score.loss_in(2)}, 0, 0.0
    )
    assert values.value == Score.loss_in(10)
    assert values.optimal == (0,)
    assert not values.blunder_available
    assert not any(values.is_blunder(move) for move in values.values)
    assert values.discriminating  # it still separates agents, just not on outcome
    assert values.loss(2) == 8


def test_a_position_where_every_move_preserves_the_value_measures_nothing():
    values = MoveValues({0: Score.DRAW, 1: Score.DRAW, 2: Score.DRAW}, 0, 0.0)
    assert not values.discriminating
    assert not values.blunder_available
    assert all(values.is_optimal(move) for move in values.values)


def test_outcome_of_reads_the_three_bands():
    assert outcome_of(Score.win_in(42)) == WIN
    assert outcome_of(Score.loss_in(42)) == LOSS
    assert outcome_of(Score.DRAW) == DRAW


# ------------------------------------------------------------------------- the loader


def test_every_downloaded_test_set_parses_strictly():
    """All 6,000 positions, replayed move by move on a real board. The files arrived over
    plain HTTP, so they are validated rather than believed."""
    total = 0
    for label, _ in SETS:
        positions = load_set(label)
        assert len(positions) == 1000
        total += len(positions)
    assert total == 6000


def test_the_loader_refuses_every_way_a_line_can_be_wrong():
    good = "112233"  # six discs, three in a row for each side, nobody has won
    assert not parse_position("end-easy", 1, f"{good} 3").board().is_terminal()
    for bad in (
        "",  # blank
        good,  # no score
        f"{good} 3 4",  # too many fields
        f"{good} x",  # score is not a number
        f"{good} 19",  # off Pons's scale
        f"{good} -19",
        "8 3",  # not a column
        "1111111 3",  # seventh disc in a six-high column
        "1122334 3",  # the position is already won
        "11223344 3",  # and this one continues past the win
    ):
        with pytest.raises(BadPosition):
            parse_position("end-easy", 1, bad)


def test_a_loaded_position_is_legal_and_still_in_progress():
    for position in load_set("middle-easy", 50):
        board = position.board()
        assert board.moves_played == position.stones == len(position.moves)
        assert not board.is_terminal()
        assert board.legal_moves()


def test_an_unknown_test_set_is_refused():
    with pytest.raises(BadPosition):
        load_set("no-such-set")


def test_missing_benchmark_data_explains_how_to_download_it(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "TESTSETS", tmp_path)
    with pytest.raises(FileNotFoundError, match="python -m experiments.download_testsets"):
        load_set("end-easy")


def test_download_manifest_covers_every_benchmark_file():
    assert set(download_testsets.SHA256) == {filename for _, filename in SETS}
    for checksum in download_testsets.SHA256.values():
        assert len(checksum) == 64
        int(checksum, 16)


@pytest.fixture
def benchmark_download(monkeypatch):
    payload = b"mock benchmark input\n"
    filename = "Test_L3_R1"
    monkeypatch.setattr(
        download_testsets, "SHA256", {filename: hashlib.sha256(payload).hexdigest()}
    )
    opener = Mock(return_value=io.BytesIO(payload))
    monkeypatch.setattr(download_testsets, "urlopen", opener)
    return filename, payload, opener


def test_download_verifies_bytes_before_publishing_the_file(tmp_path, benchmark_download):
    filename, payload, opener = benchmark_download
    target = download_testsets.download_file(filename, tmp_path)
    assert target.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [target]
    opener.assert_called_once_with(download_testsets.BASE_URL + filename, timeout=30)


def test_verified_local_download_does_not_need_the_network(tmp_path, benchmark_download):
    filename, payload, opener = benchmark_download
    (tmp_path / filename).write_bytes(payload)
    assert download_testsets.download_file(filename, tmp_path).read_bytes() == payload
    opener.assert_not_called()


def test_corrupt_download_never_creates_a_benchmark_file(tmp_path, benchmark_download):
    filename, _, opener = benchmark_download
    opener.return_value = io.BytesIO(b"corrupted response")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_testsets.download_file(filename, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_local_download_is_not_silently_overwritten(tmp_path, benchmark_download):
    filename, _, opener = benchmark_download
    target = tmp_path / filename
    target.write_bytes(b"local content")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_testsets.download_file(filename, tmp_path)
    assert target.read_bytes() == b"local content"
    opener.assert_not_called()


def test_download_rejects_an_oversized_response(tmp_path, benchmark_download, monkeypatch):
    filename, _, _ = benchmark_download
    monkeypatch.setattr(download_testsets, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="size limit"):
        download_testsets.download_file(filename, tmp_path)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- the oracle


def test_the_published_scores_and_our_solver_agree_exactly():
    """The gate of the phase, at a size that fits in a test suite.

    It is a two-way validation: the artifact validates our solver and our solver validates
    the artifact. Outcome *and* mate distance -- agreeing only on who wins would leave the
    child-to-parent ply correction untested.
    """
    for label, count in (("end-easy", 40), ("middle-easy", 6)):
        for position in load_set(label, count):
            verdict = verify_position(VerifyTask(position))
            assert verdict.ours == verdict.published, (label, position.index, position.moves)


def test_the_value_of_a_position_is_the_best_of_its_children():
    """Step 2 checked against step 1 on the same positions. This is what would catch the
    off-by-one in `one_ply_further`: the outcome would survive it and the distance would
    not."""
    oracle = Oracle()
    for position in load_set("end-easy", 20):
        board = position.board()
        direct, _ = oracle.solve(board)
        assert oracle.move_values(board).value == direct


def test_the_oracle_refuses_a_value_it_did_not_prove():
    """A heuristic score means the search did not settle the position, and an oracle that
    returned one would be scoring agents against an opinion. Checked by shortening the
    search rather than by trusting that it is long enough."""
    oracle = Oracle()
    oracle._search = NegamaxAgent(2)
    board = load_set("begin-hard", 1)[0].board()
    with pytest.raises(Unsettled):
        oracle.solve(board)


def test_a_position_the_oracle_cannot_settle_is_excluded_and_counted():
    """Begin-Hard is the stated limit of the phase. A position from it is not dropped
    silently: it comes back as a verdict that says so, and every rate is taken over the
    settled subset."""
    position = load_set("begin-hard", 1)[0]
    verdict = verify_position(VerifyTask(position, oracle_nodes=2000))
    assert not verdict.settled
    assert verdict.published == position.score
    rows = oracle_rows([verdict], workers=1, seconds=0.0)
    assert rows[0]["attempted"] == 1
    assert rows[0]["settled"] == 0
    assert rows[0]["unsettled"] == 1


def test_an_oracle_budgeted_by_a_clock_is_refused():
    """It would settle a different set of positions on every run, so the benchmark would
    not be a benchmark. P24, one level further down than the runner's refusal."""
    assert DEFAULT_BUDGET.nodes and DEFAULT_BUDGET.time is None
    with pytest.raises(ValueError):
        Oracle(Budget(time=1.0))


def test_a_proof_does_not_need_an_exhausted_tree():
    """The observation the phase is affordable because of: a value in the win or loss band
    is exact even though the search stopped early, and only a DRAW needs the whole tree.

    Checked against the unbounded answer on positions cheap enough to settle both ways.
    """
    generous = Oracle(Budget(nodes=5_000_000))
    stingy = Oracle(Budget(nodes=20_000))
    proved = 0
    for position in load_set("end-easy", 30):
        board = position.board()
        truth, _ = generous.solve(board)
        try:
            guess, stats = stingy.solve(board)
        except Unsettled:
            continue
        proved += 1
        assert guess == truth
    assert proved >= 25


def test_the_oracle_always_plays_a_value_preserving_move():
    """The end-to-end self-check: the one agent that must score 100% optimal and 0%
    blunders by construction."""
    oracle = Oracle()
    for position in load_set("end-easy", 15):
        board = position.board()
        values = oracle.move_values(board)
        move, _ = oracle.choose_move(board)
        assert values.is_optimal(move)
        assert not values.is_blunder(move)


def test_the_benchmark_sets_contain_all_three_populations():
    """The split this phase reports instead of P1's seat split has to be non-degenerate,
    so it is measured rather than assumed. Floors are the measured numbers."""
    outcomes = [verify_position(VerifyTask(p)).outcome for p in load_set("end-easy", 40)]
    assert outcomes.count(WIN) >= 5
    assert outcomes.count(LOSS) >= 5
    assert outcomes.count(DRAW) >= 5


def test_a_real_lost_position_offers_no_blunder_to_any_agent():
    """The hand-built case above, found in the data rather than invented."""
    oracle = Oracle()
    lost = 0
    for position in load_set("end-easy", 30):
        values = oracle.move_values(position.board())
        if outcome_of(values.value) is not LOSS:
            continue
        lost += 1
        assert not values.blunder_available
        assert not any(values.is_blunder(move) for move in values.values)
    assert lost >= 5


# ------------------------------------------------------------------- the work unit


def test_a_positions_seed_is_a_pure_function_of_its_identity():
    """Several agents carry state and some are stochastic, so a score that depended on
    which worker got which position would not be a score at all."""
    assert position_seed(2026, "end-easy", 7) == position_seed(2026, "end-easy", 7)
    seen = {
        position_seed(2026, label, index)
        for label, _ in SETS
        for index in range(1000)
    }
    assert len(seen) == len(SETS) * 1000
    assert position_seed(2026, "end-easy", 7) != position_seed(2027, "end-easy", 7)


def test_the_digest_does_not_care_which_order_the_verdicts_arrive_in():
    positions = load_set("end-easy", 8)
    tasks = [ScoreTask(p, LADDER.agents[:3], Budget(nodes=500), 2026) for p in positions]
    verdicts = [score_position(task) for task in tasks]
    shuffled = list(verdicts)
    random.Random(7).shuffle(shuffled)
    assert digest_of(verdicts) == digest_of(shuffled)
    assert digest_of(verdicts) != digest_of(verdicts[:-1])


def test_the_benchmark_reproduces_between_one_worker_and_several():
    """The gate of Phase 12 in this phase's shape. One worker means no multiprocessing at
    all, so this compares the previous state of the project against spawned interpreters
    -- and a spawned child is a fresh interpreter, which is also the pygame-free
    assertion."""
    positions = load_set("end-easy", 8)
    tasks = [ScoreTask(p, LADDER.agents, Budget(nodes=400), 2026) for p in positions]
    serial, _ = execute(tasks, score_position, workers=1, progress=False)
    parallel, _ = execute(tasks, score_position, workers=2, progress=False)
    assert digest_of(serial) == digest_of(parallel)


def test_a_wall_clock_budget_is_refused_in_parallel_here_too():
    """P24's rule, and the runner already refuses it for tournaments. A timed search under
    load measures the load."""
    with pytest.raises(TimedMatchRefused):
        score(["end-easy"], {"end-easy": 2}, LADDER.agents, Budget(time=0.01), workers=4)


def test_a_wall_clock_budget_is_allowed_in_one_process():
    verdicts, _ = score(
        ["end-easy"], {"end-easy": 2}, LADDER.agents[:2], Budget(time=0.01),
        workers=1, progress=False,
    )
    assert len(verdicts) == 2


# ------------------------------------------------------------------------ aggregation


def scored(count: int = 10):
    positions = load_set("end-easy", count)
    tasks = [ScoreTask(p, LADDER.agents, Budget(nodes=1000), 2026) for p in positions]
    return [score_position(task) for task in tasks]


def test_every_scored_position_carries_one_move_per_agent():
    verdicts = scored()
    for verdict in verdicts:
        assert [move.label for move in verdict.moves] == [s.label for s in LADDER.agents]
        assert verdict.optimal_count >= 1
        assert verdict.optimal_count <= verdict.legal
        assert verdict.discriminating == (verdict.optimal_count < verdict.legal)


def test_the_rates_are_reported_over_the_discriminating_subset_as_well():
    """P2 applied in advance: a metric a broken agent still passes is not a metric, so the
    number of positions that actually separate two agents is reported beside every rate."""
    verdicts = scored()
    rows = agent_rows(verdicts, LADDER.agents, Budget(nodes=1000), workers=1)
    assert len(rows) == len(LADDER.agents)
    for row in rows:
        assert row["positions"] == len(verdicts)
        assert row["discriminating"] <= row["positions"]
        assert row["blunder_available"] <= row["positions"]
        assert row["blunders"] <= row["blunder_available"]
        assert 0.0 <= row["optimal_rate"] <= 1.0


def test_the_table_is_split_by_whether_the_side_to_move_is_winning_or_losing():
    verdicts = scored(20)
    rows = outcome_rows(verdicts, LADDER.agents, Budget(nodes=1000))
    by_outcome = {row["outcome"] for row in rows}
    assert by_outcome <= {"win", "draw", "loss"}
    for row in rows:
        if row["outcome"] == "loss":
            assert row["blunder_available"] == 0
            assert row["blunders"] == 0
    totals = {
        agent: sum(r["positions"] for r in rows if r["agent"] == agent)
        for agent in {row["agent"] for row in rows}
    }
    assert set(totals.values()) == {len(verdicts)}


def test_the_reach_of_the_oracle_is_recorded_for_every_set():
    """The stated limit, kept in one place so the thesis and the code cannot disagree.

    The distinction the first draft of this table got wrong is the one worth pinning: a set
    can be **provable** and not **scoreable**. Scoring needs the true value of every child,
    which is about seven solves, so Begin-Easy settles 30/30 at 2,000,000 nodes and is still
    not in the scored table.
    """
    from experiments.benchmark import REACH, SCOREABLE

    assert set(REACH) == {label for label, _ in SETS}
    assert set(SCOREABLE) < set(REACH)
    assert SOLVABLE == SCOREABLE


# --------------------------------------------- holes the mutation check found, closed


def test_an_immediate_win_is_valued_as_a_win_in_one_ply():
    """A terminal child never reaches the search, because `Agent.choose_move` refuses a
    finished position, so its value is written by hand -- and a hand-written value is
    exactly the kind no sampled property happens to exercise.

    **Not one of the 6,000 published positions contains an immediate winning move**, which
    is presumably deliberate on Pons's part and means the benchmark data can never reach
    this branch. The fixture is therefore an End-Easy position extended by one ply, kept
    late enough that every other child is cheap to solve.
    """
    board = Bitboard.from_move_string("67635256351344534443614126713657121")
    winning = [
        move
        for move in board.legal_moves()
        if (board.play(move), board.is_terminal() and board.winner() is not None, board.undo())[1]
    ]
    assert winning, "the fixture is supposed to contain a win in one"
    table = Oracle().move_values(board)
    assert table.value == Score.win_in(1)
    for move in winning:
        assert table.values[move] == Score.win_in(1)
        assert table.is_optimal(move)


def test_a_draw_is_accepted_only_when_the_tree_was_exhausted():
    """The asymmetry the bounded oracle rests on: a win or a loss is proved by the band it
    lands in, whatever depth the search stopped at, but a zero is also what a drawn
    heuristic looks like.

    **The witness had to be searched for, not sampled** -- Phase 1's lesson again, that a
    fault needs a position where it can actually fire. A premature zero is rare: scanning
    120 End-Easy positions at four budgets produced exactly one, and it is pinned here.
    The sampled half below is the broader property: any draw that does come back must come
    from a search that reached every empty cell.
    """
    premature = Bitboard.from_move_string("1313327526554131647611374372646")
    with pytest.raises(Unsettled):  # value 0 at depth 3 of 11 -- a heuristic, not a draw
        Oracle(Budget(nodes=100)).solve(premature)

    stingy = Oracle(Budget(nodes=2000))
    refused = draws = 0
    for position in load_set("end-easy", 60):
        board = position.board()
        try:
            value, stats = stingy.solve(board)
        except Unsettled:
            refused += 1
            continue
        if value == Score.DRAW:
            draws += 1
            assert stats.depth >= CELLS - board.moves_played
    assert refused >= 1 and draws >= 1


def _hand_built_verdicts():
    """Three positions that separate the three metrics and the two populations.

    A blundered move in a position that exposed one, an optimal move in a position where
    every move was optimal, and a slower-than-necessary loss in a position where no move
    could change the outcome.
    """
    def verdict(index, *, legal, optimal_count, blunder_available, loss, blunder):
        return Verdict(
            label="end-easy",
            index=index,
            stones=30,
            legal=legal,
            published=0,
            ours=0,
            value=Score.DRAW,
            outcome=DRAW,
            optimal_count=optimal_count,
            discriminating=optimal_count < legal,
            blunder_available=blunder_available,
            oracle_nodes=0,
            oracle_seconds=0.0,
            moves=(AgentMove("x", 3, loss == 0, blunder, loss, 0, 0.0),),
        )

    return [
        verdict(0, legal=3, optimal_count=1, blunder_available=True, loss=9995, blunder=True),
        verdict(1, legal=3, optimal_count=3, blunder_available=False, loss=0, blunder=False),
        verdict(2, legal=3, optimal_count=1, blunder_available=False, loss=4, blunder=False),
    ]


def test_the_three_rates_have_three_different_denominators():
    """Each of these was a surviving mutant: the discriminating rate taken over everything,
    the blunder rate taken over the positions that exposed one, and the mean ply loss taken
    over the blunders as well. All three are true of the same numbers only by accident."""
    rows = agent_rows(
        _hand_built_verdicts(), [AgentSpec("x", "random")], Budget(nodes=1), workers=1
    )
    row = rows[0]
    assert row["positions"] == 3 and row["discriminating"] == 2
    assert row["optimal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["optimal_rate_discriminating"] == pytest.approx(0.0, abs=1e-4)
    assert row["blunder_available"] == 1 and row["blunders"] == 1
    assert row["blunder_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["blunder_rate_exposed"] == pytest.approx(1.0, abs=1e-4)
    assert row["mean_ply_loss"] == pytest.approx(2.0)  # (0 + 4) / 2, the blunder excluded
    assert row["mean_value_loss"] == pytest.approx((9995 + 0 + 4) / 3, abs=0.1)


def test_the_digest_covers_what_every_agent_played():
    """A digest that only fingerprints the ground truth would reproduce between one worker
    and eight for the wrong reason -- the agents are the thing being measured."""
    verdicts = _hand_built_verdicts()
    before = digest_of(verdicts)
    moved = dataclasses.replace(
        verdicts[0], moves=(dataclasses.replace(verdicts[0].moves[0], move=5),)
    )
    assert digest_of([moved] + verdicts[1:]) != before


def test_the_oracle_is_not_a_playable_agent():
    """BUILD_PLAN section 1 reserved this name with "benchmarking only" beside it, and
    this is what enforces it. A full-depth search from an early position is hours, so
    neither the window nor a tournament config may be able to ask for one -- the window
    does not change at all in this phase.
    """
    import session
    from experiments.tournament import BUILDERS

    assert "oracle" not in BUILDERS
    assert "oracle" not in session.ALGORITHMS
    assert len(session.ALGORITHMS) == 5  # Phase 11's five, unchanged
