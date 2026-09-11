"""Phase 4: the baseline agents, and the seam they plug into.

Three things are under test here. The `Agent` wrapper, which is the interface every later
agent inherits and must therefore be hard to misuse. The two baseline policies, checked as
properties over thousands of random positions rather than a handful of hand-picked ones.
And the match runner, which has to be reproducible before any number it prints is worth
quoting.

The oracle for "is there a win here" is `NaiveBoard.scan_for_winner`, the independent
full-board scan from Phase 1 -- not the incremental detector the agent itself sees.
"""

from __future__ import annotations

import random

import pytest

from agents.base import Agent
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from core.board_naive import NaiveBoard
from core.types import GameResult, Move
from experiments.match import Tally, game_seeds, play_game, play_match
from search.ordering import CENTER_FIRST
from search.stats import UNLIMITED, Budget, SearchStats

# --------------------------------------------------------------------------- helpers


def random_position(rng: random.Random, max_plies: int = 30) -> NaiveBoard:
    board = NaiveBoard()
    for _ in range(rng.randrange(max_plies)):
        if board.is_terminal():
            break
        board.play(rng.choice(board.legal_moves()))
    return board


def winning_columns(board: NaiveBoard) -> list[Move]:
    """Columns that win at once for the player to move, per the independent scan."""
    me = board.player_to_move
    found = []
    for col in board.legal_moves():
        board.play(col)
        if board.scan_for_winner() is me:
            found.append(col)
        board.undo()
    return found


def safe_columns(board: NaiveBoard) -> list[Move]:
    """Columns after which the opponent cannot win with their very next disc."""
    safe = []
    for col in board.legal_moves():
        board.play(col)
        if board.is_terminal() or not winning_columns(board):
            safe.append(col)
        board.undo()
    return safe


class RogueAgent(Agent):
    """Every way an agent can misbehave at the seam, switchable."""

    name = "rogue"
    _FIRST_LEGAL = object()  # distinct from None, which is itself an illegal move to test

    def __init__(self, *, move=_FIRST_LEGAL, leave_moves: int = 0) -> None:
        super().__init__()
        self._move = move
        self._leave_moves = leave_moves

    def _select_move(self, board, budget, stats):
        for _ in range(self._leave_moves):
            board.play(board.legal_moves()[0])
        if self._move is RogueAgent._FIRST_LEGAL:
            return board.legal_moves()[0]
        return self._move


# --------------------------------------------------------------------------- the seam


def test_an_agent_will_not_move_in_a_finished_position():
    board = NaiveBoard.from_move_string("1122334")
    assert board.is_terminal()
    with pytest.raises(ValueError, match="game is over"):
        RandomAgent().choose_move(board)


@pytest.mark.parametrize("move", [7, -1, None, 0])
def test_an_illegal_column_is_rejected(move):
    board = NaiveBoard.from_moves([0] * 6)  # column 0 is now full
    with pytest.raises(ValueError, match="not legal"):
        RogueAgent(move=move).choose_move(board)


def test_an_unbalanced_search_is_caught():
    """A play() without a matching undo() is the classic bug of the phases ahead."""
    with pytest.raises(RuntimeError, match="no matching undo"):
        RogueAgent(leave_moves=3).choose_move(NaiveBoard())


def test_the_callers_board_survives_a_broken_agent():
    """Proof that the agent is handed a copy: the caller's board is untouched even when
    the agent wrecks the one it was given."""
    board = NaiveBoard.from_move_string("4455")
    before = str(board)
    with pytest.raises(RuntimeError):
        RogueAgent(leave_moves=5).choose_move(board)
    assert board.moves_played == 4
    assert str(board) == before


def test_the_wrapper_fills_in_what_the_agent_must_not():
    move, stats = RandomAgent(seed=1).choose_move(NaiveBoard())
    assert stats.move == move
    assert stats.elapsed > 0


def test_budget_describes_itself():
    assert str(UNLIMITED) == "unlimited"
    assert str(Budget(nodes=1000)) == "1000 nodes"
    assert str(Budget(nodes=10, time=0.5, depth=7)) == "10 nodes, 0.5s, depth 7"


# --------------------------------------------------------------------------- random


def test_random_agent_plays_only_legal_moves(board_class):
    agent = RandomAgent(seed=7)
    for _ in range(50):
        board = board_class()
        while not board.is_terminal():
            move, _ = agent.choose_move(board)
            assert move in board.legal_moves()
            board.play(move)


def test_random_agent_does_no_lookahead():
    _, stats = RandomAgent(seed=3).choose_move(NaiveBoard())
    assert stats.nodes == 0
    assert stats.depth == 0


def test_the_same_seed_replays_the_same_game():
    def game(seed: int) -> list[Move]:
        agent = RandomAgent()
        agent.new_game(seed)
        board = NaiveBoard()
        while not board.is_terminal():
            move, _ = agent.choose_move(board)
            board.play(move)
        return list(board.history)

    assert game(2026) == game(2026)
    assert game(2026) != game(1)


# --------------------------------------------------------------------------- greedy


def test_greedy_always_takes_a_win_it_can_see():
    rng = random.Random(11)
    agent = GreedyAgent()
    checked = 0
    for _ in range(2_000):
        board = random_position(rng)
        if board.is_terminal():
            continue
        wins = winning_columns(board)
        if not wins:
            continue
        move, _ = agent.choose_move(board)
        assert move in wins, f"missed the win in {wins}\n{board}"
        checked += 1
    assert checked > 200, f"only {checked} positions actually had a win to take"


def test_greedy_never_walks_into_a_loss_it_could_avoid():
    rng = random.Random(12)
    agent = GreedyAgent()
    checked = 0
    for _ in range(2_500):
        board = random_position(rng)
        if board.is_terminal() or winning_columns(board):
            continue  # rule 1 outranks rule 2; that case is the test above
        safe = safe_columns(board)
        if not safe or len(safe) == len(board.legal_moves()):
            continue  # nothing to get wrong unless some column really is a gift
        move, _ = agent.choose_move(board)
        assert move in safe, f"handed over the game; safe were {safe}\n{board}"
        checked += 1
    assert checked > 200, f"only {checked} positions had an avoidable loss"


def test_greedy_plays_the_centre_and_breaks_ties_to_its_left():
    assert CENTER_FIRST == (3, 2, 4, 1, 5, 0, 6)
    move, _ = GreedyAgent().choose_move(NaiveBoard())
    assert move == 3

    # Column 3 alternates X O X O X O: full, no line, and no threat anywhere on the board.
    # Columns 2 and 4 are equally central, so the tie-break has to pick 2.
    quiet = NaiveBoard.from_moves([3] * 6)
    assert not quiet.can_play(3) and quiet.winner() is None
    move, _ = GreedyAgent().choose_move(quiet)
    assert move == 2


def test_greedy_is_deterministic():
    rng = random.Random(13)
    for _ in range(200):
        board = random_position(rng)
        if board.is_terminal():
            continue
        first, _ = GreedyAgent(seed=1).choose_move(board)
        second, _ = GreedyAgent(seed=99).choose_move(board)
        assert first == second


def test_greedy_reports_what_it_looked_at():
    move, stats = GreedyAgent().choose_move(NaiveBoard())
    assert stats.depth == 2
    # Seven columns priced for an immediate win, then column 3 plus its seven replies.
    assert stats.nodes == 15
    assert stats.value is None  # no score scale exists before Phase 5


# --------------------------------------------------------------------------- the harness


def test_a_game_record_reports_both_seats():
    """ONE_WINS in a GameRecord means whoever moved first won, whichever agent that was."""
    record = play_game(GreedyAgent(), RandomAgent(seed=5), board_class=NaiveBoard)
    assert 7 <= record.plies <= 42
    assert record.result in (GameResult.ONE_WINS, GameResult.TWO_WINS, GameResult.DRAW)
    assert record.first_nodes > 0 and record.second_nodes == 0


def test_a_match_reproduces_from_its_seed():
    def run() -> tuple:
        return play_match(RandomAgent(), GreedyAgent(), games=40, seed=4242).fingerprint()

    assert run() == run()


def test_a_different_seed_gives_a_different_match():
    first = play_match(RandomAgent(), GreedyAgent(), games=40, seed=1).fingerprint()
    second = play_match(RandomAgent(), GreedyAgent(), games=40, seed=2).fingerprint()
    assert first != second


def test_the_seats_are_shared_evenly():
    match = play_match(RandomAgent(), RandomAgent(), games=40, seed=8)
    assert match.a_first.games == match.b_first.games == 20


def test_random_against_random_reproduces_the_phase_2_baseline():
    """Phase 2 measured 55.4% / 44.3% / 0.26% over 50,000 games with a single RNG driving
    both sides. A different harness with two seeded agents has to land in the same place."""
    match = play_match(RandomAgent(), RandomAgent(), games=1_000, seed=2026)
    assert 0.51 < match.first_mover_wins / match.games < 0.60
    assert match.overall.draws / match.games < 0.02
    assert 19 < match.mean_plies < 24


def test_greedy_beats_random_decisively():
    """The Phase 4 gate. Both seats, because the first-mover edge is worth 5 points."""
    match = play_match(GreedyAgent(), RandomAgent(), games=2_000, seed=2026)
    assert match.a_first.score > 0.95
    assert match.b_first.score > 0.90


def test_greedy_never_loses_a_game_it_could_have_saved():
    """The end-to-end form of the blocking rule, and the sharpest property in the phase.

    Greedy answers every single immediate threat, so the only way it can lose is from a
    position where *every* legal column hands the opponent a win. Any other loss means the
    safety rule leaks -- something no win rate would reveal, since greedy scores 98.6%
    either way.
    """
    rng = random.Random(2026)
    greedy, opponent = GreedyAgent(), RandomAgent()
    losses = 0

    for index in range(500):
        opponent.new_game(rng.randrange(2**63))
        greedy_first = index % 2 == 0
        board = NaiveBoard()
        before_greedys_last_move = None

        while not board.is_terminal():
            if (board.moves_played % 2 == 0) == greedy_first:
                before_greedys_last_move = board.copy()
                move, _ = greedy.choose_move(board)
            else:
                move, _ = opponent.choose_move(board)
            board.play(move)

        if board.result() is GameResult.DRAW:
            continue
        if (board.result() is GameResult.ONE_WINS) == greedy_first:
            continue

        losses += 1
        assert not safe_columns(before_greedys_last_move), (
            f"lost from a position that still had safe columns\n{before_greedys_last_move}"
        )

    assert losses > 0, "greedy lost nothing in this sample, so the property proved nothing"


def test_match_aggregates_do_not_depend_on_the_order_games_are_played_in():
    """Each game is fixed by its own seed pair, so the totals must survive a reshuffle.

    This is the guard for state leaking between games. It costs nothing today -- both
    agents here are stateless between games -- and it is exactly what breaks first when
    Phase 11 lets MCTS keep its tree.
    """
    games, seed = 200, 99
    in_order = play_match(RandomAgent(), RandomAgent(), games=games, seed=seed)

    shuffled = list(enumerate(game_seeds(seed, games)))
    random.Random(5).shuffle(shuffled)
    agent_a, agent_b = RandomAgent(), RandomAgent()
    tally, plies = Tally(), 0
    for index, (seed_a, seed_b) in shuffled:
        agent_a.new_game(seed_a)
        agent_b.new_game(seed_b)
        a_first = index % 2 == 0
        first, second = (agent_a, agent_b) if a_first else (agent_b, agent_a)
        game = play_game(first, second)
        plies += game.plies
        if game.result is GameResult.DRAW:
            tally.draws += 1
        elif (game.result is GameResult.ONE_WINS) == a_first:
            tally.wins += 1
        else:
            tally.losses += 1

    assert (tally.wins, tally.draws, tally.losses, plies) == (
        in_order.overall.wins,
        in_order.overall.draws,
        in_order.overall.losses,
        in_order.plies,
    )
