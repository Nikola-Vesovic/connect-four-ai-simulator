"""Phase 3: the thin layer that is genuinely about the window.

Everything that decides anything is tested in `test_session.py` without pygame. What is
left here is what only a window can be wrong about: whether a click lands in the column it
looks like it lands in, whether a frame draws at all, and -- the one that matters -- whether
the window keeps drawing while a search is running.

The whole file skips if pygame is not installed, so the measurement harness and its tests
stay runnable on an interpreter that has never seen it (BUILD_PLAN 2.3, the PyPy escape
hatch).
"""

from __future__ import annotations

import os
import time

import pytest

pygame = pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # a display that needs no display

import config  # noqa: E402
from agents.base import Agent  # noqa: E402
from core.rules import COLS, ROWS  # noqa: E402
from core.types import Player  # noqa: E402
from game import Game  # noqa: E402
from session import Controller, SeatSpec, Session  # noqa: E402

HUMAN = SeatSpec(controller=Controller.HUMAN)


class SlowAgent(Agent):
    """An agent that takes long enough to notice. Stands in for a deep search."""

    name = "slow"

    def __init__(self, seconds: float = 0.4) -> None:
        super().__init__()
        self.seconds = seconds

    def _select_move(self, board, budget, stats):
        time.sleep(self.seconds)
        stats.nodes = 4321
        stats.depth = 9
        return board.legal_moves()[0]


@pytest.fixture
def game():
    pygame.init()
    made = Game(Session((HUMAN, HUMAN)))
    yield made
    pygame.quit()


def click(game: Game, pos) -> None:
    game.on_click(pos)


def test_a_frame_draws_in_every_state(game: Game):
    game.draw()  # empty board

    for col in (0, 1, 0, 1, 0, 1, 0):  # seat 0 wins, so the winning line is drawn too
        game.session.drop(col)
    assert game.session.is_over
    game.draw()

    game.new_game()
    game.session.set_seat(1, SeatSpec(controller=Controller.MACHINE, algorithm="greedy"))
    game.draw()


def test_a_click_lands_in_the_column_it_points_at(game: Game):
    for col in range(COLS):
        x, y, w, h = config.cell_rect(ROWS - 1, col)
        click(game, (x + w // 2, y + h // 2))
        assert game.session.history[-1].column == col
        game.new_game()


def test_the_disc_appears_at_the_bottom_of_the_clicked_column(game: Game):
    x, y, w, h = config.cell_rect(0, 4)
    click(game, (x + w // 2, y + h // 2))
    assert game.session.board.cell(0, 4) == Player.ONE
    assert game.session.board.moves_played == 1

    click(game, (x + w // 2, y + h // 2))
    assert game.session.board.cell(1, 4) == Player.TWO


def test_the_strip_above_the_board_counts_as_the_board(game: Game):
    """Connect Four is played by dropping from the top; clicking there must work."""
    x, _, w, _ = config.cell_rect(ROWS - 1, 2)
    click(game, (x + w // 2, config.BOARD_Y - config.DROP_H // 2))
    assert game.session.board.cell(0, 2) == Player.ONE


def test_clicks_outside_the_board_are_not_moves(game: Game):
    for pos in (
        (config.BOARD_X - 5, config.BOARD_Y + 10),  # left of the board
        (config.PANEL_X + 40, config.BOARD_Y + 10),  # over the panel
        (config.BOARD_X + 20, config.HEADER_H // 2),  # the header
        (config.BOARD_X + 20, config.CONTROLS_Y + 5),  # below the board
    ):
        click(game, pos)
    assert game.session.board.moves_played == 0


def test_the_panel_controls_change_the_session(game: Game):
    """Clicking a widget must change the game's configuration and nothing else."""
    place = config.seat_layout(1)

    controller = next(w for w in game.widgets if getattr(w, "label", None) == "kontrola"
                      and w.rect.x == place.inner_x)
    controller.click(controller.rect.center)
    assert game.session.seats[1].controller is Controller.MACHINE
    assert game.session.agents[1] is not None

    pruning = next(w for w in game.widgets
                   if getattr(w, "label", None) == "alfa-beta odsecanje"
                   and w.rect.x == place.inner_x)
    assert game.session.seats[1].pruning is True
    pruning.click(pruning.rect.center)
    assert game.session.seats[1].pruning is False
    assert game.session.board.moves_played == 0


def test_a_disabled_control_ignores_its_click(game: Game):
    """Seat 0 is a human, so its depth spinner must not do anything."""
    place = config.seat_layout(0)
    depth = next(w for w in game.widgets if getattr(w, "label", None) == "dubina"
                 and w.rect.x == place.inner_x)
    before = game.session.seats[0].depth
    assert depth.click(depth.rect.center) is False
    assert game.session.seats[0].depth == before


def test_the_window_keeps_drawing_while_the_search_runs(game: Game):
    """The anti-freeze property, measured rather than asserted.

    A search that takes 0.6 s must not cost a frame more than a fraction of that. If the
    search ran on the event loop, one frame would take the whole 0.6 s and the window
    would be reported as unresponsive. The threshold is three times under that and three
    times over a plausible scheduling hiccup, because this is a timing test and a machine
    under load must not be able to fail it for the wrong reason.
    """
    search_seconds = 0.6
    game.session.set_seat(0, SeatSpec(controller=Controller.MACHINE, algorithm="random"))
    game.session.agents[0] = SlowAgent(search_seconds)

    frames_while_thinking = 0
    slowest_frame = 0.0
    deadline = time.perf_counter() + 6.0

    while time.perf_counter() < deadline and game.session.board.moves_played == 0:
        started = time.perf_counter()
        game.update()
        game.draw()
        elapsed = time.perf_counter() - started
        if game.thinking:
            frames_while_thinking += 1
            slowest_frame = max(slowest_frame, elapsed)

    assert game.session.board.moves_played == 1, "the search never came back"
    # The robust half: a synchronous search spans exactly one frame, a threaded one at
    # 60 fps spans dozens. The frame-time bound is the direct statement of the same thing,
    # kept generous because a loaded machine must not be able to fail it for the wrong
    # reason -- a search on the event loop would blow it by a factor of two either way.
    assert frames_while_thinking > 10, frames_while_thinking
    assert slowest_frame < search_seconds / 2, f"a frame took {slowest_frame:.3f}s"
    assert game.session.last_stats(0).nodes == 4321


def test_a_new_game_discards_a_search_that_was_still_running(game: Game):
    """Otherwise the answer to the old position would be played on the new one."""
    game.session.set_seat(0, SeatSpec(controller=Controller.MACHINE, algorithm="random"))
    game.session.agents[0] = SlowAgent(0.3)

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and not game.thinking:
        game.update()  # a machine move waits out MACHINE_PAUSE_MS before it starts
    assert game.thinking

    game.new_game()
    assert not game.thinking

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and not game.thinking:
        game.update()

    # The abandoned search's answer must never have been applied.
    assert game.session.board.moves_played == 0
    assert game.session.totals[0].moves == 0
