"""The window: the event loop, the control panel, and the thread the search runs on.

Mirrors the departmental framework's `game.py` -- the object that owns the screen, the
clock and the renderer, and runs `events / update / draw` until the user quits. What it
does *not* own is the game: every question about legality, turns and terminal positions
goes to `Session`, which has no pygame in it and is tested without one.

**Why the search runs on a worker thread.** A move is not free. Measured over sixteen real
positions with the full evaluation, the worst single move this interface permits costs
20.9 s (plain minimax at its capped depth of 7) or 30.9 s (alpha-beta at 9). Called
straight from the event loop, the window would stop repainting and stop answering the
close button for that whole time, which on macOS is reported as "application not
responding" -- indistinguishable from a crash, and exactly the thing the mentor must not
be the first to find.

So `Session.plan` -- which only reads -- runs on a daemon thread, and only the main thread
ever calls `Session.commit`. The window keeps repainting at 60 fps with a live elapsed
counter, so a long search *looks* like a long search. Three details make that safe:

* the agent is handed a private copy of the board by `Agent.choose_move` (Phase 4), so
  the worker cannot touch the game even in principle;
* results carry a generation number, so a search that was still running when the user
  started a new game has its answer discarded rather than applied to the wrong position;
* the settings are locked while a search is running, so what finishes is what was asked
  for.

An abandoned search keeps a core busy until it finishes on its own. That is bounded by the
same measurement the depth caps come from: at most about 31 seconds.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import replace

import pygame

import config
from core.rules import COLS, ROWS
from core.types import EMPTY
from gui.renderer import Renderer
from gui.sprites import DiscSprite
from gui.widgets import Button, Checkbox, Spinner
from session import ALGORITHM_KEYS, ALGORITHMS, MODES, TIME_BUDGETS, Controller, Session

CONTROLLER_LABELS = {Controller.HUMAN: "čovek", Controller.MACHINE: "mašina"}

HELP = "Klik na kolonu = potez   •   Razmak = pauza   •   N = nova igra   •   Esc = izlaz"


class Game:
    def __init__(self, session: Session | None = None) -> None:
        pygame.display.set_caption(config.CAPTION)
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        config.init_fonts()

        self.clock = pygame.time.Clock()
        self.session = session if session is not None else Session()
        self.renderer = Renderer(self.screen)
        self.widgets = self._build_widgets()

        self.running = True
        self.paused = False
        self.hover_col: int | None = None
        self.falling: DiscSprite | None = None
        self.notice: str | None = None

        self._inbox: queue.Queue = queue.Queue()
        self._generation = 0
        self._pending: int | None = None
        self._think_started = 0.0
        # Same rule at the start of the process as after any move: a machine move never
        # begins within MACHINE_PAUSE_MS of the board settling.
        self._settled_at = pygame.time.get_ticks()

    # ---------------------------------------------------------------------- the panel

    def _seat_field(self, index: int, **changes) -> None:
        self.session.set_seat(index, replace(self.session.seats[index], **changes))

    def _build_widgets(self) -> list:
        widgets: list = []
        for index in (0, 1):
            widgets.extend(self._seat_widgets(index))
        widgets.extend(self._buttons())
        return widgets

    def _seat_widgets(self, index: int) -> list:
        place = config.seat_layout(index)
        x, width = place.inner_x, place.width
        seats = self.session.seats

        def row(y: int, height: int) -> pygame.Rect:
            return pygame.Rect(x, y, width, height)

        def machine() -> bool:
            return not self.thinking and seats[index].controller is Controller.MACHINE

        def uses(attribute: str):
            return lambda: machine() and getattr(seats[index].spec, attribute)

        def switch(field: str, label: str, y: int, enabled):
            return Checkbox(
                row(y, config.CHECK_H), label,
                lambda: getattr(seats[index], field),
                lambda value: self._seat_field(index, **{field: value}),
                enabled,
            )

        return [
            Spinner(
                row(place.spinner_y[0], config.ROW_H), "kontrola",
                lambda: list(Controller),
                lambda: seats[index].controller,
                lambda value: self._seat_field(index, controller=value),
                lambda value: CONTROLLER_LABELS[value],
                lambda: not self.thinking,
            ),
            Spinner(
                row(place.spinner_y[1], config.ROW_H), "algoritam",
                lambda: list(ALGORITHM_KEYS),
                lambda: seats[index].algorithm,
                lambda value: self._seat_field(index, algorithm=value),
                lambda value: ALGORITHMS[value].label,
                machine,
            ),
            Spinner(
                row(place.spinner_y[2], config.ROW_H), "dubina",
                lambda: list(range(1, seats[index].max_depth + 1)),
                lambda: seats[index].depth,
                lambda value: self._seat_field(index, depth=value),
                str,
                # Greyed out once the clock is in charge: under iterative deepening the
                # depth is what the search reached, which is a result and not a setting.
                lambda: machine() and seats[index].spec.uses_depth
                and not seats[index].deepens,
            ),
            Spinner(
                row(place.spinner_y[3], config.ROW_H), "vreme",
                lambda: list(TIME_BUDGETS),
                lambda: seats[index].time_budget,
                lambda value: self._seat_field(index, time_budget=value),
                lambda value: f"{value:g} s",
                lambda: machine() and seats[index].deepens,
            ),
            switch("pruning", "alfa-beta odsecanje", place.optimisation_y[0],
                   uses("uses_pruning")),
            # Ordering only bites where there are cutoffs to bring forward, so this one is
            # greyed out whenever pruning is off rather than merely when the algorithm
            # cannot prune.
            switch("move_ordering", "sortiranje poteza", place.optimisation_y[1],
                   lambda: machine() and seats[index].can_order),
            # The table, unlike the ordering, does not need pruning to mean something --
            # it needs only that the same position occur twice, which any tree does.
            switch("transposition", "tabela transpozicija", place.optimisation_y[2],
                   uses("uses_transposition")),
            # The only switch in the window that makes the worst case of a move *smaller*:
            # it hands the depth control over to the clock beside it.
            switch("deepening", "progresivno produbljivanje", place.optimisation_y[3],
                   uses("uses_deepening")),
            switch("threats", "pretnje", place.evaluation_y[0], uses("uses_evaluation")),
            switch("center", "centar", place.evaluation_y[1], uses("uses_evaluation")),
            switch("parity", "parnost", place.evaluation_y[2], uses("uses_evaluation")),
        ]

    def _buttons(self) -> list:
        buttons = [
            Button(pygame.Rect(*config.button_rect(0)), "Nova igra", self.new_game),
            Button(pygame.Rect(*config.button_rect(1)),
                   lambda: "Nastavi" if self.paused else "Pauza", self.toggle_pause),
            Button(pygame.Rect(*config.button_rect(2)), "Korak", self.step,
                   lambda: self.can_step),
        ]
        for slot, (label, controllers) in enumerate(MODES, start=3):
            buttons.append(
                Button(
                    pygame.Rect(*config.button_rect(slot)), label,
                    lambda c=controllers: self.session.set_mode(c),
                    lambda: not self.thinking,
                )
            )
        return buttons

    # -------------------------------------------------------------------------- state

    @property
    def thinking(self) -> bool:
        return self._pending is not None

    @property
    def think_seconds(self) -> float:
        return time.perf_counter() - self._think_started if self.thinking else 0.0

    @property
    def can_step(self) -> bool:
        return (
            self.paused
            and not self.thinking
            and self.falling is None
            and self.session.awaiting_machine
        )

    @property
    def message(self) -> str:
        if self.notice:
            return self.notice
        if self.thinking:
            return "Pretraga traje — prozor i dalje radi, može se zatvoriti u svakom trenutku."
        if self.session.is_over:
            return "Kraj partije. „Nova igra“ (ili N) počinje sledeću."
        if self.paused:
            return "Pauzirano. „Korak“ odigrava tačno jedan potez mašine."
        if self.session.awaiting_human:
            return HELP
        return "Mašina je na potezu."

    # ------------------------------------------------------------------------ actions

    def new_game(self) -> None:
        self._abandon_search()
        self.session.new_game()
        self.falling = None
        self.notice = None
        self._settled_at = pygame.time.get_ticks()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def step(self) -> None:
        if self.can_step:
            self._start_search()

    def quit(self) -> None:
        self.running = False

    # ------------------------------------------------------------------- the search

    def _start_search(self) -> None:
        self._generation += 1
        generation = self._generation
        self._pending = generation
        self._think_started = time.perf_counter()
        session = self.session

        def work() -> None:
            try:
                outcome = session.plan()
            except BaseException as error:  # reported on the main thread, never swallowed
                outcome = error
            self._inbox.put((generation, outcome))

        threading.Thread(target=work, name="c4-search", daemon=True).start()

    def _abandon_search(self) -> None:
        """Stop waiting for whatever is running. The thread finishes and is ignored."""
        self._generation += 1
        self._pending = None

    def _collect(self) -> None:
        while True:
            try:
                generation, outcome = self._inbox.get_nowait()
            except queue.Empty:
                return
            if generation != self._pending:
                continue  # answer to a position that is no longer on the board
            self._pending = None
            if isinstance(outcome, BaseException):
                self.notice = f"Greška u pretrazi: {outcome}"
                self.paused = True
                return
            move, stats = outcome
            self._play(move, stats)

    def _play(self, col: int, stats) -> None:
        player = self.session.board.player_to_move
        self.session.commit(col, stats)
        row = self._landing_row(col)
        if row is not None:
            self.falling = DiscSprite(player, row, col)

    def _landing_row(self, col: int) -> int | None:
        for row in reversed(range(ROWS)):
            if self.session.board.cell(row, col) != EMPTY:
                return row
        return None

    # --------------------------------------------------------------------- main loop

    def run(self) -> None:
        while self.running:
            self.events()
            self.update()
            self.draw()
            self.clock.tick(config.FPS)

    def events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif event.type == pygame.KEYDOWN:
                self.on_key(event.key)
            elif event.type == pygame.MOUSEMOTION:
                self.hover_col = self._column_under(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.on_click(event.pos)

    def on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.quit()
        elif key == pygame.K_SPACE:
            self.toggle_pause()
        elif key == pygame.K_n:
            self.new_game()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.step()

    def on_click(self, pos: tuple[int, int]) -> None:
        for widget in self.widgets:
            if widget.click(pos):
                self.notice = None
                return
        col = self._column_under(pos)
        if col is not None and self.falling is None:
            self.session.drop(col)

    def _column_under(self, pos: tuple[int, int]) -> int | None:
        """The board column a screen position points at, drop strip included."""
        x, y = pos
        if not (config.BOARD_Y - config.DROP_H <= y <= config.BOARD_Y + config.BOARD_H):
            return None
        col = config.column_at(x)
        return col if col is not None and 0 <= col < COLS else None

    def update(self) -> None:
        self._collect()

        if self.falling is not None:
            self.falling.update()
            if self.falling.landed:
                self.falling = None
                self._settled_at = pygame.time.get_ticks()
            return

        if self.thinking or self.paused or not self.session.awaiting_machine:
            return
        if pygame.time.get_ticks() - self._settled_at < config.MACHINE_PAUSE_MS:
            return
        self._start_search()

    def draw(self) -> None:
        self.renderer.draw(self)
