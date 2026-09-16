"""Everything the window draws. It reads state; it never changes any.

Mirrors the departmental framework's `gui/renderer.py`: one object that owns the screen
and knows how to put the model on it. The only board knowledge here is `board.cell()` and
the row flip -- `core.rules` keeps row 0 at the bottom so the two board implementations
line up bit for bit, and a screen counts downwards, so exactly one place turns one into
the other. That place is `config.cell_rect`.

The labels are Serbian; the algorithm names are not, because "minimax" and "alpha-beta"
are the names used in the course material and in the literature.
"""

from __future__ import annotations

import pygame

import config
from core.rules import COLS, ROWS, Board
from core.types import EMPTY, GameResult, Player
from gui.sprites import disc_surface
from session import Controller, SeatTotals, Session

SEAT_NAMES = ("IGRAČ 1", "IGRAČ 2")
SEAT_COLOUR_NAMES = ("crveni", "žuti")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def seconds(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}".replace(".", ",") + " s"


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.sheet = self._build_sheet()
        self._discs = {player: disc_surface(player, config.DISC_RADIUS) for player in Player}
        self._ghosts = {
            player: disc_surface(player, config.DISC_RADIUS, config.GHOST_ALPHA)
            for player in Player
        }

    def _build_sheet(self) -> pygame.Surface:
        """The blue board with the holes punched out of it.

        Blitted *over* the discs rather than under them, which is what makes a falling
        disc disappear behind the board and show through the holes, as it does in the
        physical game.
        """
        sheet = pygame.Surface((config.BOARD_W, config.BOARD_H), pygame.SRCALPHA)
        sheet.fill(config.BOARD_BLUE)
        for row in range(ROWS):
            for col in range(COLS):
                x, y, w, h = config.cell_rect(row, col)
                local = (x - config.BOARD_X + w // 2, y - config.BOARD_Y + h // 2)
                pygame.draw.circle(sheet, config.HOLE_RIM, local, config.HOLE_RADIUS + 2)
                pygame.draw.circle(sheet, (0, 0, 0, 0), local, config.HOLE_RADIUS)
        return sheet

    # ------------------------------------------------------------------------ helpers

    def text(self, message, font, colour, pos, right=False, center=False) -> pygame.Rect:
        image = font.render(message, True, colour)
        rect = image.get_rect()
        if right:
            rect.midright = pos
        elif center:
            rect.center = pos
        else:
            rect.midleft = pos
        self.screen.blit(image, rect)
        return rect

    def section(self, title: str, x: int, y: int, width: int) -> None:
        """A small heading with a rule under it."""
        self.text(title, config.FONT_SMALL, config.TEXT_DIM, (x, y + 9))
        pygame.draw.line(self.screen, config.PANEL_EDGE, (x, y + 21), (x + width, y + 21))

    def stat(self, label: str, value: str, x: int, y: int, width: int, colour=None) -> int:
        middle = y + config.LINE_H // 2
        self.text(label, config.FONT_SMALL, config.TEXT_MUTED, (x, middle))
        self.text(value, config.FONT_MONO, colour or config.TEXT, (x + width, middle), right=True)
        return y + config.LINE_H

    # -------------------------------------------------------------------------- frame

    def draw(self, view) -> None:
        self.screen.fill(config.BACKGROUND)
        mouse = pygame.mouse.get_pos()

        self.draw_header(view)
        self.draw_drop_zone(view)
        self.draw_board(view)
        for index in (0, 1):
            self.draw_seat_panel(view, index)
        for widget in view.widgets:
            widget.draw(self.screen, mouse)
        self.draw_footer(view)

        pygame.display.flip()

    def draw_header(self, view) -> None:
        session: Session = view.session
        pygame.draw.rect(self.screen, config.PANEL_BG,
                         pygame.Rect(0, 0, config.WIDTH, config.HEADER_H))
        pygame.draw.line(self.screen, config.PANEL_EDGE,
                         (0, config.HEADER_H - 1), (config.WIDTH, config.HEADER_H - 1))

        middle = config.HEADER_H // 2
        self.text(session.mode_label, config.FONT_BODY, config.TEXT_MUTED,
                  (config.GUTTER, middle))

        message, colour = self.status(view)
        self.text(message, config.FONT_TITLE, colour, (config.WIDTH // 2, middle), center=True)

        right = (config.WIDTH - config.GUTTER, middle)
        if view.thinking:
            self.text(f"razmišlja… {seconds(view.think_seconds, 1)}",
                      config.FONT_BODY, config.WARN, right, right=True)
        elif view.paused:
            self.text("PAUZA", config.FONT_BODY, config.WARN, right, right=True)

    def status(self, view) -> tuple[str, tuple[int, int, int]]:
        session: Session = view.session
        if session.is_over:
            result = session.result
            if result is GameResult.DRAW:
                return "Nerešeno", config.TEXT
            seat = 0 if result is GameResult.ONE_WINS else 1
            return (f"Pobedio {SEAT_NAMES[seat]} ({SEAT_COLOUR_NAMES[seat]})",
                    config.DISC_COLOURS[seat])

        seat = session.seat_to_move
        agent = session.agents[seat]
        who = "čovek" if agent is None else agent.name
        return f"Na potezu: {SEAT_NAMES[seat]} — {who}", config.DISC_COLOURS[seat]

    # -------------------------------------------------------------------------- board

    def draw_drop_zone(self, view) -> None:
        session: Session = view.session
        if view.falling is not None or not session.awaiting_human:
            return
        if view.hover_col is None or not session.board.can_play(view.hover_col):
            return
        ghost = self._ghosts[session.board.player_to_move]
        x, _, w, _ = config.cell_rect(ROWS - 1, view.hover_col)
        centre = (x + w // 2, config.BOARD_Y - config.DROP_H // 2)
        self.screen.blit(ghost, ghost.get_rect(center=centre))

    def draw_board(self, view) -> None:
        board: Board = view.session.board
        # The disc that is still in the air is already on the board as far as the game is
        # concerned; the animation is the only thing that has not caught up.
        airborne = (view.falling.row, view.falling.col) if view.falling else None
        for row in range(ROWS):
            for col in range(COLS):
                occupant = board.cell(row, col)
                if occupant == EMPTY or (row, col) == airborne:
                    continue
                image = self._discs[Player(occupant)]
                self.screen.blit(image, image.get_rect(center=config.cell_center(row, col)))

        if view.falling is not None:
            self.screen.blit(view.falling.image, view.falling.rect)

        self.screen.blit(self.sheet, (config.BOARD_X, config.BOARD_Y))

        if view.falling is not None:
            return

        last = view.session.last_move()
        if last is not None:
            row = self._top_of_column(board, last.column)
            if row is not None:
                pygame.draw.circle(self.screen, config.LAST_MOVE,
                                   config.cell_center(row, last.column),
                                   config.DISC_RADIUS - 4, width=2)

        for row, col in view.session.winning_cells() or ():
            pygame.draw.circle(self.screen, config.HIGHLIGHT,
                               config.cell_center(row, col), config.DISC_RADIUS, width=3)

    @staticmethod
    def _top_of_column(board: Board, col: int) -> int | None:
        """Topmost occupied cell of a column -- where its most recent disc landed."""
        for row in reversed(range(ROWS)):
            if board.cell(row, col) != EMPTY:
                return row
        return None

    # -------------------------------------------------------------------------- panel

    def draw_seat_panel(self, view, index: int) -> None:
        session: Session = view.session
        place = config.seat_layout(index)
        panel = pygame.Rect(*place.panel)

        active = not session.is_over and session.seat_to_move == index
        pygame.draw.rect(self.screen, config.PANEL_BG, panel, border_radius=8)
        pygame.draw.rect(self.screen,
                         config.DISC_COLOURS[index] if active else config.PANEL_EDGE,
                         panel, width=2 if active else 1, border_radius=8)

        x, width = place.inner_x, place.width
        pygame.draw.circle(self.screen, config.DISC_COLOURS[index],
                           (x + 8, place.title_y + 13), 8)
        self.text(SEAT_NAMES[index], config.FONT_BODY, config.TEXT,
                  (x + 24, place.title_y + 13))

        self.section("PODEŠAVANJA", x, place.settings_header_y, width)
        self.section("OPTIMIZACIJE", x, place.optimisation_header_y, width)
        self.section("FUNKCIJA PROCENE", x, place.evaluation_header_y, width)

        self.draw_agent_line(view, index, place)
        self.draw_last_move(view, index, place)
        self.draw_totals(session.totals[index], place)

    def draw_agent_line(self, view, index: int, place) -> None:
        session: Session = view.session
        spec = session.seats[index]
        agent = session.agents[index]
        x, width = place.inner_x, place.width

        if spec.controller is Controller.HUMAN:
            self.text("ovim igračem upravlja čovek", config.FONT_SMALL, config.TEXT_DIM,
                      (x, place.agent_y + 11))
            return

        name = agent.name if agent is not None else spec.spec.label
        self.text(f"agent: {name}", config.FONT_SMALL, config.TEXT_MUTED,
                  (x, place.agent_y + 11))

        estimate = spec.estimated_seconds
        middle = place.estimate_y + 12
        if not spec.spec.uses_depth or estimate is None:
            self.text("cena po potezu: zanemarljiva", config.FONT_SMALL, config.GOOD,
                      (x, middle))
            return

        if estimate >= config.ESTIMATE_ALARM_SECONDS:
            colour = config.BAD
        elif estimate >= config.ESTIMATE_WARN_SECONDS:
            colour = config.WARN
        else:
            colour = config.GOOD
        self.text("izmereno najgore", config.FONT_SMALL, config.TEXT_MUTED, (x, middle))
        self.text(seconds(estimate, 2), config.FONT_MONO, colour, (x + width, middle), right=True)

    def draw_last_move(self, view, index: int, place) -> None:
        session: Session = view.session
        x, width = place.inner_x, place.width
        self.section("POSLEDNJI POTEZ", x, place.last_move_header_y, width)

        stats = session.last_stats(index)
        y = place.last_move_y
        if stats is None:
            played = any(record.seat == index for record in session.history)
            self.text("odigrao čovek" if played else "—", config.FONT_SMALL,
                      config.TEXT_DIM, (x, y + config.LINE_H // 2))
            return

        y = self.stat("kolona", str((stats.move or 0) + 1), x, y, width)
        y = self.stat("vrednost", "—" if stats.value is None else f"{stats.value:+d}",
                      x, y, width)
        y = self.stat("dubina", str(stats.depth), x, y, width)
        y = self.stat("čvorova", thousands(stats.nodes), x, y, width, config.TEXT)
        y = self.stat("odsecanja", thousands(stats.cutoffs), x, y, width,
                      config.GOOD if stats.cutoffs else config.TEXT_MUTED)
        # How good the move ordering was: the share of cut nodes that cut on the first
        # move tried. 100% would be a perfect ordering.
        y = self.stat("red poteza",
                      f"{stats.first_cutoff_rate:.0%}" if stats.cutoffs else "—",
                      x, y, width,
                      config.GOOD if stats.first_cutoff_rate >= 0.8 else config.TEXT_MUTED)
        # How much of the tree was answered out of the transposition table instead of
        # being searched: the share of probes that ended the node there and then.
        y = self.stat("transpozicije",
                      f"{stats.tt_return_rate:.0%}" if stats.tt_probes else "—",
                      x, y, width,
                      config.GOOD if stats.tt_returns else config.TEXT_MUTED)
        self.stat("vreme", seconds(stats.elapsed), x, y, width)

    def draw_totals(self, totals: SeatTotals, place) -> None:
        x, width = place.inner_x, place.width
        self.section("UKUPNO U PARTIJI", x, place.totals_header_y, width)
        y = place.totals_y
        y = self.stat("poteza", str(totals.moves), x, y, width)
        y = self.stat("čvorova", thousands(totals.nodes), x, y, width)
        y = self.stat("vreme", seconds(totals.seconds, 2), x, y, width)
        self.stat("čvorova/potez", thousands(round(totals.nodes_per_move)), x, y, width)

    # ------------------------------------------------------------------------- footer

    def draw_footer(self, view) -> None:
        self.text(view.message, config.FONT_SMALL, config.TEXT_MUTED,
                  (config.GUTTER, config.FOOTER_Y))
