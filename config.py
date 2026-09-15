"""Constants for the window: geometry, colours, fonts, timing.

Mirrors the departmental pygame framework's `config.py` -- a flat module of tunables that
everything else reads -- with one deliberate difference: **this module does not import
pygame at the top level.** The board is always 6x7, so the whole layout is arithmetic and
can be computed at import time, which keeps `config` importable from a headless test. Only
the fonts need a live pygame, and they are built by `init_fonts()` once the display exists.

The seat panel's layout is computed once, in `seat_layout`, and both the renderer and the
control panel read it from there. Two copies of "the depth spinner is the third row" would
eventually disagree by three pixels and nobody would know which was right.

Nothing here decides anything about the game. These are pixels and colours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.rules import COLS, ROWS

GAME_FOLDER = os.path.dirname(__file__)
IMG_FOLDER = os.path.join(GAME_FOLDER, "img")

CAPTION = "Connect Four — minimax / alfa-beta / MCTS"

# --------------------------------------------------------------------------- geometry

CELL = 74  # one board square, in pixels
DISC_RADIUS = CELL // 2 - 4
HOLE_RADIUS = CELL // 2 - 3
BOARD_PAD = 10  # blue frame around the holes
GUTTER = 12  # space between the big blocks

BOARD_W = COLS * CELL + 2 * BOARD_PAD
BOARD_H = ROWS * CELL + 2 * BOARD_PAD
DROP_H = CELL  # the strip above the board where the hovered disc is previewed

HEADER_H = 46
SEAT_PANEL_W = 292
PANEL_W = 2 * SEAT_PANEL_W + GUTTER
LEFT_W = BOARD_W

BOARD_X = GUTTER
BOARD_Y = HEADER_H + GUTTER + DROP_H
PANEL_X = GUTTER + LEFT_W + GUTTER
PANEL_Y = HEADER_H + GUTTER

# Buttons sit under the board, in the space the panel needs and the board does not.
CONTROLS_Y = BOARD_Y + BOARD_H + GUTTER
BUTTON_H = 34
BUTTON_GAP = 8
BUTTONS_PER_ROW = 3

# Row heights inside a seat panel.
TITLE_H = 30
SECTION_H = 28  # heading plus the rule under it
ROW_H = 30  # one spinner
CHECK_H = 26  # one checkbox
LINE_H = 20  # one line of statistics
PANEL_PAD = 12
PANEL_INSET = 14
GROUP_GAP = 6

SPINNERS = 3  # kontrola, algoritam, dubina
OPTIMISATIONS = 2  # alpha-beta pruning, move ordering. Phases 8-11 add one each.
EVAL_SWITCHES = 3  # threats, center, parity
LAST_MOVE_LINES = 7
TOTAL_LINES = 4


@dataclass(frozen=True, slots=True)
class SeatLayout:
    """Where everything inside one seat panel goes, in absolute screen coordinates."""

    panel: tuple[int, int, int, int]
    inner_x: int
    width: int
    title_y: int
    settings_header_y: int
    spinner_y: tuple[int, ...]
    optimisation_header_y: int
    optimisation_y: tuple[int, ...]
    evaluation_header_y: int
    evaluation_y: tuple[int, ...]
    agent_y: int
    estimate_y: int
    last_move_header_y: int
    last_move_y: int
    totals_header_y: int
    totals_y: int
    height: int


def _seat_layout(index: int) -> SeatLayout:
    x = PANEL_X + index * (SEAT_PANEL_W + GUTTER)
    y = PANEL_Y + PANEL_PAD

    title_y = y
    y += TITLE_H

    settings_header_y = y
    y += SECTION_H
    spinner_y = tuple(y + row * ROW_H for row in range(SPINNERS))
    y += SPINNERS * ROW_H + GROUP_GAP

    optimisation_header_y = y
    y += SECTION_H
    optimisation_y = tuple(y + row * CHECK_H for row in range(OPTIMISATIONS))
    y += OPTIMISATIONS * CHECK_H + GROUP_GAP

    evaluation_header_y = y
    y += SECTION_H
    evaluation_y = tuple(y + row * CHECK_H for row in range(EVAL_SWITCHES))
    y += EVAL_SWITCHES * CHECK_H + GROUP_GAP + 2

    agent_y = y
    y += 22
    estimate_y = y
    y += 24 + GROUP_GAP

    last_move_header_y = y
    y += SECTION_H
    last_move_y = y
    y += LAST_MOVE_LINES * LINE_H + GROUP_GAP

    totals_header_y = y
    y += SECTION_H
    totals_y = y
    y += TOTAL_LINES * LINE_H + PANEL_PAD

    height = y - PANEL_Y
    return SeatLayout(
        panel=(x, PANEL_Y, SEAT_PANEL_W, height),
        inner_x=x + PANEL_INSET,
        width=SEAT_PANEL_W - 2 * PANEL_INSET,
        title_y=title_y,
        settings_header_y=settings_header_y,
        spinner_y=spinner_y,
        optimisation_header_y=optimisation_header_y,
        optimisation_y=optimisation_y,
        evaluation_header_y=evaluation_header_y,
        evaluation_y=evaluation_y,
        agent_y=agent_y,
        estimate_y=estimate_y,
        last_move_header_y=last_move_header_y,
        last_move_y=last_move_y,
        totals_header_y=totals_header_y,
        totals_y=totals_y,
        height=height,
    )


SEAT_LAYOUTS: tuple[SeatLayout, ...] = (_seat_layout(0), _seat_layout(1))
CONTENT_H = SEAT_LAYOUTS[0].height

WIDTH = GUTTER + LEFT_W + GUTTER + PANEL_W + GUTTER
HEIGHT = HEADER_H + GUTTER + CONTENT_H + GUTTER

FOOTER_Y = HEIGHT - GUTTER - 8


def seat_layout(index: int) -> SeatLayout:
    return SEAT_LAYOUTS[index]


def button_rect(slot: int) -> tuple[int, int, int, int]:
    """The nth control button, filling rows of `BUTTONS_PER_ROW` under the board."""
    width = (LEFT_W - (BUTTONS_PER_ROW - 1) * BUTTON_GAP) // BUTTONS_PER_ROW
    row, column = divmod(slot, BUTTONS_PER_ROW)
    return (
        BOARD_X + column * (width + BUTTON_GAP),
        CONTROLS_Y + row * (BUTTON_H + BUTTON_GAP),
        width,
        BUTTON_H,
    )


# --------------------------------------------------------------------------- colours

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BACKGROUND = (22, 24, 32)
BOARD_BLUE = (30, 62, 148)
HOLE_RIM = (24, 48, 112)

RED = (219, 68, 68)  # player one
YELLOW = (233, 196, 60)  # player two
DISC_COLOURS = (RED, YELLOW)
GHOST_ALPHA = 90

PANEL_BG = (32, 35, 46)
PANEL_EDGE = (58, 63, 80)
PANEL_ACTIVE = (72, 110, 190)

TEXT = (228, 232, 240)
TEXT_MUTED = (146, 154, 172)
TEXT_DIM = (96, 102, 118)

GOOD = (112, 200, 126)
WARN = (232, 178, 70)
BAD = (226, 96, 96)

BUTTON_BG = (46, 51, 66)
BUTTON_HOVER = (62, 69, 88)
BUTTON_EDGE = (78, 85, 106)

HIGHLIGHT = (255, 255, 255)
LAST_MOVE = (255, 255, 255)

# --------------------------------------------------------------------------- timing

FPS = 60
DROP_PIXELS_PER_FRAME = 22  # how fast a disc falls; cosmetic only
MACHINE_PAUSE_MS = 350  # breathing room between machine moves, so a game is watchable

# Above this many seconds the cost estimate is drawn in amber, above the second in red.
ESTIMATE_WARN_SECONDS = 1.0
ESTIMATE_ALARM_SECONDS = 5.0

# --------------------------------------------------------------------------- fonts

FONT_TITLE = None
FONT_BODY = None
FONT_SMALL = None
FONT_MONO = None


def init_fonts() -> None:
    """Build the fonts. Call once, after `pygame.init()`.

    `pygame.font.Font(None, size)` is the font bundled with pygame, so the window looks
    the same on every machine and nothing has to be installed alongside it.
    """
    global FONT_TITLE, FONT_BODY, FONT_SMALL, FONT_MONO
    import pygame

    pygame.font.init()
    FONT_TITLE = pygame.font.Font(None, 30)
    FONT_BODY = pygame.font.Font(None, 23)
    FONT_SMALL = pygame.font.Font(None, 20)
    FONT_MONO = pygame.font.Font(None, 21)


# ------------------------------------------------------------------- board coordinates


def cell_rect(row: int, col: int) -> tuple[int, int, int, int]:
    """Pixel rectangle of one board square.

    Row 0 is the BOTTOM row (`core.rules`) and a screen counts downwards. This is the one
    place in the project where that flip happens.
    """
    x = BOARD_X + BOARD_PAD + col * CELL
    y = BOARD_Y + BOARD_PAD + (ROWS - 1 - row) * CELL
    return x, y, CELL, CELL


def cell_center(row: int, col: int) -> tuple[int, int]:
    x, y, w, h = cell_rect(row, col)
    return x + w // 2, y + h // 2


def column_at(x: int) -> int | None:
    """Which column a horizontal pixel position falls in, or None if outside the board."""
    inner = x - (BOARD_X + BOARD_PAD)
    if inner < 0:
        return None
    col = inner // CELL
    return col if 0 <= col < COLS else None
