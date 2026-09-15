"""Disc sprites, including the one that is still falling.

Mirrors the departmental framework's `gui/sprites.py`, where the agent sprite walks
towards its target cell a few pixels per frame instead of teleporting. Here the same idea
animates a dropped disc down its column.

The animation is **cosmetic and nothing waits on it for correctness**: `Session` already
has the move when the disc starts falling. The window simply does not ask for the next
move until the disc has landed, so a machine-versus-machine game is watchable rather than
a flicker of final positions.
"""

from __future__ import annotations

import pygame

import config
from core.rules import ROWS
from core.types import Player


def disc_surface(player: Player, radius: int, alpha: int = 255) -> pygame.Surface:
    """A single disc, drawn once and reused."""
    size = radius * 2 + 2
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    colour = config.DISC_COLOURS[int(player) - 1]
    pygame.draw.circle(surface, (*colour, alpha), (size // 2, size // 2), radius)
    pygame.draw.circle(
        surface,
        (*[min(255, channel + 40) for channel in colour], alpha),
        (size // 2, size // 2), radius, width=2,
    )
    return surface


class DiscSprite(pygame.sprite.Sprite):
    """A disc falling from above the board to its final cell."""

    def __init__(self, player: Player, row: int, col: int) -> None:
        super().__init__()
        self.player = player
        self.row = row
        self.col = col
        self.image = disc_surface(player, config.DISC_RADIUS)
        self.rect = self.image.get_rect()

        target_x, target_y = config.cell_center(row, col)
        self.rect.center = (target_x, config.cell_center(ROWS - 1, col)[1] - config.CELL)
        self.target_y = target_y
        self.speed = config.DROP_PIXELS_PER_FRAME

    @property
    def landed(self) -> bool:
        return self.rect.centery >= self.target_y

    def update(self) -> None:
        if self.landed:
            return
        self.speed += 3  # a little gravity, so a long drop does not crawl
        self.rect.centery = min(self.target_y, self.rect.centery + self.speed)
