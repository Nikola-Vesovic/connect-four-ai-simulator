"""The smallest set of clickable controls the panel needs. pygame has none of its own.

Three widgets cover every setting in this application:

* `Spinner`  -- one of a list of values, with arrows on either side (seat, algorithm, depth)
* `Checkbox` -- a switch (alpha-beta pruning, each evaluation component)
* `Button`   -- an action (new game, pause, step)

All three read and write through a getter and a setter rather than holding state of their
own, so the only copy of "what the user chose" is the one in `Session`. A widget cannot
drift out of step with the game, because it has nothing to drift with. The `enabled`
callback is the same idea: the pruning checkbox greys itself out when the chosen algorithm
has no pruning to switch, without anyone having to remember to tell it.

`visible` is the stronger form of `enabled`, added in Phase 11 for one reason: MCTS is a
fifth *algorithm* rather than a sixth checkbox, so it brings five switches of its own and
they occupy the same five rows of the panel that alpha-beta's do. A greyed-out
"alfa-beta odsecanje" is an honest label for a negamax seat and a meaningless one for an
MCTS seat, so that row shows a different control instead -- and the window does not grow
by a single pixel.
"""

from __future__ import annotations

from typing import Callable, Sequence

import pygame

import config


def _always() -> bool:
    return True


class Widget:
    def __init__(self, rect: pygame.Rect, enabled: Callable[[], bool] = _always,
                 visible: Callable[[], bool] = _always) -> None:
        self.rect = rect
        self.enabled = enabled
        self.visible = visible

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.visible() and self.enabled() and self.rect.collidepoint(pos)

    def click(self, pos: tuple[int, int]) -> bool:
        """Handle a click inside this widget. True if the widget consumed it."""
        raise NotImplementedError

    def draw(self, surface: pygame.Surface, mouse: tuple[int, int]) -> None:
        raise NotImplementedError


def _text(surface, font, message, colour, pos, right=False, center=False) -> pygame.Rect:
    image = font.render(message, True, colour)
    rect = image.get_rect()
    if right:
        rect.midright = pos
    elif center:
        rect.center = pos
    else:
        rect.midleft = pos
    surface.blit(image, rect)
    return rect


class Spinner(Widget):
    """`label   < value >` -- the arrows step through a list of options."""

    ARROW_W = 22
    VALUE_W = 112

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        options: Callable[[], Sequence],
        get: Callable[[], object],
        set_: Callable[[object], None],
        fmt: Callable[[object], str] = str,
        enabled: Callable[[], bool] = _always,
        visible: Callable[[], bool] = _always,
    ) -> None:
        super().__init__(rect, enabled, visible)
        self.label = label
        self.options = options
        self.get = get
        self.set = set_
        self.fmt = fmt

    @property
    def _box(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - self.VALUE_W, self.rect.y + 2,
                           self.VALUE_W, self.rect.h - 4)

    @property
    def _left(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - self.VALUE_W, self.rect.y,
                           self.ARROW_W, self.rect.h)

    @property
    def _right(self) -> pygame.Rect:
        return pygame.Rect(self.rect.right - self.ARROW_W, self.rect.y, self.ARROW_W, self.rect.h)

    def _step(self, delta: int) -> None:
        options = list(self.options())
        if not options:
            return
        try:
            index = options.index(self.get())
        except ValueError:
            index = 0
        self.set(options[(index + delta) % len(options)])

    def click(self, pos: tuple[int, int]) -> bool:
        if not self.hit(pos):
            return False
        self._step(-1 if self._left.collidepoint(pos) else 1)
        return True

    def draw(self, surface: pygame.Surface, mouse: tuple[int, int]) -> None:
        if not self.visible():
            return
        live = self.enabled()
        label_colour = config.TEXT_MUTED if live else config.TEXT_DIM
        value_colour = config.TEXT if live else config.TEXT_DIM
        _text(surface, config.FONT_SMALL, self.label, label_colour,
              (self.rect.x, self.rect.centery))

        box = self._box
        pygame.draw.rect(surface, config.BUTTON_BG, box, border_radius=5)
        pygame.draw.rect(surface, config.BUTTON_EDGE, box, width=1, border_radius=5)

        for arrow, glyph in ((self._left, "<"), (self._right, ">")):
            hot = live and arrow.collidepoint(mouse)
            _text(surface, config.FONT_BODY, glyph,
                  config.TEXT if hot else (config.TEXT_MUTED if live else config.TEXT_DIM),
                  arrow.center, center=True)

        _text(surface, config.FONT_SMALL, self.fmt(self.get()), value_colour,
              box.center, center=True)


class Checkbox(Widget):
    """A switch. The box is filled when on, hollow when off, dim when not applicable."""

    BOX = 16

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        get: Callable[[], bool],
        set_: Callable[[bool], None],
        enabled: Callable[[], bool] = _always,
        visible: Callable[[], bool] = _always,
    ) -> None:
        super().__init__(rect, enabled, visible)
        self.label = label
        self.get = get
        self.set = set_

    def click(self, pos: tuple[int, int]) -> bool:
        if not self.hit(pos):
            return False
        self.set(not self.get())
        return True

    def draw(self, surface: pygame.Surface, mouse: tuple[int, int]) -> None:
        if not self.visible():
            return
        live = self.enabled()
        on = bool(self.get())
        hot = live and self.rect.collidepoint(mouse)

        box = pygame.Rect(self.rect.x, self.rect.centery - self.BOX // 2, self.BOX, self.BOX)
        if live and on:
            pygame.draw.rect(surface, config.PANEL_ACTIVE, box, border_radius=4)
            pygame.draw.lines(
                surface, config.WHITE, False,
                [(box.x + 3, box.centery), (box.centerx - 1, box.bottom - 4),
                 (box.right - 3, box.y + 4)], 2,
            )
        else:
            pygame.draw.rect(surface, config.PANEL_BG, box, border_radius=4)
        pygame.draw.rect(
            surface,
            config.BUTTON_EDGE if live else config.TEXT_DIM,
            box, width=1, border_radius=4,
        )

        colour = config.TEXT_DIM if not live else (config.TEXT if hot or on else config.TEXT_MUTED)
        _text(surface, config.FONT_SMALL, self.label, colour,
              (box.right + 10, self.rect.centery))


class Button(Widget):
    def __init__(
        self,
        rect: pygame.Rect,
        label: Callable[[], str] | str,
        action: Callable[[], None],
        enabled: Callable[[], bool] = _always,
    ) -> None:
        super().__init__(rect, enabled)
        self._label = label
        self.action = action

    @property
    def label(self) -> str:
        return self._label() if callable(self._label) else self._label

    def click(self, pos: tuple[int, int]) -> bool:
        if not self.hit(pos):
            return False
        self.action()
        return True

    def draw(self, surface: pygame.Surface, mouse: tuple[int, int]) -> None:
        live = self.enabled()
        hot = live and self.rect.collidepoint(mouse)
        pygame.draw.rect(
            surface,
            config.BUTTON_HOVER if hot else config.BUTTON_BG,
            self.rect, border_radius=6,
        )
        pygame.draw.rect(
            surface,
            config.BUTTON_EDGE if live else config.PANEL_EDGE,
            self.rect, width=1, border_radius=6,
        )
        _text(surface, config.FONT_SMALL, self.label,
              config.TEXT if live else config.TEXT_DIM, self.rect.center, center=True)
