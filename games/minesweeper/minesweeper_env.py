# minesweeper_env.py
# The Minesweeper game environment
# This defines the game rules, the board, and what the agent can see and do

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys

# ── Board Settings ──────────────────────────────────────────────────────────
ROWS        = 9
COLS        = 9
NUM_MINES   = 10

# ── Colours for the pygame window ───────────────────────────────────────────
WHITE       = (255, 255, 255)
GRAY        = (180, 180, 180)
DARK_GRAY   = (100, 100, 100)
BLACK       = (0,   0,   0)
RED         = (200,  30,  30)
GREEN       = (30,  180,  30)
BLUE        = (30,   30, 200)

NUMBER_COLORS = {
    1: (0,   0, 255),
    2: (0, 128,   0),
    3: (255,  0,   0),
    4: (0,   0, 128),
    5: (128,  0,   0),
    6: (0, 128, 128),
    7: (0,   0,   0),
    8: (128, 128, 128),
}

CELL_SIZE   = 50
MARGIN      = 2

# ── Cell States (what the agent sees) ───────────────────────────────────────
HIDDEN      = -1   # not yet revealed
FLAGGED     = -2   # flagged by agent (optional, advanced)
MINE        = -3   # mine — only shown on game over


class MinesweeperEnv(gym.Env):
    """
    A Minesweeper environment following the OpenAI Gymnasium interface.

    The agent sees the board as a grid of integers:
        -1  = hidden cell
        0-8 = revealed cell with that many adjacent mines
        -3  = mine (game over state only)

    The agent picks a cell to reveal each step.
    Reward structure:
        +1   for safely revealing a new cell
        +10  for winning (all safe cells revealed)
        -10  for hitting a mine
        -0.1 small penalty per step to encourage efficiency
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.rows        = ROWS
        self.cols        = COLS
        self.num_mines   = NUM_MINES

        # Action space: agent picks any cell (flattened index)
        self.action_space = spaces.Discrete(self.rows * self.cols)

        # Observation space: the board grid values (-3 to 8)
        self.observation_space = spaces.Box(
            low=-3, high=8,
            shape=(self.rows, self.cols),
            dtype=np.int32
        )

        # Pygame setup (only if we're rendering)
        self.window  = None
        self.clock   = None
        self.font    = None

    # ── Core Gym Methods ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Start a new game."""
        super().reset(seed=seed)

        self.mine_grid  = self._place_mines()
        self.count_grid = self._compute_counts()
        self.visible    = np.full((self.rows, self.cols), HIDDEN, dtype=np.int32)
        self.done       = False
        self.won        = False
        self.steps      = 0

        if self.render_mode == "human":
            self._init_pygame()

        return self.visible.copy(), {}

    def step(self, action):
        """Agent reveals a cell."""
        row = action // self.cols
        col = action  % self.cols

        # Small step penalty to discourage dawdling
        reward = -0.1

        if self.done:
            return self.visible.copy(), 0, True, False, {}

        # Ignore already-revealed cells (valid but wasteful move)
        if self.visible[row, col] != HIDDEN:
            return self.visible.copy(), -0.5, False, False, {}

        # Hit a mine
        if self.mine_grid[row, col]:
            self.visible[row, col] = MINE
            self.done  = True
            self.won   = False
            reward     = -10
            return self.visible.copy(), reward, True, False, {}

        # Safe cell — reveal it (and flood-fill if count is 0)
        self._reveal(row, col)
        reward += 1

        # Check for win
        hidden_safe = np.sum(
            (self.visible == HIDDEN) & (~self.mine_grid)
        )
        if hidden_safe == 0:
            self.done = True
            self.won  = True
            reward    = 10

        self.steps += 1

        if self.render_mode == "human":
            self.render()

        return self.visible.copy(), reward, self.done, False, {}

    def render(self):
        """Draw the current board state in a pygame window."""
        if self.window is None:
            self._init_pygame()

        self.window.fill(WHITE)

        for r in range(self.rows):
            for c in range(self.cols):
                x = c * (CELL_SIZE + MARGIN) + MARGIN
                y = r * (CELL_SIZE + MARGIN) + MARGIN
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                val  = self.visible[r, c]

                if val == HIDDEN:
                    pygame.draw.rect(self.window, GRAY, rect)
                    pygame.draw.rect(self.window, DARK_GRAY, rect, 2)

                elif val == MINE:
                    pygame.draw.rect(self.window, RED, rect)
                    label = self.font.render("💣", True, BLACK)
                    self.window.blit(label, label.get_rect(center=rect.center))

                else:
                    pygame.draw.rect(self.window, WHITE, rect)
                    pygame.draw.rect(self.window, DARK_GRAY, rect, 1)
                    if val > 0:
                        color = NUMBER_COLORS.get(val, BLACK)
                        label = self.font.render(str(val), True, color)
                        self.window.blit(label, label.get_rect(center=rect.center))

        # Status bar at the bottom
        status = "💥 BOOM!" if (self.done and not self.won) else \
                 "🎉 YOU WIN!" if self.won else \
                 f"Steps: {self.steps}"
        status_surf = self.font.render(status, True, BLACK)
        self.window.blit(status_surf, (10, self.rows * (CELL_SIZE + MARGIN) + 5))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _place_mines(self):
        """Randomly place mines on the board."""
        mines = np.zeros((self.rows, self.cols), dtype=bool)
        positions = self.np_random.choice(
            self.rows * self.cols, size=self.num_mines, replace=False
        )
        for pos in positions:
            mines[pos // self.cols][pos % self.cols] = True
        return mines

    def _compute_counts(self):
        """For each cell, count how many of its neighbours are mines."""
        counts = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mine_grid[r, c]:
                    counts[r, c] = -1
                    continue
                total = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            total += self.mine_grid[nr, nc]
                counts[r, c] = total
        return counts

    def _reveal(self, row, col):
        """Reveal a cell. If it has 0 adjacent mines, flood-fill neighbours."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if self.visible[row, col] != HIDDEN:
            return
        if self.mine_grid[row, col]:
            return

        self.visible[row, col] = self.count_grid[row, col]

        # Flood fill for empty cells
        if self.count_grid[row, col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal(row + dr, col + dc)

    def _init_pygame(self):
        """Initialize the pygame window."""
        pygame.init()
        width  = self.cols * (CELL_SIZE + MARGIN) + MARGIN
        height = self.rows * (CELL_SIZE + MARGIN) + MARGIN + 40
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DataForge Game Agent — Minesweeper")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("segoeui", 22)
