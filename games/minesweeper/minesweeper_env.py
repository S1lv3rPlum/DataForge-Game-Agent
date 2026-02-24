# minesweeper_env.py
# The Minesweeper game environment — Version 2.0
# Now with:
#   - Difficulty selection screen (Beginner / Medium / Hard)
#   - Flagging mechanic (right click to flag a mine)
#   - Improved reward structure
#   - Progress display (bombs flagged, board cleared %)
#   - Expanded action space (reveal + flag)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys

# ── Difficulty Configurations ─────────────────────────────────────────────────
DIFFICULTIES = {
    "Beginner": {
        "rows": 9, "cols": 9, "mines": 10,
        "desc": "9x9 Grid • 10 Mines • 71 Safe Cells"
    },
    "Medium": {
        "rows": 14, "cols": 14, "mines": 25,
        "desc": "14x14 Grid • 25 Mines • 171 Safe Cells"
    },
    "Hard": {
        "rows": 24, "cols": 24, "mines": 50,
        "desc": "24x24 Grid • 50 Mines • 526 Safe Cells"
    }
}

# ── Colours ───────────────────────────────────────────────────────────────────
WHITE       = (255, 255, 255)
GRAY        = (180, 180, 180)
DARK_GRAY   = (100, 100, 100)
BLACK       = (0,   0,   0)
RED         = (200,  30,  30)
GREEN       = (30,  180,  30)
BLUE        = (30,   30, 200)
YELLOW      = (255, 200,   0)
BG_COLOR    = (30,   30,  46)   # dark background for difficulty screen
BTN_COLORS  = {
    "Beginner": (46, 160,  67),   # green
    "Medium":   (210, 140,   0),  # amber
    "Hard":     (200,  30,  30),  # red
}
BTN_HOVER   = (80,  80, 120)

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

CELL_SIZE   = 40
MARGIN      = 2
STATUS_H    = 60   # height of status bar at bottom

# ── Cell States ───────────────────────────────────────────────────────────────
HIDDEN      = -1
FLAGGED     = -2
MINE        = -3


class MinesweeperEnv(gym.Env):
    """
    Minesweeper environment following the OpenAI Gymnasium interface.

    Action space (per difficulty):
        First  N actions  = reveal cell N
        Second N actions  = flag/unflag cell N
        Where N = rows * cols

    Observation:
        Grid of integers — HIDDEN(-1), FLAGGED(-2), MINE(-3), or 0-8

    Rewards:
        Safe reveal         = +1.0
        Correct flag        = +5.0
        Wrong flag          = -3.0
        Unflag correct      = -2.0  (penalty for removing correct flag)
        Hit a mine          = -10.0
        Win                 = +20.0
        Repeated action     = -0.5
        Step penalty        = -0.1
    """

    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode=None, difficulty=None):
        super().__init__()

        self.render_mode = render_mode
        self.difficulty  = difficulty   # None = show selection screen

        # Will be set after difficulty selection
        self.rows     = None
        self.cols     = None
        self.num_mines= None

        self.window   = None
        self.clock    = None
        self.font     = None
        self.font_lg  = None
        self.font_sm  = None

        # Placeholder spaces — updated after difficulty chosen
        self.observation_space = spaces.Box(
            low=-3, high=8, shape=(9, 9), dtype=np.int32)
        self.action_space = spaces.Discrete(9 * 9 * 2)

    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

    # Show difficulty screen only if not already set
        if self.difficulty is None:
            if self.render_mode == "human":
                self._init_pygame_basic()
                self.difficulty = self._difficulty_screen()
            else:
                self.difficulty = "Beginner"

        cfg             = DIFFICULTIES[self.difficulty]
        self.rows       = cfg["rows"]
        self.cols       = cfg["cols"]
        self.num_mines  = cfg["mines"]

        # Update spaces for chosen difficulty
        n = self.rows * self.cols
        self.observation_space = spaces.Box(
            low=-3, high=8,
            shape=(self.rows, self.cols),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(n * 2)

    # Board state
        self.mine_grid     = self._place_mines()
        self.count_grid    = self._compute_counts()
        self.visible       = np.full((self.rows, self.cols), HIDDEN, dtype=np.int32)
        self.flags         = np.zeros((self.rows, self.cols), dtype=bool)
        self.done          = False
        self.won           = False
        self.steps         = 0
        self.correct_flags = 0
        self.safe_revealed = 0
        self.safe_total    = self.rows * self.cols - self.num_mines

        if self.render_mode == "human":
            self._init_pygame()

        return self.visible.copy(), {}

    def step(self, action):
        if self.done:
            return self.visible.copy(), 0, True, False, {}

        n          = self.rows * self.cols
        is_flag    = action >= n
        cell_idx   = action % n
        row        = cell_idx // self.cols
        col        = cell_idx  % self.cols

        reward     = -0.1   # step penalty

        if is_flag:
            reward += self._handle_flag(row, col)
        else:
            reward += self._handle_reveal(row, col)

        # Check win
        if not self.done:
            if self.safe_revealed == self.safe_total:
                self.done = True
                self.won  = True
                reward    = 20.0

        self.steps += 1

        if self.render_mode == "human":
            self.render()

        return self.visible.copy(), reward, self.done, False, {}

    def render(self):
        if self.window is None:
            self._init_pygame()

        # Handle pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        self.window.fill((20, 20, 35))

        for r in range(self.rows):
            for c in range(self.cols):
                x    = c * (CELL_SIZE + MARGIN) + MARGIN
                y    = r * (CELL_SIZE + MARGIN) + MARGIN
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                val  = self.visible[r, c]

                if val == HIDDEN:
                    pygame.draw.rect(self.window, GRAY, rect, border_radius=3)
                    pygame.draw.rect(self.window, DARK_GRAY, rect, 1, border_radius=3)

                elif val == FLAGGED:
                    pygame.draw.rect(self.window, YELLOW, rect, border_radius=3)
                    label = self.font.render("🚩", True, BLACK)
                    self.window.blit(label, label.get_rect(center=rect.center))

                elif val == MINE:
                    pygame.draw.rect(self.window, RED, rect, border_radius=3)
                    label = self.font.render("💣", True, BLACK)
                    self.window.blit(label, label.get_rect(center=rect.center))

                else:
                    pygame.draw.rect(self.window, (50, 50, 70), rect, border_radius=3)
                    pygame.draw.rect(self.window, DARK_GRAY, rect, 1, border_radius=3)
                    if val > 0:
                        color = NUMBER_COLORS.get(val, WHITE)
                        label = self.font.render(str(val), True, color)
                        self.window.blit(label, label.get_rect(center=rect.center))

        self._draw_status()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    # ── Action Handlers ───────────────────────────────────────────────────────

    def _handle_reveal(self, row, col):
        """Reveal a cell. Returns reward delta."""
        if self.visible[row, col] != HIDDEN or self.flags[row, col]:
            return -0.5   # already revealed or flagged — wasteful

        if self.mine_grid[row, col]:
            self.visible[row, col] = MINE
            self.done = True
            self.won  = False
            return -10.0

        self._reveal(row, col)
        return 1.0

    def _handle_flag(self, row, col):
        """Flag or unflag a cell. Returns reward delta."""
        if self.visible[row, col] != HIDDEN:
            return -0.5   # can't flag a revealed cell

        if self.flags[row, col]:
            # Unflagging
            self.flags[row, col]   = False
            self.visible[row, col] = HIDDEN
            if self.mine_grid[row, col]:
                self.correct_flags -= 1
                return -2.0   # removed a correct flag — penalize
            return 0.0

        else:
            # Placing a flag
            self.flags[row, col]   = True
            self.visible[row, col] = FLAGGED
            if self.mine_grid[row, col]:
                self.correct_flags += 1
                return 5.0    # correct flag — reward!
            return -3.0       # wrong flag — penalize

    # ── Board Helpers ─────────────────────────────────────────────────────────

    def _place_mines(self):
        mines     = np.zeros((self.rows, self.cols), dtype=bool)
        positions = self.np_random.choice(
            self.rows * self.cols, size=self.num_mines, replace=False)
        for pos in positions:
            mines[pos // self.cols][pos % self.cols] = True
        return mines

    def _compute_counts(self):
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
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return
        if self.visible[row, col] != HIDDEN or self.flags[row, col]:
            return
        if self.mine_grid[row, col]:
            return

        self.visible[row, col]  = self.count_grid[row, col]
        self.safe_revealed      += 1

        if self.count_grid[row, col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    self._reveal(row + dr, col + dc)

    # ── Status Bar ────────────────────────────────────────────────────────────

    def _draw_status(self):
        board_h    = self.rows * (CELL_SIZE + MARGIN) + MARGIN
        status_y   = board_h
        board_w    = self.cols * (CELL_SIZE + MARGIN) + MARGIN

        pygame.draw.rect(self.window, (15, 15, 25),
                         pygame.Rect(0, status_y, board_w, STATUS_H))

        # Bombs flagged progress
        flag_pct   = self.correct_flags / self.num_mines
        bar_w      = int((board_w - 20) * flag_pct)
        pygame.draw.rect(self.window, (60, 60, 80),
                         pygame.Rect(10, status_y + 8, board_w - 20, 12),
                         border_radius=6)
        if bar_w > 0:
            pygame.draw.rect(self.window, YELLOW,
                             pygame.Rect(10, status_y + 8, bar_w, 12),
                             border_radius=6)

        # Board cleared progress
        clear_pct  = self.safe_revealed / self.safe_total if self.safe_total > 0 else 0
        bar_w2     = int((board_w - 20) * clear_pct)
        pygame.draw.rect(self.window, (60, 60, 80),
                         pygame.Rect(10, status_y + 26, board_w - 20, 12),
                         border_radius=6)
        if bar_w2 > 0:
            pygame.draw.rect(self.window, GREEN,
                             pygame.Rect(10, status_y + 26, bar_w2, 12),
                             border_radius=6)

        # Labels
        flag_lbl  = self.font_sm.render(
            f"🚩 Mines: {self.correct_flags}/{self.num_mines}  "
            f"({flag_pct*100:.0f}%)", True, YELLOW)
        clear_lbl = self.font_sm.render(
            f"✅ Cleared: {self.safe_revealed}/{self.safe_total}  "
            f"({clear_pct*100:.0f}%)", True, GREEN)
        step_lbl  = self.font_sm.render(
            f"Steps: {self.steps}  |  {self.difficulty}", True, GRAY)

        self.window.blit(flag_lbl,  (10, status_y + 6))
        self.window.blit(clear_lbl, (10, status_y + 24))
        self.window.blit(step_lbl,
                         (board_w - step_lbl.get_width() - 10, status_y + 6))

        # Win/Loss overlay
        if self.done:
            msg   = "🎉 YOU WIN!" if self.won else "💥 BOOM!"
            color = GREEN if self.won else RED
            lbl   = self.font_lg.render(msg, True, color)
            self.window.blit(lbl, lbl.get_rect(
                center=(board_w // 2, status_y + STATUS_H // 2)))

    # ── Human Play ────────────────────────────────────────────────────────────

    def play_human_episode(self, ai_snapshot=None):
        """
        Run a full human-controlled episode.
        Left click = reveal, Right click = flag.
        Returns dict of episode stats.
        After game ends shows comparison popup if ai_snapshot provided.
        """
        self.render()

        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    row, col = self._mouse_to_cell(event.pos)
                    if row is None:
                        continue

                    n = self.rows * self.cols

                    if event.button == 1:
                        # Left click — reveal
                        action = row * self.cols + col
                        self.step(action)

                    elif event.button == 3:
                        # Right click — flag
                        action = n + row * self.cols + col
                        self.step(action)

            self.render()

        # Game over — show result briefly then popup
        self.render()
        pygame.time.wait(800)

        stats = {
            "won":           self.won,
            "reward":        self._calculate_reward(),
            "steps":         self.steps,
            "correct_flags": self.correct_flags,
            "safe_revealed": self.safe_revealed,
            "safe_total":    self.safe_total,
        }

        if ai_snapshot:
            action = self._show_comparison_popup(stats, ai_snapshot)
            return stats, action

        return stats, "play_again"

    def _calculate_reward(self):
        """Estimate reward for human game."""
        r = self.safe_revealed * 1.0
        r += self.correct_flags * 5.0
        if self.won:
            r += 20.0
        else:
            r -= 10.0
        r -= self.steps * 0.1
        return r

    def _mouse_to_cell(self, pos):
        """Convert mouse position to board row/col. Returns None if outside board."""
        x, y   = pos
        col    = x // (CELL_SIZE + MARGIN)
        row    = y // (CELL_SIZE + MARGIN)
        board_h = self.rows * (CELL_SIZE + MARGIN) + MARGIN

        if y >= board_h:
            return None, None
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None, None

    def _show_comparison_popup(self, human_stats, ai_snapshot):
        """
        Show post-game popup comparing human to AI agents.
        Returns 'play_again' or 'menu'.
        """
        board_w  = self.cols * (CELL_SIZE + MARGIN) + MARGIN
        board_h  = self.rows * (CELL_SIZE + MARGIN) + MARGIN + STATUS_H

        popup_w  = min(board_w - 20, 420)
        popup_h  = 340
        popup_x  = (board_w - popup_w) // 2
        popup_y  = (board_h - popup_h) // 2

        popup_rect     = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
        btn_play_rect  = pygame.Rect(
            popup_x + 20, popup_y + popup_h - 60,
            (popup_w - 60) // 2, 44
        )
        btn_menu_rect  = pygame.Rect(
            popup_x + popup_w // 2 + 10, popup_y + popup_h - 60,
            (popup_w - 60) // 2, 44
        )

        title_font = pygame.font.SysFont("segoeui", 18, bold=True)
        body_font  = pygame.font.SysFont("segoeui", 14)
        btn_font   = pygame.font.SysFont("segoeui", 15, bold=True)

        # Build cheeky message
        msg = self._cheeky_message(human_stats, ai_snapshot)

        while True:
            mouse = pygame.mouse.get_pos()

            # Dim background
            overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.window.blit(overlay, (0, 0))

            # Popup background
            pygame.draw.rect(self.window, (20, 20, 40),
                             popup_rect, border_radius=14)
            pygame.draw.rect(self.window, (78, 205, 196),
                             popup_rect, 2, border_radius=14)

            # Title
            result   = "🎉 You Won!" if human_stats["won"] else "💥 Boom!"
            color    = (30, 180, 30) if human_stats["won"] else (200, 30, 30)
            title    = title_font.render(result, True, color)
            self.window.blit(title, title.get_rect(
                center=(popup_x + popup_w // 2, popup_y + 28)))

            # Human stats
            y = popup_y + 58
            lines = [
                ("Your Score", f"Reward: {human_stats['reward']:.1f}  |  "
                               f"Steps: {human_stats['steps']}  |  "
                               f"Flags: {human_stats['correct_flags']}/"
                               f"{self.num_mines}",
                 (255, 230, 109)),
            ]

            # AI comparison lines
            for agent_name, snap in ai_snapshot.items():
                if snap["episodes"] == 0:
                    continue
                wr  = snap["wins"] / snap["episodes"] * 100
                avg = snap["avg_reward"]
                if agent_name == "Random Agent":
                    col = (255, 107, 107)
                else:
                    col = (78, 205, 196)
                lines.append((
                    agent_name,
                    f"Avg Reward: {avg:.1f}  |  Win Rate: {wr:.1f}%  |  "
                    f"Games: {snap['episodes']}",
                    col
                ))

            for label, value, col in lines:
                lbl = body_font.render(label, True, col)
                val = body_font.render(value, True, (200, 200, 200))
                self.window.blit(lbl, (popup_x + 16, y))
                y  += 20
                self.window.blit(val, (popup_x + 16, y))
                y  += 26

            # Cheeky message
            y += 4
            msg_lbl = body_font.render(msg, True, (255, 230, 109))
            self.window.blit(msg_lbl, msg_lbl.get_rect(
                center=(popup_x + popup_w // 2, y)))

            # Buttons
            play_hovered = btn_play_rect.collidepoint(mouse)
            menu_hovered = btn_menu_rect.collidepoint(mouse)

            pygame.draw.rect(
                self.window,
                (50, 120, 50) if play_hovered else (30, 80, 30),
                btn_play_rect, border_radius=8)
            pygame.draw.rect(
                self.window,
                (80, 80, 160) if menu_hovered else (40, 40, 100),
                btn_menu_rect, border_radius=8)

            play_lbl = btn_font.render("▶ Play Again", True, (255, 255, 255))
            menu_lbl = btn_font.render("← Menu", True, (255, 255, 255))

            self.window.blit(play_lbl, play_lbl.get_rect(
                center=btn_play_rect.center))
            self.window.blit(menu_lbl, menu_lbl.get_rect(
                center=btn_menu_rect.center))

            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_play_rect.collidepoint(event.pos):
                        return "play_again"
                    if btn_menu_rect.collidepoint(event.pos):
                        return "menu"

    def _cheeky_message(self, human_stats, ai_snapshot):
        """Return a contextual cheeky message based on results."""
        learning_snap = ai_snapshot.get("DQN Learning Agent",
                                        {"wins": 0, "episodes": 0})
        learning_wr   = (learning_snap["wins"] /
                         learning_snap["episodes"] * 100) \
                        if learning_snap["episodes"] > 0 else 0

        human_reward  = human_stats["reward"]
        learning_avg  = learning_snap.get("avg_reward", -999)

        if not human_stats["won"] and human_stats["steps"] < 3:
            return "Even the random agent does better sometimes... 😬"
        if learning_wr > 50:
            return "It's officially smarter than you at this. 🧠"
        if learning_avg > human_reward:
            return "The AI is catching up... 🤖"
        if human_stats["won"]:
            return "Nice one! The AI noticed. 👀"
        if learning_wr > 20:
            return "Enjoy it while it lasts! 🎓"
        return "The AI is watching and learning... 👁"
    
    # ── Difficulty Selection Screen ───────────────────────────────────────────

    def _difficulty_screen(self):
        """Show a difficulty selection screen and return chosen difficulty."""
        W = 480
        H = 400
        pygame.display.set_mode((W, H))
        pygame.display.set_caption("DataForge Game Agent — Select Difficulty")

        title_font = pygame.font.SysFont("segoeui", 28, bold=True)
        btn_font   = pygame.font.SysFont("segoeui", 20, bold=True)
        desc_font  = pygame.font.SysFont("segoeui", 13)

        buttons = {}
        btn_h   = 70
        btn_w   = 380
        start_y = 120

        for i, (name, cfg) in enumerate(DIFFICULTIES.items()):
            rect = pygame.Rect((W - btn_w) // 2,
                               start_y + i * (btn_h + 16),
                               btn_w, btn_h)
            buttons[name] = {"rect": rect, "cfg": cfg}

        while True:
            mouse_pos = pygame.mouse.get_pos()
            self.window.fill(BG_COLOR)

            # Title
            title = title_font.render("💣 DataForge Minesweeper Agent", True, WHITE)
            self.window.blit(title, title.get_rect(center=(W // 2, 60)))

            sub = desc_font.render("Select difficulty to begin", True, GRAY)
            self.window.blit(sub, sub.get_rect(center=(W // 2, 95)))

            for name, data in buttons.items():
                rect    = data["rect"]
                hovered = rect.collidepoint(mouse_pos)
                color   = BTN_HOVER if hovered else BTN_COLORS[name]

                pygame.draw.rect(self.window, color, rect, border_radius=10)
                pygame.draw.rect(self.window, WHITE, rect, 2, border_radius=10)

                lbl  = btn_font.render(name, True, WHITE)
                desc = desc_font.render(data["cfg"]["desc"], True, WHITE)

                self.window.blit(lbl,  lbl.get_rect(
                    center=(rect.centerx, rect.centery - 12)))
                self.window.blit(desc, desc.get_rect(
                    center=(rect.centerx, rect.centery + 14)))

            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, data in buttons.items():
                        if data["rect"].collidepoint(event.pos):
                            return name

    # ── Pygame Init ───────────────────────────────────────────────────────────

    def _init_pygame_basic(self):
        """Minimal pygame init for difficulty screen before board size is known."""
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((480, 400))
            self.clock  = pygame.time.Clock()
            self.font    = pygame.font.SysFont("segoeui", 20)
            self.font_lg = pygame.font.SysFont("segoeui", 26, bold=True)
            self.font_sm = pygame.font.SysFont("segoeui", 13)

    def _init_pygame(self):
        """Full pygame init sized to the chosen board."""
        board_w = self.cols * (CELL_SIZE + MARGIN) + MARGIN
        board_h = self.rows * (CELL_SIZE + MARGIN) + MARGIN + STATUS_H

        if self.window is None:
            pygame.init()

        self.window  = pygame.display.set_mode((board_w, board_h))
        pygame.display.set_caption(
            f"DataForge Game Agent — Minesweeper ({self.difficulty})")
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("segoeui", 20)
        self.font_lg = pygame.font.SysFont("segoeui", 26, bold=True)
        self.font_sm = pygame.font.SysFont("segoeui", 13)
