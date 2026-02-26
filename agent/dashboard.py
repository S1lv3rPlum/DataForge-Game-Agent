# dashboard.py
# Live performance dashboard for DataForge Game Agent — Version 6.0
# Now with:
#   - Pure pygame rendering — no matplotlib/tkinter conflicts
#   - Charts rendered to surface and refreshed on timer
#   - Proper pygame buttons for watch controls
#   - Stable window — no freeze on move
#   - All four agents tracked (Random, Learning, Human, future)

import pygame
import numpy as np
import json
import os
import time
import math
from collections import deque

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_HISTORY  = 300
SHARED_FILE  = "dashboard_data.json"
REFRESH_RATE = 2.0    # seconds between chart redraws

# ── Dashboard Dimensions ──────────────────────────────────────────────────────
DASH_W = 1100
DASH_H = 720
PAD    = 16

# ── Colors ────────────────────────────────────────────────────────────────────
BG          = (15,  15,  25)
PANEL_BG    = (22,  33,  62)
PANEL_EDGE  = (68,  68, 102)
TEXT_WHITE  = (240, 240, 240)
TEXT_GRAY   = (170, 170, 190)
TEXT_DIM    = (100, 100, 120)
TEAL        = (78,  205, 196)
BTN_BG      = (30,  30,  55)
BTN_HOVER   = (50,  50,  90)
BTN_EDGE    = (80,  80, 130)

# ── Agent Styles ──────────────────────────────────────────────────────────────
AGENT_STYLES = {
    "Random Agent": {
        "color": (255, 107, 107),
        "label": "Random Agent",
        "dash":  True,
    },
    "DQN Learning Agent": {
        "color": (78,  205, 196),
        "label": "DQN Learning Agent",
        "dash":  False,
    },
    "Human": {
        "color": (255, 230, 109),
        "label": "Human Player",
        "dash":  False,
    },
}

DEFAULT_STYLE = {
    "color": (180, 180, 180),
    "label": "Agent",
    "dash":  False,
}

# ── Shared watch state ────────────────────────────────────────────────────────
watch_request = {"agent": None}  # None=headless, agent name, or "MENU"


class DashButton:
    """Simple pygame button."""

    def __init__(self, rect, label, color, text_color=TEXT_WHITE):
        self.rect       = pygame.Rect(rect)
        self.label      = label
        self.color      = color
        self.text_color = text_color
        self.hovered    = False

    def draw(self, surface, font):
        bg = BTN_HOVER if self.hovered else self.color
        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_EDGE, self.rect, 1, border_radius=8)
        lbl = font.render(self.label, True, self.text_color)
        surface.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Dashboard:
    """
    Pure pygame dashboard window.
    Charts drawn directly with pygame — no matplotlib.
    Runs on main thread, agents write to shared JSON.

    Parameters:
        agent_name  : name of this agent
        show_window : if False, only writes data
        human_mode  : if True, hides AI watch buttons
    """

    def __init__(self, agent_name="Agent",
                 show_window=True, human_mode=False):
        self.agent_name  = agent_name
        self.show_window = show_window
        self.human_mode  = human_mode
        self.style       = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)

        # Local history
        self.episodes        = []
        self.rewards         = deque(maxlen=MAX_HISTORY)
        self.win_rates       = deque(maxlen=MAX_HISTORY)
        self.steps_history   = deque(maxlen=MAX_HISTORY)
        self.flags_history   = deque(maxlen=MAX_HISTORY)
        self.cleared_history = deque(maxlen=MAX_HISTORY)

        self.wins           = 0
        self.losses         = 0
        self.high_score     = float("-inf")
        self.current_streak = 0
        self.best_streak    = 0

        self.screen      = None
        self.clock       = None
        self.font_lg     = None
        self.font_md     = None
        self.font_sm     = None
        self.buttons     = []
        self._last_draw  = 0
        self._running    = True

        self._init_shared_file()

        if self.show_window:
            self._build_window()

    # ── Window Builder ────────────────────────────────────────────────────────

    def _build_window(self):
        pygame.init()
        pygame.display.init()

        # Position window top-left with some margin
        os.environ["SDL_VIDEO_WINDOW_POS"] = "40,40"

        self.screen = pygame.display.set_mode((DASH_W, DASH_H))
        pygame.display.set_caption(
            "DataForge Game Agent — Live Dashboard")

        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("segoeui", 18, bold=True)
        self.font_md = pygame.font.SysFont("segoeui", 14)
        self.font_sm = pygame.font.SysFont("segoeui", 12)

        self._build_buttons()

    def _build_buttons(self):
        self.buttons = []
        btn_y  = DASH_H - 52
        btn_h  = 36
        btn_w  = 180

        if not self.human_mode:
            self.buttons.append(DashButton(
                (PAD, btn_y, btn_w, btn_h),
                "🎲 Watch Random",
                (58, 26, 26),
                (255, 107, 107)
            ))
            self.buttons.append(DashButton(
                (PAD + btn_w + 10, btn_y, btn_w, btn_h),
                "🧠 Watch Learning",
                (20, 58, 58),
                (78, 205, 196)
            ))
            self.buttons.append(DashButton(
                (PAD + (btn_w + 10) * 2, btn_y, btn_w, btn_h),
                "⚡ Headless (Fast)",
                BTN_BG,
                TEXT_GRAY
            ))

        # Menu button always present
        self.buttons.append(DashButton(
            (DASH_W - btn_w - PAD, btn_y, btn_w, btn_h),
            "← Back to Menu",
            BTN_BG,
            TEXT_GRAY
        ))

        # Watch label
        self.watch_label = "👁 Headless (Fast)" \
                           if not self.human_mode else "👤 Human Mode"

    # ── Main Thread Loop ──────────────────────────────────────────────────────

    def main_loop(self, stop_flag):
        """
        Blocks on main thread.
        Handles pygame events and redraws charts on timer.
        stop_flag dict with 'done' key.
        """
        while self._running and not stop_flag.get("done"):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    watch_request["agent"] = "MENU"
                    self._running = False
                    break

                for i, btn in enumerate(self.buttons):
                    if btn.handle_event(event):
                        self._on_button(i)

            # Redraw on timer
            now = time.time()
            if now - self._last_draw >= REFRESH_RATE:
                self._draw()
                self._last_draw = now

            if watch_request.get("agent") == "MENU":
                self._running = False
                break

            self.clock.tick(30)

        # Final draw
        try:
            self._draw()
        except Exception:
            pass

    def _on_button(self, index):
        if self.human_mode:
            # Only menu button
            watch_request["agent"] = "MENU"
            self.watch_label = "Returning..."
            return

        if index == 0:
            watch_request["agent"] = "Random Agent"
            self.watch_label = "👁 Watching: 🎲 Random Agent"
        elif index == 1:
            watch_request["agent"] = "DQN Learning Agent"
            self.watch_label = "👁 Watching: 🧠 Learning Agent"
        elif index == 2:
            watch_request["agent"] = None
            self.watch_label = "👁 Headless (Fast)"
        else:
            watch_request["agent"] = "MENU"
            self.watch_label = "Returning..."

    # ── Public Update ─────────────────────────────────────────────────────────

    def update(self, episode, reward, won, steps,
               correct_flags=0, pct_cleared=0.0):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps_history.append(steps)
        self.flags_history.append(correct_flags)
        self.cleared_history.append(pct_cleared * 100)

        if won:
            self.wins += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak,
                                   self.current_streak)
        else:
            self.losses += 1
            self.current_streak = 0

        total = self.wins + self.losses
        self.win_rates.append((self.wins / total) * 100)
        self.high_score = max(self.high_score, reward)

        self._write_shared_data()

    def close(self):
        self._cleanup_shared_file()
        self._running = False

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        if self.screen is None:
            return

        self.screen.fill(BG)

        # Title bar
        title = self.font_lg.render(
            "⚡ DataForge Game Agent — Live Dashboard",
            True, TEAL)
        self.screen.blit(title, (PAD, PAD))

        # Watch label
        wlbl = self.font_sm.render(self.watch_label, True, (255, 230, 109))
        self.screen.blit(wlbl, (PAD, PAD + 26))

        all_data = self._read_shared_data()

        # ── Chart layout ──────────────────────────────────────────────────────
        chart_y   = 60
        chart_h   = (DASH_H - chart_y - 70) // 2
        half_w    = (DASH_W - PAD * 3) // 2
        full_w    = DASH_W - PAD * 2

        # Row 1
        self._draw_line_chart(
            pygame.Rect(PAD, chart_y, full_w, chart_h),
            all_data, "win_rates",
            "Win Rate Over Time (%)",
            y_min=0, y_max=100,
            ref_line=50
        )

        # Row 2 left
        self._draw_line_chart(
            pygame.Rect(PAD, chart_y + chart_h + PAD, half_w, chart_h),
            all_data, "rewards",
            "Reward Per Episode",
            ref_line=0
        )

        # Row 2 right
        self._draw_line_chart(
            pygame.Rect(PAD * 2 + half_w,
                        chart_y + chart_h + PAD,
                        half_w, chart_h),
            all_data, "steps",
            "Steps Per Episode"
        )

        # Stats panel — bottom right of row 2
        # (flags chart replaced by stats when 4 agents)
        self._draw_stats_panel(
            pygame.Rect(PAD, chart_y + chart_h * 2 + PAD * 2,
                        full_w, 0),   # height unused
            all_data
        )

        # Buttons
        for btn in self.buttons:
            btn.draw(self.screen, self.font_sm)

        pygame.display.flip()

    def _draw_line_chart(self, rect, all_data, key,
                         title, y_min=None, y_max=None,
                         ref_line=None):
        """Draw a line chart in the given rect."""
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_EDGE, rect, 1, border_radius=8)

        # Title
        t = self.font_sm.render(title, True, TEXT_WHITE)
        self.screen.blit(t, (rect.x + 8, rect.y + 6))

        if not all_data:
            return

        # Chart area with padding
        cx = rect.x + 48
        cy = rect.y + 24
        cw = rect.w - 56
        ch = rect.h - 34

        if cw <= 0 or ch <= 0:
            return

        # Gather all values for scaling
        all_vals = []
        for data in all_data.values():
            vals = data.get(key, [])
            if vals:
                all_vals.extend(vals)

        if not all_vals:
            return

        data_min = y_min if y_min is not None else min(all_vals)
        data_max = y_max if y_max is not None else max(all_vals)

        if data_max == data_min:
            data_max = data_min + 1

        def to_screen(idx, total, val):
            x = cx + int(idx / max(total - 1, 1) * cw)
            y = cy + ch - int((val - data_min) /
                              (data_max - data_min) * ch)
            return x, max(cy, min(cy + ch, y))

        # Reference line
        if ref_line is not None and data_min <= ref_line <= data_max:
            ry = cy + ch - int((ref_line - data_min) /
                               (data_max - data_min) * ch)
            pygame.draw.line(
                self.screen, PANEL_EDGE,
                (cx, ry), (cx + cw, ry), 1
            )

        # Y axis labels
        for pct in [0, 0.5, 1.0]:
            val = data_min + pct * (data_max - data_min)
            y   = cy + ch - int(pct * ch)
            lbl = self.font_sm.render(f"{val:.0f}", True, TEXT_DIM)
            self.screen.blit(lbl, (rect.x + 2, y - 6))

        # Lines per agent
        legend_y = rect.y + 6
        for agent_name, data in all_data.items():
            style = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            vals  = data.get(key, [])
            color = style["color"]

            if len(vals) < 2:
                # Draw legend dot only
                pygame.draw.circle(
                    self.screen, color,
                    (rect.x + rect.w - 120, legend_y + 6), 4)
                lbl = self.font_sm.render(
                    style["label"], True, color)
                self.screen.blit(
                    lbl, (rect.x + rect.w - 112, legend_y))
                legend_y += 16
                continue

            total  = len(vals)
            points = [to_screen(i, total, v)
                      for i, v in enumerate(vals)]

            # Dashed or solid line
            if style.get("dash"):
                for i in range(0, len(points) - 1, 2):
                    pygame.draw.line(
                        self.screen, color,
                        points[i], points[min(i + 1, len(points) - 1)],
                        2)
            else:
                if len(points) > 1:
                    pygame.draw.lines(
                        self.screen, color, False, points, 2)

            # Smoothed trend for steps
            if key == "steps" and len(vals) > 10:
                smooth = np.convolve(
                    vals, np.ones(10) / 10, mode="valid")
                s_pts  = [to_screen(i, len(smooth), v)
                          for i, v in enumerate(smooth)]
                if len(s_pts) > 1:
                    dim_color = tuple(max(0, c - 60) for c in color)
                    pygame.draw.lines(
                        self.screen, dim_color,
                        False, s_pts, 1)

            # Fill under win rate
            if key == "win_rates" and len(points) > 1:
                fill_pts = [(cx, cy + ch)] + points + [(cx + cw, cy + ch)]
                fill_surf = pygame.Surface(
                    (rect.w, rect.h), pygame.SRCALPHA)
                alpha_color = (*color, 25)
                pygame.draw.polygon(fill_surf, alpha_color,
                    [(p[0] - rect.x, p[1] - rect.y)
                     for p in fill_pts])
                self.screen.blit(fill_surf, rect.topleft)

            # Legend
            pygame.draw.line(
                self.screen, color,
                (rect.x + rect.w - 120, legend_y + 6),
                (rect.x + rect.w - 100, legend_y + 6), 2)
            lbl = self.font_sm.render(style["label"], True, color)
            self.screen.blit(lbl, (rect.x + rect.w - 96, legend_y))
            legend_y += 16

    def _draw_stats_panel(self, rect, all_data):
        """Draw compact stats row at bottom."""
        if not all_data:
            return

        panel_w = (DASH_W - PAD * 2) // max(len(all_data), 1)
        x       = PAD

        for agent_name, data in all_data.items():
            style  = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            color  = style["color"]
            total  = data["wins"] + data["losses"]
            wr     = (data["wins"] / total * 100) if total > 0 else 0.0
            hs     = f"{data['high_score']:.1f}" \
                     if data["high_score"] != float("-inf") else "--"

            panel  = pygame.Rect(x, DASH_H - 120,
                                 panel_w - 8, 58)
            pygame.draw.rect(self.screen, PANEL_BG,
                             panel, border_radius=8)
            pygame.draw.rect(self.screen, color,
                             panel, 1, border_radius=8)

            name_lbl = self.font_sm.render(
                style["label"], True, color)
            self.screen.blit(name_lbl, (panel.x + 8, panel.y + 6))

            stats_txt = (f"Ep: {total}  W: {data['wins']}  "
                         f"L: {data['losses']}  "
                         f"WR: {wr:.1f}%  "
                         f"Best: {hs}  "
                         f"Streak: {data['best_streak']}")
            stats_lbl = self.font_sm.render(stats_txt, True, TEXT_GRAY)
            self.screen.blit(stats_lbl, (panel.x + 8, panel.y + 26))

            x += panel_w

    # ── Shared File ───────────────────────────────────────────────────────────

    def _init_shared_file(self):
        if not os.path.exists(SHARED_FILE):
            self._write_shared_data()

    def _write_shared_data(self):
        try:
            existing = self._read_shared_data()
        except Exception:
            existing = {}

        existing[self.agent_name] = {
            "win_rates":   list(self.win_rates),
            "rewards":     list(self.rewards),
            "steps":       list(self.steps_history),
            "flags":       list(self.flags_history),
            "cleared":     list(self.cleared_history),
            "wins":        self.wins,
            "losses":      self.losses,
            "high_score":  self.high_score,
            "best_streak": self.best_streak,
        }

        with open(SHARED_FILE, "w") as f:
            json.dump(existing, f)

    def _read_shared_data(self):
        if not os.path.exists(SHARED_FILE):
            return {}
        try:
            with open(SHARED_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _cleanup_shared_file(self):
        try:
            data = self._read_shared_data()
            if self.agent_name in data:
                del data[self.agent_name]
            if data:
                with open(SHARED_FILE, "w") as f:
                    json.dump(data, f)
            elif os.path.exists(SHARED_FILE):
                os.remove(SHARED_FILE)
        except Exception:
            pass
