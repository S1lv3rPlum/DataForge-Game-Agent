# dashboard.py
# Live performance dashboard for DataForge Game Agent — Version 3.0
# Now with:
#   - Single window for both agents
#   - show_window parameter to prevent duplicate windows
#   - Resize crash protection
#   - High contrast colors for light and dark mode
#   - Clear legends on all graphs

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from collections import deque

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_HISTORY  = 300
UPDATE_EVERY = 1
SHARED_FILE  = "dashboard_data.json"

# ── Agent Color Palette ───────────────────────────────────────────────────────
AGENT_STYLES = {
    "Random Agent": {
        "color":     "#FF6B6B",
        "marker":    "o",
        "linestyle": "--",
        "label":     "Random Agent",
    },
    "DQN Learning Agent": {
        "color":     "#4ECDC4",
        "marker":    "s",
        "linestyle": "-",
        "label":     "DQN Learning Agent",
    },
}

DEFAULT_STYLE = {
    "color":     "#FFE66D",
    "marker":    "^",
    "linestyle": "-",
    "label":     "Agent",
}


class Dashboard:
    """
    Live matplotlib dashboard showing agent performance in real time.

    Parameters:
        agent_name  : name of this agent
        show_window : if False, writes data but does not open a window.
                      Use False for the second agent in Both mode to
                      prevent duplicate dashboard windows.
    """

    def __init__(self, agent_name="Agent", show_window=True):
        self.agent_name  = agent_name
        self.show_window = show_window
        self.style       = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)

        # Local history
        self.episodes        = []
        self.rewards         = deque(maxlen=MAX_HISTORY)
        self.win_rates       = deque(maxlen=MAX_HISTORY)
        self.steps_history   = deque(maxlen=MAX_HISTORY)
        self.flags_history   = deque(maxlen=MAX_HISTORY)
        self.cleared_history = deque(maxlen=MAX_HISTORY)

        self.wins            = 0
        self.losses          = 0
        self.high_score      = float("-inf")
        self.current_streak  = 0
        self.best_streak     = 0

        self.fig         = None
        self.stats_text  = None

        # Always initialize shared file
        self._init_shared_file()

        # Only build the window if show_window is True
        if self.show_window:
            self._build_window()

    # ── Window Builder ────────────────────────────────────────────────────────

    def _build_window(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title(
            "DataForge Game Agent — Live Dashboard")

        # Disable resize to prevent crashes
        try:
            self.fig.canvas.manager.window.resizable(False, False)
        except Exception:
            pass

        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            hspace=0.55, wspace=0.35,
            top=0.93, bottom=0.08
        )

        self.ax_winrate = self.fig.add_subplot(gs[0, :])
        self.ax_reward  = self.fig.add_subplot(gs[1, 0])
        self.ax_steps   = self.fig.add_subplot(gs[1, 1])
        self.ax_flags   = self.fig.add_subplot(gs[2, 0])
        self.ax_stats   = self.fig.add_subplot(gs[2, 1])

        for ax in [self.ax_winrate, self.ax_reward,
                   self.ax_steps, self.ax_flags, self.ax_stats]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.title.set_color("#ffffff")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        self.fig.suptitle(
            "DataForge Game Agent — Live Performance Dashboard",
            color="#ffffff", fontsize=13, fontweight="bold", y=0.98
        )

        self.ax_stats.axis("off")
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, self._format_stats(),
            transform=self.ax_stats.transAxes,
            ha="center", va="center",
            fontsize=10, color="white",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#0f3460",
                      edgecolor="#4ECDC4",
                      linewidth=1.5)
        )

        plt.show(block=False)
        plt.pause(0.1)

    # ── Public Method ─────────────────────────────────────────────────────────

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
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.losses += 1
            self.current_streak = 0

        total = self.wins + self.losses
        self.win_rates.append((self.wins / total) * 100)
        self.high_score = max(self.high_score, reward)

        self._write_shared_data()

        if self.show_window and episode % UPDATE_EVERY == 0:
            self._draw()

    def close(self):
        self._cleanup_shared_file()
        if self.show_window and self.fig is not None:
            plt.ioff()
            plt.show(block=True)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        if self.fig is None:
            return

        all_data = self._read_shared_data()

        # ── Win Rate ──────────────────────────────────────────────────────────
        self.ax_winrate.cla()
        self._style_axis(self.ax_winrate,
                         "Win Rate Over Time",
                         "Episode", "Win Rate %")
        self.ax_winrate.set_ylim(0, 100)
        self.ax_winrate.axhline(
            y=50, color="#444466", linestyle=":", linewidth=1, alpha=0.7)

        for agent_name, data in all_data.items():
            style    = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            episodes = list(range(len(data["win_rates"])))
            if not episodes:
                continue
            self.ax_winrate.plot(
                episodes, data["win_rates"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2,
                label=style["label"],
                marker=style["marker"],
                markevery=max(1, len(episodes) // 20),
                markersize=4
            )
            self.ax_winrate.fill_between(
                episodes, data["win_rates"],
                alpha=0.12, color=style["color"]
            )

        self.ax_winrate.legend(
            loc="upper left",
            facecolor="#1a1a2e",
            edgecolor="#444466",
            labelcolor="#ffffff",
            fontsize=9
        )

        # ── Reward ────────────────────────────────────────────────────────────
        self.ax_reward.cla()
        self._style_axis(self.ax_reward,
                         "Reward Per Episode",
                         "Episode", "Reward")
        self.ax_reward.axhline(
            y=0, color="#888888", linestyle="-", linewidth=0.8)

        for agent_name, data in all_data.items():
            style    = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            episodes = list(range(len(data["rewards"])))
            if not episodes:
                continue
            self.ax_reward.plot(
                episodes, data["rewards"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                label=style["label"],
                alpha=0.9
            )

        # High score annotation for this agent
        if self.high_score != float("-inf"):
            self.ax_reward.annotate(
                f"High: {self.high_score:.1f}",
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right", va="top",
                color="#FFE66D", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#1a1a2e",
                          edgecolor="#FFE66D",
                          alpha=0.8)
            )

        self.ax_reward.legend(
            loc="upper left",
            facecolor="#1a1a2e",
            edgecolor="#444466",
            labelcolor="#ffffff",
            fontsize=8
        )

       # ── Steps ─────────────────────────────────────────────────────────────
        self.ax_steps.cla()
        self._style_axis(self.ax_steps,
                         "Steps Per Episode",
                         "Episode", "Steps")

        for agent_name, data in all_data.items():
            style    = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            episodes = list(range(len(data["steps"])))
            if not episodes:
                continue

            self.ax_steps.plot(
                episodes, data["steps"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                label=style["label"],
                alpha=0.9
            )

            # Smoothed trend line per agent
            if len(data["steps"]) > 10:
                smooth = np.convolve(
                    data["steps"],
                    np.ones(10) / 10, mode="valid"
                )
                self.ax_steps.plot(
                    list(range(len(smooth))), smooth,
                    color=style["color"],
                    linewidth=2.5,
                    linestyle="--",
                    alpha=0.6
                )

        self.ax_steps.legend(
            loc="upper left",
            facecolor="#1a1a2e",
            edgecolor="#444466",
            labelcolor="#ffffff",
            fontsize=8
        )

        # ── Flags ─────────────────────────────────────────────────────────────
        self.ax_flags.cla()
        self._style_axis(self.ax_flags,
                         "Correct Flags",
                         "Episode", "Flags")

        for agent_name, data in all_data.items():
            style    = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
            episodes = list(range(len(data["flags"])))
            if not episodes:
                continue
            self.ax_flags.plot(
                episodes, data["flags"],
                color=style["color"],
                linewidth=1.5,
                linestyle=style["linestyle"],
                label=style["label"],
                alpha=0.9
            )

        self.ax_flags.legend(
            loc="upper left",
            facecolor="#1a1a2e",
            edgecolor="#444466",
            labelcolor="#ffffff",
            fontsize=8
        )

        # ── Stats Panel ───────────────────────────────────────────────────────
        self.stats_text.set_text(self._format_stats(all_data))

        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _format_stats(self, all_data=None):
        lines = ["── Agent Performance ──\n"]

        if all_data:
            for agent_name, data in all_data.items():
                style = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
                total = data["wins"] + data["losses"]
                wr    = (data["wins"] / total * 100) if total > 0 else 0.0
                hs    = f"{data['high_score']:.1f}" \
                        if data["high_score"] != float("-inf") else "--"
                lines.append(
                    f"{style['label']}\n"
                    f"  Episodes  : {total}\n"
                    f"  Wins      : {data['wins']}\n"
                    f"  Losses    : {data['losses']}\n"
                    f"  Win Rate  : {wr:.1f}%\n"
                    f"  High Score: {hs}\n"
                    f"  Best Streak: {data['best_streak']}\n"
                )
        else:
            total = self.wins + self.losses
            wr    = (self.wins / total * 100) if total > 0 else 0.0
            hs    = f"{self.high_score:.1f}" \
                    if self.high_score != float("-inf") else "--"
            lines.append(
                f"{self.style['label']}\n"
                f"  Episodes  : {total}\n"
                f"  Wins      : {self.wins}\n"
                f"  Losses    : {self.losses}\n"
                f"  Win Rate  : {wr:.1f}%\n"
                f"  High Score: {hs}\n"
                f"  Best Streak: {self.best_streak}\n"
            )

        return "\n".join(lines)

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

    # ── Axis Styling ──────────────────────────────────────────────────────────

    def _style_axis(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#16213e")
        ax.set_title(title,   color="#ffffff", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
