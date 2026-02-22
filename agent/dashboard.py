# dashboard.py
# Live performance dashboard for DataForge Game Agent — Version 2.0
# Now with:
#   - Dual agent support (Random + Learning on same dashboard)
#   - High contrast colors for light and dark mode
#   - Clear legends on all graphs
#   - Episode progress tracking
#   - Shared JSON file for cross-terminal communication

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os
from collections import deque

# ── Settings ──────────────────────────────────────────────────────────────────
MAX_HISTORY  = 300
UPDATE_EVERY = 1
SHARED_FILE  = "dashboard_data.json"   # shared between terminals

# ── Agent Color Palette ───────────────────────────────────────────────────────
# High contrast, readable on both light and dark backgrounds
AGENT_STYLES = {
    "Random Agent": {
        "color":     "#FF6B6B",   # coral red
        "marker":    "o",
        "linestyle": "--",
        "label":     "🎲 Random Agent",
    },
    "DQN Learning Agent": {
        "color":     "#4ECDC4",   # teal
        "marker":    "s",
        "linestyle": "-",
        "label":     "🧠 DQN Learning Agent",
    },
}

# Fallback for any other agent name
DEFAULT_STYLE = {
    "color":     "#FFE66D",   # yellow
    "marker":    "^",
    "linestyle": "-",
    "label":     "Agent",
}


class Dashboard:
    """
    Live matplotlib dashboard showing agent performance in real time.
    Supports multiple agents simultaneously via shared JSON file.

    Usage (single agent):
        dash = Dashboard(agent_name="Random Agent")
        dash.update(episode, reward, won, steps, correct_flags, pct_cleared)
        dash.close()

    Usage (dual agent — run in separate terminals, same dashboard):
        Both agents create their own Dashboard instance with their own name.
        Data is merged automatically via the shared JSON file.
    """

    def __init__(self, agent_name="Agent"):
        self.agent_name = agent_name
        self.style      = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)

        # Local history for this agent
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

        # Initialize shared file if not exists
        self._init_shared_file()

        # Build the figure
        plt.ion()
        self.fig = plt.figure(figsize=(13, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title(
            "DataForge Game Agent — Live Dashboard")

        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            hspace=0.55, wspace=0.35,
            top=0.93, bottom=0.08
        )

        self.ax_winrate  = self.fig.add_subplot(gs[0, :])
        self.ax_reward   = self.fig.add_subplot(gs[1, 0])
        self.ax_steps    = self.fig.add_subplot(gs[1, 1])
        self.ax_flags    = self.fig.add_subplot(gs[2, 0])
        self.ax_stats    = self.fig.add_subplot(gs[2, 1])

        for ax in [self.ax_winrate, self.ax_reward,
                   self.ax_steps, self.ax_flags, self.ax_stats]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#cccccc", labelsize=8)
            ax.title.set_color("#ffffff")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        # Figure title
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
        """Call after every episode to update dashboard."""

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

        # Write to shared file so other agents can see this data
        self._write_shared_data()

        if episode % UPDATE_EVERY == 0:
            self._draw()

    def close(self):
        """Keep dashboard open after training ends."""
        self._cleanup_shared_file()
        plt.ioff()
        plt.show(block=True)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        # Load all agents' data from shared file
        all_data = self._read_shared_data()

        # ── Win Rate (all agents on same graph) ──────────────────────────────
        self.ax_winrate.cla()
        self._style_axis(self.ax_winrate,
                         "Win Rate Over Time",
                         "Episode", "Win Rate %")
        self.ax_winrate.set_ylim(0, 100)
        self.ax_winrate.axhline(y=50, color="#444466",
                                linestyle=":", linewidth=1, alpha=0.7)

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

        # ── Reward (this agent only) ──────────────────────────────────────────
        self.ax_reward.cla()
        self._style_axis(self.ax_reward,
                         f"Reward — {self.style['label']}",
                         "Episode", "Reward")
        ep_range = list(range(len(self.rewards)))
        colors   = ["#4ECDC4" if r > 0 else "#FF6B6B"
                    for r in self.rewards]
        self.ax_reward.bar(ep_range, list(self.rewards),
                           color=colors, width=0.8, alpha=0.85)
        self.ax_reward.axhline(y=0, color="#888888",
                               linestyle="-", linewidth=0.8)

        # High score annotation
        if self.high_score != float("-inf"):
            hs_str = f"High: {self.high_score:.1f}"
            self.ax_reward.annotate(
                hs_str,
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right", va="top",
                color="#FFE66D", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#1a1a2e",
                          edgecolor="#FFE66D",
                          alpha=0.8)
            )

        # ── Steps Per Episode ─────────────────────────────────────────────────
        self.ax_steps.cla()
        self._style_axis(self.ax_steps,
                         f"Steps — {self.style['label']}",
                         "Episode", "Steps")
        self.ax_steps.plot(
            ep_range, list(self.steps_history),
            color=self.style["color"],
            linewidth=1.5, alpha=0.9
        )
        # Smoothed trend line
        if len(self.steps_history) > 10:
            smooth = np.convolve(
                list(self.steps_history),
                np.ones(10) / 10, mode="valid"
            )
            self.ax_steps.plot(
                list(range(len(smooth))), smooth,
                color="#FFE66D", linewidth=2,
                linestyle="--", label="10-ep avg",
                alpha=0.9
            )
            self.ax_steps.legend(
                loc="upper left",
                facecolor="#1a1a2e",
                edgecolor="#444466",
                labelcolor="#ffffff",
                fontsize=8
            )

        # ── Correct Flags ─────────────────────────────────────────────────────
        self.ax_flags.cla()
        self._style_axis(self.ax_flags,
                         f"Correct Flags — {self.style['label']}",
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

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _format_stats(self, all_data=None):
        """Format the stats panel text."""
        lines = ["── Agent Performance ──\n"]

        if all_data:
            for agent_name, data in all_data.items():
                style = AGENT_STYLES.get(agent_name, DEFAULT_STYLE)
                total = data["wins"] + data["losses"]
                wr    = (data["wins"] / total * 100) if total > 0 else 0.0
                hs    = f"{data['high_score']:.1f}" \
                        if data["high_score"] != float("-inf") else "—"
                lines.append(
                    f"{style['label']}\n"
                    f"  Episodes : {total}\n"
                    f"  Wins     : {data['wins']}\n"
                    f"  Losses   : {data['losses']}\n"
                    f"  Win Rate : {wr:.1f}%\n"
                    f"  High Score: {hs}\n"
                    f"  Best Streak: {data['best_streak']}\n"
                )
        else:
            total = self.wins + self.losses
            wr    = (self.wins / total * 100) if total > 0 else 0.0
            hs    = f"{self.high_score:.1f}" \
                    if self.high_score != float("-inf") else "—"
            lines.append(
                f"{self.style['label']}\n"
                f"  Episodes : {total}\n"
                f"  Wins     : {self.wins}\n"
                f"  Losses   : {self.losses}\n"
                f"  Win Rate : {wr:.1f}%\n"
                f"  High Score: {hs}\n"
                f"  Best Streak: {self.best_streak}\n"
            )

        return "\n".join(lines)

    # ── Shared File (cross-terminal communication) ────────────────────────────

    def _init_shared_file(self):
        """Create shared data file if it doesn't exist."""
        if not os.path.exists(SHARED_FILE):
            self._write_shared_data()

    def _write_shared_data(self):
        """Write this agent's data to the shared file."""
        try:
            existing = self._read_shared_data()
        except Exception:
            existing = {}

        existing[self.agent_name] = {
            "win_rates":  list(self.win_rates),
            "rewards":    list(self.rewards),
            "steps":      list(self.steps_history),
            "flags":      list(self.flags_history),
            "cleared":    list(self.cleared_history),
            "wins":       self.wins,
            "losses":     self.losses,
            "high_score": self.high_score,
            "best_streak":self.best_streak,
        }

        with open(SHARED_FILE, "w") as f:
            json.dump(existing, f)

    def _read_shared_data(self):
        """Read all agents' data from shared file."""
        if not os.path.exists(SHARED_FILE):
            return {}
        try:
            with open(SHARED_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _cleanup_shared_file(self):
        """Remove this agent's entry when done."""
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
