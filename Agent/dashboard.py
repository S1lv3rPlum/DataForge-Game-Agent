# dashboard.py
# Live performance dashboard for DataForge Game Agent
# Displays real-time stats and graphs alongside the game window
# Works with both random and learning agents

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque

# ── Settings ────────────────────────────────────────────────────────────────
MAX_HISTORY = 200      # how many episodes to show on graphs
UPDATE_EVERY = 1       # update dashboard every N episodes


class Dashboard:
    """
    Live matplotlib dashboard showing agent performance in real time.
    
    Usage:
        dash = Dashboard(agent_name="Random Agent")
        dash.update(episode, reward, won, steps)
        dash.close()
    """

    def __init__(self, agent_name="Agent", color="#4C72B0"):
        self.agent_name  = agent_name
        self.color       = color

        # History tracking
        self.episodes       = []
        self.rewards        = deque(maxlen=MAX_HISTORY)
        self.win_rates      = deque(maxlen=MAX_HISTORY)
        self.steps_history  = deque(maxlen=MAX_HISTORY)
        self.wins           = 0
        self.losses         = 0
        self.high_score     = float("-inf")
        self.current_streak = 0
        self.best_streak    = 0

        # Set up the figure
        plt.ion()
        self.fig = plt.figure(
            figsize=(10, 8),
            facecolor="#1e1e1e"
        )
        self.fig.canvas.manager.set_window_title(
            f"DataForge Game Agent — {agent_name} Dashboard"
        )

        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.45, wspace=0.35)

        # ── Subplots ────────────────────────────────────────────────────────
        self.ax_winrate  = self.fig.add_subplot(gs[0, :])   # full width top
        self.ax_reward   = self.fig.add_subplot(gs[1, 0])   # middle left
        self.ax_steps    = self.fig.add_subplot(gs[1, 1])   # middle right
        self.ax_stats    = self.fig.add_subplot(gs[2, :])   # full width bottom

        for ax in [self.ax_winrate, self.ax_reward,
                   self.ax_steps, self.ax_stats]:
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")

        self._style_axis(self.ax_winrate, "Win Rate Over Time", "Episode", "Win Rate %")
        self._style_axis(self.ax_reward,  "Reward Per Episode", "Episode", "Reward")
        self._style_axis(self.ax_steps,   "Steps Per Episode",  "Episode", "Steps")

        self.ax_stats.axis("off")
        self.stats_text = self.ax_stats.text(
            0.5, 0.5, self._format_stats(),
            transform=self.ax_stats.transAxes,
            ha="center", va="center",
            fontsize=13, color="white",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#3a3a3a", edgecolor="#555555")
        )

        plt.tight_layout(pad=2.0)
        plt.show(block=False)
        plt.pause(0.1)

    # ── Public Method ────────────────────────────────────────────────────────

    def update(self, episode, reward, won, steps):
        """Call this after every episode to update the dashboard."""

        # Update trackers
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.steps_history.append(steps)

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

        if episode % UPDATE_EVERY == 0:
            self._draw()

    def close(self):
        plt.ioff()
        plt.show(block=True)

    # ── Internal Drawing ─────────────────────────────────────────────────────

    def _draw(self):
        ep_range = list(range(len(self.win_rates)))

        # Win rate
        self.ax_winrate.cla()
        self._style_axis(self.ax_winrate,
                         f"{self.agent_name} — Win Rate Over Time",
                         "Episode", "Win Rate %")
        self.ax_winrate.plot(ep_range, list(self.win_rates),
                             color=self.color, linewidth=2)
        self.ax_winrate.fill_between(ep_range, list(self.win_rates),
                                     alpha=0.2, color=self.color)
        self.ax_winrate.set_ylim(0, 100)

        # Reward
        self.ax_reward.cla()
        self._style_axis(self.ax_reward, "Reward Per Episode",
                         "Episode", "Reward")
        colors = [GREEN if r > 0 else RED
                  for r in self.rewards
                  for GREEN, RED in [("#4CAF50", "#F44336")]]
        self.ax_reward.bar(ep_range, list(self.rewards),
                           color=colors, width=0.8)

        # Steps
        self.ax_steps.cla()
        self._style_axis(self.ax_steps, "Steps Per Episode",
                         "Episode", "Steps")
        self.ax_steps.plot(ep_range, list(self.steps_history),
                           color="#FF9800", linewidth=1.5)

        # Stats panel
        self.stats_text.set_text(self._format_stats())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _format_stats(self):
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total > 0 else 0.0
        hs    = f"{self.high_score:.1f}" if self.high_score != float('-inf') else "—"
        return (
            f"  {self.agent_name}  |  "
            f"Episodes: {total}  |  "
            f"Wins: {self.wins}  |  "
            f"Losses: {self.losses}  |  "
            f"Win Rate: {wr:.1f}%  |  "
            f"High Score: {hs}  |  "
            f"Best Streak: {self.best_streak}  "
        )

    def _style_axis(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#2d2d2d")
        ax.set_title(title,  color="white", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
