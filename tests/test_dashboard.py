# tests/test_dashboard.py
# Tests for the dashboard data layer (no display needed)
# Run with: pytest tests/test_dashboard.py -v

import pytest
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We only test the data layer — no window opened
from agent.dashboard import Dashboard, SHARED_FILE, AGENT_STYLES

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def silent_dash():
    """Dashboard with no window — data layer only."""
    dash = Dashboard(agent_name="Random Agent", show_window=False)
    yield dash
    dash.close()
    # Clean up shared file
    if os.path.exists(SHARED_FILE):
        os.remove(SHARED_FILE)

@pytest.fixture
def learning_dash():
    dash = Dashboard(agent_name="DQN Learning Agent", show_window=False)
    yield dash
    dash.close()
    if os.path.exists(SHARED_FILE):
        os.remove(SHARED_FILE)

# ── Data Write/Read Tests ─────────────────────────────────────────────────────

class TestSharedFile:

    def test_shared_file_created_on_init(self, silent_dash):
        assert os.path.exists(SHARED_FILE)

    def test_data_written_after_update(self, silent_dash):
        silent_dash.update(
            episode=1, reward=5.0, won=True,
            steps=10, correct_flags=2, pct_cleared=0.5
        )
        data = silent_dash._read_shared_data()
        assert "Random Agent" in data

    def test_win_tracked_correctly(self, silent_dash):
        silent_dash.update(episode=1, reward=10.0, won=True,
                           steps=5, correct_flags=1, pct_cleared=0.3)
        assert silent_dash.wins == 1
        assert silent_dash.losses == 0

    def test_loss_tracked_correctly(self, silent_dash):
        silent_dash.update(episode=1, reward=-10.0, won=False,
                           steps=5, correct_flags=0, pct_cleared=0.1)
        assert silent_dash.wins == 0
        assert silent_dash.losses == 1

    def test_high_score_updates(self, silent_dash):
        silent_dash.update(episode=1, reward=5.0, won=False,
                           steps=10, correct_flags=0, pct_cleared=0.1)
        silent_dash.update(episode=2, reward=15.0, won=True,
                           steps=8, correct_flags=3, pct_cleared=0.8)
        assert silent_dash.high_score == pytest.approx(15.0)

    def test_win_rate_calculation(self, silent_dash):
        silent_dash.update(episode=1, reward=10.0, won=True,
                           steps=5, correct_flags=1, pct_cleared=0.5)
        silent_dash.update(episode=2, reward=-5.0, won=False,
                           steps=3, correct_flags=0, pct_cleared=0.1)
        silent_dash.update(episode=3, reward=10.0, won=True,
                           steps=7, correct_flags=2, pct_cleared=0.6)
        wr = list(silent_dash.win_rates)[-1]
        assert wr == pytest.approx(66.67, abs=0.1)

    def test_streak_tracking(self, silent_dash):
        for i in range(5):
            silent_dash.update(episode=i+1, reward=10.0, won=True,
                               steps=5, correct_flags=1, pct_cleared=0.5)
        assert silent_dash.best_streak == 5

    def test_streak_resets_on_loss(self, silent_dash):
        for i in range(3):
            silent_dash.update(episode=i+1, reward=10.0, won=True,
                               steps=5, correct_flags=1, pct_cleared=0.5)
        silent_dash.update(episode=4, reward=-5.0, won=False,
                           steps=3, correct_flags=0, pct_cleared=0.1)
        assert silent_dash.current_streak == 0
        assert silent_dash.best_streak == 3

    def test_multiple_agents_in_shared_file(self, silent_dash, learning_dash):
        silent_dash.update(episode=1, reward=5.0, won=False,
                           steps=10, correct_flags=0, pct_cleared=0.1)
        learning_dash.update(episode=1, reward=8.0, won=True,
                              steps=7, correct_flags=2, pct_cleared=0.5)
        data = silent_dash._read_shared_data()
        assert "Random Agent" in data
        assert "DQN Learning Agent" in data

    def test_cleanup_removes_agent(self, silent_dash):
        silent_dash.update(episode=1, reward=5.0, won=False,
                           steps=10, correct_flags=0, pct_cleared=0.1)
        silent_dash.close()
        data = silent_dash._read_shared_data()
        assert "Random Agent" not in data

    def test_no_data_bleed_between_sessions(self):
        """Fresh dashboard should not see previous session data."""
        # Session 1
        dash1 = Dashboard(agent_name="Random Agent", show_window=False)
        dash1.update(episode=1, reward=99.0, won=True,
                     steps=1, correct_flags=0, pct_cleared=1.0)
        dash1.close()

        # Clear file as launcher does between sessions
        if os.path.exists(SHARED_FILE):
            os.remove(SHARED_FILE)

        # Session 2
        dash2 = Dashboard(agent_name="Random Agent", show_window=False)
        assert dash2.wins == 0
        assert dash2.high_score == float("-inf")
        dash2.close()

        if os.path.exists(SHARED_FILE):
            os.remove(SHARED_FILE)

# ── Agent Style Tests ─────────────────────────────────────────────────────────

class TestAgentStyles:

    def test_all_known_agents_have_styles(self):
        for name in ["Random Agent", "DQN Learning Agent", "Human"]:
            assert name in AGENT_STYLES

    def test_styles_have_required_fields(self):
        for name, style in AGENT_STYLES.items():
            assert "color" in style
            assert "label" in style
            assert "dash" in style
            assert len(style["color"]) == 3

# ── Multi-Agent Data Tests ────────────────────────────────────────────────────

class TestMultiAgentData:

    def teardown_method(self):
        """Clean up shared file after each test."""
        if os.path.exists(SHARED_FILE):
            os.remove(SHARED_FILE)

    def test_both_agents_appear_in_shared_file(self):
        """Both agents should write to same shared file."""
        random_dash   = Dashboard(agent_name="Random Agent",
                                  show_window=False)
        learning_dash = Dashboard(agent_name="DQN Learning Agent",
                                  show_window=False)

        random_dash.update(episode=1, reward=5.0, won=False,
                           steps=10, correct_flags=0, pct_cleared=0.1)
        learning_dash.update(episode=1, reward=8.0, won=True,
                             steps=7, correct_flags=2, pct_cleared=0.5)

        data = random_dash._read_shared_data()
        assert "Random Agent" in data
        assert "DQN Learning Agent" in data

        random_dash.close()
        learning_dash.close()

    def test_all_three_agents_appear_in_human_mode(self):
        """Human mode should have three agents in shared file."""
        human_dash    = Dashboard(agent_name="Human",
                                  show_window=False)
        random_dash   = Dashboard(agent_name="Random Agent",
                                  show_window=False)
        learning_dash = Dashboard(agent_name="DQN Learning Agent",
                                  show_window=False)

        human_dash.update(episode=1, reward=12.0, won=True,
                          steps=15, correct_flags=3, pct_cleared=0.8)
        random_dash.update(episode=1, reward=3.0, won=False,
                           steps=8, correct_flags=0, pct_cleared=0.1)
        learning_dash.update(episode=1, reward=7.0, won=False,
                             steps=11, correct_flags=1, pct_cleared=0.3)

        data = human_dash._read_shared_data()
        assert "Human" in data
        assert "Random Agent" in data
        assert "DQN Learning Agent" in data

        human_dash.close()
        random_dash.close()
        learning_dash.close()

    def test_agent_data_has_all_required_chart_keys(self):
        """Every agent entry must have all keys the charts need."""
        dash = Dashboard(agent_name="Random Agent", show_window=False)
        dash.update(episode=1, reward=5.0, won=False,
                    steps=10, correct_flags=2, pct_cleared=0.3)

        data     = dash._read_shared_data()
        agent    = data["Random Agent"]
        required = ["win_rates", "rewards", "steps",
                    "flags", "wins", "losses",
                    "high_score", "best_streak"]

        for key in required:
            assert key in agent, f"Missing chart key: {key}"

        dash.close()

    def test_win_rates_list_correct_length(self):
        """Win rates list should match number of episodes."""
        dash = Dashboard(agent_name="Random Agent", show_window=False)

        for i in range(10):
            dash.update(episode=i+1, reward=float(i),
                        won=i % 3 == 0, steps=i+5,
                        correct_flags=0, pct_cleared=0.1)

        data = dash._read_shared_data()
        assert len(data["Random Agent"]["win_rates"]) == 10
        assert len(data["Random Agent"]["rewards"])   == 10
        assert len(data["Random Agent"]["steps"])     == 10

        dash.close()

    def test_agents_independent_stats(self):
        """Each agent should track its own wins independently."""
        random_dash   = Dashboard(agent_name="Random Agent",
                                  show_window=False)
        learning_dash = Dashboard(agent_name="DQN Learning Agent",
                                  show_window=False)

        # Random wins 1 out of 3
        for i in range(3):
            random_dash.update(episode=i+1, reward=1.0,
                               won=(i == 0), steps=5,
                               correct_flags=0, pct_cleared=0.1)

        # Learning wins 3 out of 3
        for i in range(3):
            learning_dash.update(episode=i+1, reward=10.0,
                                 won=True, steps=5,
                                 correct_flags=2, pct_cleared=0.8)

        data = random_dash._read_shared_data()

        assert data["Random Agent"]["wins"]         == 1
        assert data["Random Agent"]["losses"]       == 2
        assert data["DQN Learning Agent"]["wins"]   == 3
        assert data["DQN Learning Agent"]["losses"] == 0

        random_dash.close()
        learning_dash.close()

    def test_second_agent_does_not_overwrite_first(self):
        """Writing second agent data should not erase first agent."""
        random_dash   = Dashboard(agent_name="Random Agent",
                                  show_window=False)
        learning_dash = Dashboard(agent_name="DQN Learning Agent",
                                  show_window=False)

        random_dash.update(episode=1, reward=99.0, won=True,
                           steps=5, correct_flags=5, pct_cleared=1.0)

        # Now learning agent writes — random data should still be there
        learning_dash.update(episode=1, reward=1.0, won=False,
                             steps=3, correct_flags=0, pct_cleared=0.1)

        data = random_dash._read_shared_data()
        assert "Random Agent" in data
        assert data["Random Agent"]["wins"] == 1
        assert data["Random Agent"]["high_score"] == pytest.approx(99.0)

        random_dash.close()
        learning_dash.close()

    def test_human_agent_style_exists(self):
        """Human agent must have a style defined for charts to render."""
        from agent.dashboard import AGENT_STYLES
        assert "Human" in AGENT_STYLES
        assert "color" in AGENT_STYLES["Human"]
        assert AGENT_STYLES["Human"]["color"] == (255, 230, 109)
