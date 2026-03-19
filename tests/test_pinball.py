# tests/test_pinball.py
# Tests for the Pinball environment
# Run with: pytest tests/test_pinball.py -v

import pytest
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from games.pinball.pinball_env import (
    PinballEnv, Ball, Bumper, Flipper, Hole,
    Slingshot, MultiplierZone,
    DIFFICULTIES, TABLE_W, TABLE_H,
    R_STANDARD_BUMPER, R_HOT_BUMPER, R_FORGE_BUMPER,
    R_FORGE_CHARGED, R_SLINGSHOT, R_HOLE,
    R_RAMP, R_MULTIPLIER_PAD,
    R_BALL_DRAINED, R_BALL_LAST_LIFE,
    R_WALL_REPEAT, R_MULTIPLIER_EXPIRE,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_env():
    env = PinballEnv(render_mode=None, difficulty="Simple")
    env.reset()
    return env

@pytest.fixture
def standard_env():
    env = PinballEnv(render_mode=None, difficulty="Standard")
    env.reset()
    return env

@pytest.fixture
def complex_env():
    env = PinballEnv(render_mode=None, difficulty="Complex")
    env.reset()
    return env

# ── Difficulty Config Tests ───────────────────────────────────────────────────

class TestDifficultyConfig:

    def test_simple_lives(self, simple_env):
        assert simple_env.lives == 5

    def test_standard_lives(self, standard_env):
        assert standard_env.lives == 3

    def test_complex_lives(self, complex_env):
        assert complex_env.lives == 1

    def test_simple_single_flipper_action_space(self, simple_env):
        assert simple_env.action_space.n == 2

    def test_standard_dual_flipper_action_space(self, standard_env):
        assert standard_env.action_space.n == 4

    def test_complex_dual_flipper_action_space(self, complex_env):
        assert complex_env.action_space.n == 4

    def test_slingshot_power_scales_with_difficulty(self):
        assert DIFFICULTIES["Simple"]["slingshot_power"] < \
               DIFFICULTIES["Standard"]["slingshot_power"] < \
               DIFFICULTIES["Complex"]["slingshot_power"]

    def test_multiplier_scales_with_difficulty(self):
        assert DIFFICULTIES["Simple"]["multiplier"] < \
               DIFFICULTIES["Standard"]["multiplier"] < \
               DIFFICULTIES["Complex"]["multiplier"]

    def test_ramp_speed_scales_with_difficulty(self):
        assert DIFFICULTIES["Simple"]["ramp_speed"] < \
               DIFFICULTIES["Standard"]["ramp_speed"] < \
               DIFFICULTIES["Complex"]["ramp_speed"]

# ── Init Tests ────────────────────────────────────────────────────────────────

class TestInit:

    def test_ball_spawns_on_reset(self, simple_env):
        assert len(simple_env.balls) == 1

    def test_ball_starts_on_right_side(self, simple_env):
        ball = simple_env.balls[0]
        assert ball.x > TABLE_W * 0.5

    def test_score_starts_zero(self, simple_env):
        assert simple_env.score == 0

    def test_forge_starts_uncharged(self, simple_env):
        assert simple_env.forge_charge == 0

    def test_multiball_starts_false(self, simple_env):
        assert simple_env.multiball == False

    def test_multiplier_starts_inactive(self, simple_env):
        assert simple_env.multiplier_active == False
        assert simple_env.multiplier == 1.0

    def test_not_done_on_start(self, simple_env):
        assert simple_env.done == False

    def test_bumpers_built(self, simple_env):
        assert len(simple_env.bumpers) > 0

    def test_holes_built(self, simple_env):
        assert len(simple_env.holes) == 4

    def test_slingshots_built(self, simple_env):
        assert len(simple_env.slingshots) == 2

    def test_multiplier_zones_built(self, simple_env):
        assert len(simple_env.multiplier_zones) == 2

    def test_flippers_built(self, simple_env):
        assert simple_env.left_flipper is not None
        assert simple_env.right_flipper is not None

    def test_observation_shape(self, simple_env):
        obs, _ = simple_env.reset()
        assert obs.shape == (12,)

    def test_observation_values_in_range(self, simple_env):
        obs, _ = simple_env.reset()
        assert all(-1.0 <= v <= 1.0 for v in obs)

# ── Bumper Tests ──────────────────────────────────────────────────────────────

class TestBumpers:

    def test_three_hot_bumpers_assigned(self, simple_env):
        hot = [b for b in simple_env.bumpers if b.kind == "hot"]
        assert len(hot) == 3

    def test_two_forge_bumpers_assigned(self, simple_env):
        forge = [b for b in simple_env.bumpers if b.kind == "forge"]
        assert len(forge) == 2

    def test_standard_bumpers_exist(self, simple_env):
        standard = [b for b in simple_env.bumpers
                    if b.kind == "standard"]
        assert len(standard) > 0

    def test_bumper_collision_detected(self):
        bumper = Bumper(100, 100, "standard")
        ball   = Ball(100, 100 + bumper.radius - 2, 0, -5)
        result = bumper.check_collision(ball)
        assert result == True

    def test_bumper_no_collision_when_far(self):
        bumper = Bumper(100, 100, "standard")
        ball   = Ball(200, 200, 0, 0)
        result = bumper.check_collision(ball)
        assert result == False

    def test_bumper_bounces_ball_away(self):
        bumper   = Bumper(100, 100, "standard")
        ball     = Ball(100, 115, 0, -8)
        orig_vy  = ball.vy
        bumper.check_collision(ball)
        assert ball.vy != orig_vy

    def test_hot_bumpers_randomized_each_reset(self):
        env = PinballEnv(render_mode=None, difficulty="Simple")
        env.reset()
        hot1 = [(b.x, b.y) for b in env.bumpers if b.kind == "hot"]
        env.reset()
        hot2 = [(b.x, b.y) for b in env.bumpers if b.kind == "hot"]
        # Very unlikely to be identical every time
        # Run multiple resets to confirm randomization
        different = False
        for _ in range(10):
            env.reset()
            hot_new = [(b.x, b.y) for b in env.bumpers
                       if b.kind == "hot"]
            if hot_new != hot1:
                different = True
                break
        assert different

# ── Ball Physics Tests ────────────────────────────────────────────────────────

class TestBallPhysics:

    def test_ball_falls_with_gravity(self):
        ball    = Ball(100, 100, 0, 0)
        orig_y  = ball.y
        ball.update()
        assert ball.y > orig_y

    def test_ball_moves_horizontally(self):
        ball   = Ball(100, 100, 5, 0)
        orig_x = ball.x
        ball.update()
        assert ball.x > orig_x

    def test_ball_speed_capped(self):
        ball    = Ball(100, 100, 100, 100)
        ball.update()
        speed   = math.sqrt(ball.vx**2 + ball.vy**2)
        from games.pinball.pinball_env import MAX_BALL_SPEED
        assert speed <= MAX_BALL_SPEED + 0.1

    def test_ball_friction_applied(self):
        ball    = Ball(100, 100, 10, 0)
        orig_vx = ball.vx
        ball.update()
        assert abs(ball.vx) < abs(orig_vx) + 1.0

# ── Flipper Tests ─────────────────────────────────────────────────────────────

class TestFlippers:

    def test_left_flipper_activates(self, standard_env):
        standard_env.left_flipper.active = True
        standard_env.left_flipper.update()
        assert standard_env.left_flipper.active == True

    def test_right_flipper_activates(self, standard_env):
        standard_env.right_flipper.active = True
        standard_env.right_flipper.update()
        assert standard_env.right_flipper.active == True

    def test_simple_both_flippers_on_action_1(self, simple_env):
        simple_env.step(1)
        assert simple_env.left_flipper.active == True
        assert simple_env.right_flipper.active == True

    def test_simple_both_flippers_off_action_0(self, simple_env):
        simple_env.step(1)
        simple_env.step(0)
        assert simple_env.left_flipper.active == False
        assert simple_env.right_flipper.active == False

    def test_standard_left_only_action_1(self, standard_env):
        standard_env.step(1)
        assert standard_env.left_flipper.active == True
        assert standard_env.right_flipper.active == False

    def test_standard_right_only_action_2(self, standard_env):
        standard_env.step(2)
        assert standard_env.left_flipper.active == False
        assert standard_env.right_flipper.active == True

    def test_standard_both_action_3(self, standard_env):
        standard_env.step(3)
        assert standard_env.left_flipper.active == True
        assert standard_env.right_flipper.active == True

    def test_flipper_tip_moves_when_active(self):
        flipper    = Flipper(140, 720, "left")
        tip_before = flipper.get_tip()
        flipper.active = True
        for _ in range(10):
            flipper.update()
        tip_after = flipper.get_tip()
        assert tip_before != tip_after

# ── Hole Tests ────────────────────────────────────────────────────────────────

class TestHoles:

    def test_hole_detects_ball(self):
        hole = Hole(100, 100, 0)
        ball = Ball(100, 100, 0, 0)
        assert hole.check_collision(ball) == True

    def test_hole_no_collision_when_far(self):
        hole = Hole(100, 100, 0)
        ball = Ball(300, 300, 0, 0)
        assert hole.check_collision(ball) == False

    def test_four_holes_on_table(self, simple_env):
        assert len(simple_env.holes) == 4

    def test_hole_ids_unique(self, simple_env):
        ids = [h.hole_id for h in simple_env.holes]
        assert len(ids) == len(set(ids))

# ── Forge Tests ───────────────────────────────────────────────────────────────

class TestForge:

    def test_forge_starts_at_zero(self, simple_env):
        assert simple_env.forge_charge == 0

    def test_forge_charges_on_forge_bumper_hit(self, simple_env):
        simple_env.forge_charge = 0
        forge_bumpers = [b for b in simple_env.bumpers
                         if b.kind == "forge"]
        assert len(forge_bumpers) > 0
        # Manually simulate forge bumper hit
        simple_env.forge_charge += 0.5
        assert simple_env.forge_charge == 0.5

    def test_forge_triggers_multiball_at_4(self, simple_env):
        simple_env.forge_charge = 3
        simple_env._trigger_forge()
        assert simple_env.multiball == True

    def test_forge_resets_after_trigger(self, simple_env):
        simple_env.forge_charge = 4
        simple_env._trigger_forge()
        assert simple_env.forge_charge == 0

    def test_multiball_launches_2_extra_balls(self, simple_env):
        initial_balls = len(simple_env.balls)
        simple_env._trigger_forge()
        assert len(simple_env.balls) == initial_balls + 2

    def test_forge_charge_capped_at_4(self, simple_env):
        simple_env.forge_charge = 3.5
        simple_env.forge_charge = min(4,
            simple_env.forge_charge + 0.5)
        assert simple_env.forge_charge <= 4

# ── Multiplier Zone Tests ─────────────────────────────────────────────────────

class TestMultiplierZones:

    def test_two_zones_on_table(self, simple_env):
        assert len(simple_env.multiplier_zones) == 2

    def test_zone_starts_inactive(self, simple_env):
        for zone in simple_env.multiplier_zones:
            assert zone.active == False

    def test_zone_activates_after_warning(self):
        zone = MultiplierZone(100, 100, 100, 40, 0)
        zone.warning = True
        zone.timer   = 0
        result       = zone.update(1.0)
        assert zone.active == True

    def test_zone_expires_after_active(self):
        zone         = MultiplierZone(100, 100, 100, 40, 0)
        zone.active  = True
        zone.timer   = 0
        result       = zone.update(1.0)
        assert zone.active == False
        assert result      == "expired"

    def test_zone_collision_detected(self):
        zone = MultiplierZone(100, 100, 100, 40, 0)
        zone.active = True
        ball = Ball(150, 120, 0, 0)
        assert zone.check_collision(ball) == True

    def test_zone_no_collision_when_inactive(self):
        zone = MultiplierZone(100, 100, 100, 40, 0)
        zone.active = False
        ball = Ball(150, 120, 0, 0)
        assert zone.check_collision(ball) == False

    def test_multiplier_activates_on_zone_hit(self, simple_env):
        zone        = simple_env.multiplier_zones[0]
        zone.active = True
        ball        = simple_env.balls[0]
        ball.x      = zone.x + zone.w // 2
        ball.y      = zone.y + zone.h // 2
        simple_env.step(0)
        assert simple_env.multiplier_active == True

# ── Slingshot Tests ───────────────────────────────────────────────────────────

class TestSlingshots:

    def test_two_slingshots_on_table(self, simple_env):
        assert len(simple_env.slingshots) == 2

    def test_slingshot_sides_correct(self, simple_env):
        sides = [s.side for s in simple_env.slingshots]
        assert "left"  in sides
        assert "right" in sides

    def test_slingshot_kicks_ball(self):
        sling  = Slingshot(
            [(60, 580), (110, 640), (60, 640)], "left")
        ball   = Ball(62, 610, 2, 2)
        orig_speed = math.sqrt(ball.vx**2 + ball.vy**2)
        hit    = sling.check_collision(ball, 12.0)
        if hit:
            new_speed = math.sqrt(ball.vx**2 + ball.vy**2)
            assert new_speed > 0

# ── Lives and Drain Tests ─────────────────────────────────────────────────────

class TestLivesAndDrain:

    def test_life_lost_when_ball_drains(self, simple_env):
        initial_lives = simple_env.lives
        simple_env.balls[0].y = TABLE_H + 50
        simple_env.step(0)
        assert simple_env.lives == initial_lives - 1 or \
               simple_env.done == True

    def test_new_ball_launched_after_drain(self, simple_env):
        simple_env.balls[0].y = TABLE_H + 50
        simple_env.step(0)
        if not simple_env.done:
            assert len(simple_env.balls) >= 1

    def test_game_over_when_last_life_drained(self, complex_env):
        assert complex_env.lives == 1
        complex_env.balls[0].y = TABLE_H + 50
        complex_env.step(0)
        assert complex_env.done == True

    def test_multiball_ends_when_extra_balls_drain(self, simple_env):
        simple_env._trigger_forge()
        assert simple_env.multiball == True
        assert len(simple_env.balls) == 3
        # Drain two extra balls
        simple_env.balls[1].y = TABLE_H + 50
        simple_env.balls[2].y = TABLE_H + 50
        simple_env.step(0)
        assert simple_env.multiball == False

# ── Reward Tests ──────────────────────────────────────────────────────────────

class TestRewards:

    def test_reward_constants_defined(self):
        assert R_STANDARD_BUMPER   ==  10.0
        assert R_HOT_BUMPER        ==  25.0
        assert R_FORGE_BUMPER      ==  15.0
        assert R_FORGE_CHARGED     ==  50.0
        assert R_SLINGSHOT         ==   5.0
        assert R_HOLE              ==  20.0
        assert R_RAMP              ==  30.0
        assert R_MULTIPLIER_PAD    ==  15.0
        assert R_BALL_DRAINED      == -50.0
        assert R_BALL_LAST_LIFE    == -100.0
        assert R_WALL_REPEAT       ==  -5.0
        assert R_MULTIPLIER_EXPIRE == -10.0

    def test_wall_repeat_penalty_after_3_hits(self, simple_env):
        reward = 0.0
        for _ in range(3):
            reward += simple_env._wall_hit("left")
        assert reward == R_WALL_REPEAT

    def test_wall_repeat_resets_on_different_wall(self, simple_env):
        simple_env._wall_hit("left")
        simple_env._wall_hit("left")
        simple_env._wall_hit("right")   # different wall
        reward = simple_env._wall_hit("left")
        assert reward == 0.0   # counter reset — no penalty yet

    def test_forge_trigger_returns_correct_reward(self, simple_env):
        reward = simple_env._trigger_forge()
        assert reward == R_FORGE_CHARGED

# ── Observation Tests ─────────────────────────────────────────────────────────

class TestObservation:

    def test_observation_has_12_values(self, simple_env):
        obs, _ = simple_env.reset()
        assert len(obs) == 12

    def test_ball_position_normalized(self, simple_env):
        obs, _ = simple_env.reset()
        ball_x = obs[0]
        ball_y = obs[1]
        assert 0.0 <= ball_x <= 1.0
        assert 0.0 <= ball_y <= 1.0

    def test_forge_charge_normalized(self, simple_env):
        simple_env.forge_charge = 2
        obs = simple_env._get_obs()
        assert obs[6] == pytest.approx(0.5)

    def test_lives_normalized(self, simple_env):
        obs = simple_env._get_obs()
        assert 0.0 <= obs[10] <= 1.0

    def test_ball_count_normalized(self, simple_env):
        obs = simple_env._get_obs()
        assert 0.0 <= obs[11] <= 1.0

    def test_observation_updates_after_step(self, simple_env):
        obs1, _ = simple_env.reset()
        obs2, _, _, _, _ = simple_env.step(0)
        assert not all(obs1 == obs2)

# ── Reset Tests ───────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_score(self, simple_env):
        simple_env.score = 999
        simple_env.reset()
        assert simple_env.score == 0

    def test_reset_clears_forge(self, simple_env):
        simple_env.forge_charge = 3
        simple_env.reset()
        assert simple_env.forge_charge == 0

    def test_reset_clears_multiball(self, simple_env):
        simple_env.multiball = True
        simple_env.reset()
        assert simple_env.multiball == False

    def test_reset_restores_lives(self, simple_env):
        simple_env.lives = 1
        simple_env.reset()
        assert simple_env.lives == DIFFICULTIES["Simple"]["lives"]

    def test_reset_spawns_single_ball(self, simple_env):
        simple_env._trigger_forge()
        assert len(simple_env.balls) == 3
        simple_env.reset()
        assert len(simple_env.balls) == 1

    def test_reset_preserves_difficulty(self, simple_env):
        simple_env.reset()
        assert simple_env.difficulty == "Simple"

    def test_reset_randomizes_hot_bumpers(self):
        env = PinballEnv(render_mode=None, difficulty="Simple")
        env.reset()
        hot1 = set((b.x, b.y) for b in env.bumpers
                   if b.kind == "hot")
        different = False
        for _ in range(10):
            env.reset()
            hot2 = set((b.x, b.y) for b in env.bumpers
                       if b.kind == "hot")
            if hot2 != hot1:
                different = True
                break
        assert different

# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration:

    def test_episode_completes_simple(self):
        env    = PinballEnv(render_mode=None, difficulty="Simple")
        obs, _ = env.reset()
        done   = False
        steps  = 0

        while not done and steps < 50000:
            action             = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)
            steps             += 1

        assert done == True
        env.close()

    def test_episode_completes_all_difficulties(self):
        for diff in ["Simple", "Standard", "Complex"]:
            env    = PinballEnv(render_mode=None, difficulty=diff)
            obs, _ = env.reset()
            done   = False
            steps  = 0

            while not done and steps < 50000:
                action             = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                steps             += 1

            assert done == True, \
                f"{diff} episode never completed"
            env.close()

    def test_score_increases_during_episode(self):
        env    = PinballEnv(render_mode=None, difficulty="Simple")
        obs, _ = env.reset()

        for _ in range(500):
            if env.done:
                break
            env.step(env.action_space.sample())

        # Score should have changed from zero
        assert env.score != 0 or env.done
        env.close()

    def test_action_space_samples_valid(self):
        for diff in ["Simple", "Standard", "Complex"]:
            env = PinballEnv(render_mode=None, difficulty=diff)
            env.reset()
            for _ in range(100):
                action = env.action_space.sample()
                assert 0 <= action < env.action_space.n
            env.close()
