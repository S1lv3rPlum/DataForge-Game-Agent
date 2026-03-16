# pinball_env.py
# DataForge Pinball Environment — Version 1.0
# Full pinball simulation with:
#   - The Forge multi-ball trigger
#   - Cerulean hot bumpers
#   - Teleport holes
#   - Left side ramp
#   - Temporary multiplier zones
#   - Slingshots scaling with difficulty
#   - Lives system
#   - Rules overlay
#   - Human and agent play modes

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys
import math
import random
import time

# ── Difficulty Configurations ─────────────────────────────────────────────────
DIFFICULTIES = {
    "Simple": {
        "lives":            5,
        "slingshot_power":  8.0,
        "ramp_speed":       4.0,
        "multiplier":       1.5,
        "dual_flipper":     False,
        "desc":             "5 Lives • Single Flipper • 1.5x Multiplier"
    },
    "Standard": {
        "lives":            3,
        "slingshot_power":  12.0,
        "ramp_speed":       6.0,
        "multiplier":       2.0,
        "dual_flipper":     True,
        "desc":             "3 Lives • Dual Flippers • 2x Multiplier"
    },
    "Complex": {
        "lives":            1,
        "slingshot_power":  16.0,
        "ramp_speed":       8.0,
        "multiplier":       3.0,
        "dual_flipper":     True,
        "desc":             "1 Life • Dual Flippers • 3x Multiplier"
    },
}

# ── Display ───────────────────────────────────────────────────────────────────
TABLE_W     = 500
TABLE_H     = 800
PANEL_W     = 200
WINDOW_W    = TABLE_W + PANEL_W
WINDOW_H    = TABLE_H

# ── Colors ────────────────────────────────────────────────────────────────────
BLACK       = (0,     0,   0)
WHITE       = (255, 255, 255)
GRAY        = (180, 180, 180)
DARK_GRAY   = (60,   60,  80)
BG_COLOR    = (15,   15,  25)
PANEL_BG    = (20,   20,  35)
CERULEAN    = (0,   123, 167)
CERULEAN_LT = (78,  205, 196)
ORANGE      = (255, 140,   0)
YELLOW      = (255, 230,  50)
RED         = (200,  30,  30)
GREEN       = (30,  180,  30)
GOLD        = (255, 200,   0)

# Forge charge colors
FORGE_COLORS = [
    (80,   80,  80),   # 0 hits — dormant gray
    (180,  80,  20),   # 1 hit  — dim orange
    (230, 140,  20),   # 2 hits — bright orange
    (255, 220,  50),   # 3 hits — yellow
    (255, 255, 255),   # 4 hits — white hot
]

# ── Physics Constants ─────────────────────────────────────────────────────────
GRAVITY         = 0.25
BALL_RADIUS     = 10
FLIPPER_LENGTH  = 70
BALL_FRICTION   = 0.995
WALL_BOUNCE     = 0.75
BUMPER_BOUNCE   = 1.2
MAX_BALL_SPEED  = 18.0

# ── Rewards ───────────────────────────────────────────────────────────────────
R_STANDARD_BUMPER   =  10.0
R_HOT_BUMPER        =  25.0
R_FORGE_BUMPER      =  15.0
R_FORGE_CHARGED     =  50.0
R_SLINGSHOT         =   5.0
R_HOLE              =  20.0
R_RAMP              =  30.0
R_MULTIPLIER_PAD    =  15.0
R_BALL_DRAINED      = -50.0
R_BALL_LAST_LIFE    = -100.0
R_WALL_REPEAT       =  -5.0
R_MULTIPLIER_EXPIRE = -10.0

# ── Rules Text ────────────────────────────────────────────────────────────────
RULES = [
    ("Standard bumper",      "+10"),
    ("Hot bumper",           "+25"),
    ("Forge bumper",         "+15 + charge"),
    ("Forge fully charged",  "+50"),
    ("Slingshot",            "+5"),
    ("Hole entry",           "+20"),
    ("Ramp completed",       "+30"),
    ("Multiplier pad",       "+15 + multiplier"),
    ("Multi-ball active",    "all x2"),
    ("Ball drained",         "-50"),
    ("Last life drained",    "-100"),
    ("Same wall x3",         "-5"),
    ("Multiplier expired",   "-10"),
]

CONTROLS_SIMPLE   = [("Spacebar", "Both flippers")]
CONTROLS_STANDARD = [
    ("← Left arrow",  "Left flipper"),
    ("→ Right arrow", "Right flipper"),
]


class Ball:
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.x  = float(x)
        self.y  = float(y)
        self.vx = float(vx)
        self.vy = float(vy)

    def update(self):
        self.vy += GRAVITY
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > MAX_BALL_SPEED:
            scale   = MAX_BALL_SPEED / speed
            self.vx *= scale
            self.vy *= scale
        self.vx *= BALL_FRICTION
        self.x  += self.vx
        self.y  += self.vy

    def pos(self):
        return (int(self.x), int(self.y))


class Bumper:
    def __init__(self, x, y, kind="standard"):
        self.x       = x
        self.y       = y
        self.kind    = kind   # standard, hot, forge
        self.radius  = 18
        self.hit_timer = 0    # flash on hit

    def check_collision(self, ball):
        dx   = ball.x - self.x
        dy   = ball.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < self.radius + BALL_RADIUS:
            if dist == 0:
                dist = 0.001
            nx      = dx / dist
            ny      = dy / dist
            speed   = math.sqrt(ball.vx**2 + ball.vy**2)
            ball.vx = nx * speed * BUMPER_BOUNCE
            ball.vy = ny * speed * BUMPER_BOUNCE
            ball.x  = self.x + nx * (self.radius + BALL_RADIUS + 1)
            ball.y  = self.y + ny * (self.radius + BALL_RADIUS + 1)
            self.hit_timer = 8
            return True
        return False


class Flipper:
    def __init__(self, x, y, side="left"):
        self.x        = x
        self.y        = y
        self.side     = side
        self.angle    = 30 if side == "left" else 150   # resting angle
        self.up_angle = -20 if side == "left" else 200  # activated angle
        self.active   = False
        self.length   = FLIPPER_LENGTH

    def get_tip(self):
        angle_rad = math.radians(self.angle)
        tx = self.x + self.length * math.cos(angle_rad)
        ty = self.y + self.length * math.sin(angle_rad)
        return (tx, ty)

    def update(self):
        target = self.up_angle if self.active else (
            30 if self.side == "left" else 150)
        diff   = target - self.angle
        self.angle += diff * 0.35

    def check_collision(self, ball):
        """Simple flipper collision — check if ball near flipper line."""
        tip    = self.get_tip()
        dx     = tip[0] - self.x
        dy     = tip[1] - self.y
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return False
        nx  = dx / length
        ny  = dy / length
        px  = ball.x - self.x
        py  = ball.y - self.y
        dot = px * nx + py * ny
        dot = max(0, min(length, dot))
        cx  = self.x + dot * nx
        cy  = self.y + dot * ny
        dist = math.sqrt((ball.x - cx)**2 + (ball.y - cy)**2)
        if dist < BALL_RADIUS + 6:
            # Reflect ball
            perp_x = -(ny)
            perp_y = nx
            speed  = math.sqrt(ball.vx**2 + ball.vy**2)
            speed  = max(speed, 8.0)
            if self.active:
                speed = min(speed * 1.4, MAX_BALL_SPEED)
            dot_v  = ball.vx * perp_x + ball.vy * perp_y
            ball.vx = perp_x * abs(dot_v) * (
                1 if self.side == "left" else -1) * -1
            ball.vy = -abs(speed)
            ball.y  = cy - BALL_RADIUS - 7
            return True
        return False


class Hole:
    def __init__(self, x, y, hole_id):
        self.x        = x
        self.y        = y
        self.hole_id  = hole_id
        self.radius   = 16
        self.pulse    = 0

    def check_collision(self, ball):
        dx   = ball.x - self.x
        dy   = ball.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        return dist < self.radius


class Slingshot:
    def __init__(self, points, side):
        self.points    = points   # list of (x,y) triangle vertices
        self.side      = side
        self.hit_timer = 0

    def check_collision(self, ball, power):
        """Check if ball hits slingshot triangle."""
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            dx  = p2[0] - p1[0]
            dy  = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            if length == 0:
                continue
            nx  = dx / length
            ny  = dy / length
            px  = ball.x - p1[0]
            py  = ball.y - p1[1]
            dot = px * nx + py * ny
            dot = max(0, min(length, dot))
            cx  = p1[0] + dot * nx
            cy  = p1[1] + dot * ny
            dist = math.sqrt((ball.x - cx)**2 + (ball.y - cy)**2)
            if dist < BALL_RADIUS + 4:
                # Kick ball away from slingshot
                away_x = ball.x - cx
                away_y = ball.y - cy
                away_d = math.sqrt(away_x**2 + away_y**2)
                if away_d == 0:
                    away_d = 0.001
                ball.vx = (away_x / away_d) * power
                ball.vy = (away_y / away_d) * power
                self.hit_timer = 8
                return True
        return False


class MultiplierZone:
    def __init__(self, x, y, w, h, zone_id):
        self.x       = x
        self.y       = y
        self.w       = w
        self.h       = h
        self.zone_id = zone_id
        self.active  = False
        self.warning = False
        self.timer   = random.uniform(8, 20)
        self.pulse   = 0

    def update(self, dt):
        self.timer -= dt
        if self.timer <= 0:
            if self.active:
                self.active  = False
                self.warning = False
                self.timer   = random.uniform(8, 20)
                return "expired"
            elif self.warning:
                self.warning = False
                self.active  = True
                self.timer   = random.uniform(5, 15)
            else:
                self.warning = True
                self.timer   = 1.5   # warning phase duration
        self.pulse = (self.pulse + 0.15) % (2 * math.pi)
        return None

    def check_collision(self, ball):
        return (self.active and
                self.x <= ball.x <= self.x + self.w and
                self.y <= ball.y <= self.y + self.h)


class PinballEnv(gym.Env):
    """
    DataForge Pinball environment.

    Action space:
        Simple   (discrete 2): 0=nothing, 1=both flippers
        Standard (discrete 4): 0=nothing, 1=left, 2=right, 3=both

    Observation:
        Ball position (x, y) normalized 0-1
        Ball velocity (vx, vy) normalized
        Flipper states (left_active, right_active)
        Forge charge (0-4)
        Active multiplier (0 or multiplier value)
        Multiplier zone states (2 values)
        Lives remaining
        Ball count (1-3 during multiball)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, difficulty=None):
        super().__init__()

        self.render_mode = render_mode
        self.difficulty  = difficulty

        self.window  = None
        self.clock   = None
        self.font_lg = None
        self.font_md = None
        self.font_sm = None

        # Observation: 12 values
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # Placeholder — updated after difficulty chosen
        self.action_space = spaces.Discrete(2)

        # Game state
        self.balls           = []
        self.score           = 0
        self.lives           = 3
        self.done            = False
        self.forge_charge    = 0
        self.multiball       = False
        self.multiplier      = 1.0
        self.multiplier_active = False
        self.multiplier_timer  = 0
        self.wall_hit_count    = 0
        self.last_wall         = None
        self.show_rules        = False
        self.total_reward      = 0.0

        # Table elements
        self.bumpers          = []
        self.holes            = []
        self.slingshots       = []
        self.multiplier_zones = []
        self.left_flipper     = None
        self.right_flipper    = None
        self.ramp_active      = False
        self.ramp_ball        = None

    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.difficulty is None:
            if self.render_mode == "human":
                self._init_pygame_basic()
                self.difficulty = self._difficulty_screen()
            else:
                self.difficulty = "Simple"

        cfg = DIFFICULTIES[self.difficulty]

        # Update action space
        if cfg["dual_flipper"]:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(2)

        self.lives             = cfg["lives"]
        self.score             = 0
        self.done              = False
        self.forge_charge      = 0
        self.multiball         = False
        self.multiplier        = 1.0
        self.multiplier_active = False
        self.multiplier_timer  = 0
        self.wall_hit_count    = 0
        self.last_wall         = None
        self.total_reward      = 0.0

        self._build_table()
        self._launch_ball()

        if self.render_mode == "human":
            self._init_pygame()

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        cfg     = DIFFICULTIES[self.difficulty]
        reward  = 0.0
        dt      = 1.0 / 60.0

        # Apply flipper actions
        if cfg["dual_flipper"]:
            self.left_flipper.active  = action in (1, 3)
            self.right_flipper.active = action in (2, 3)
        else:
            self.left_flipper.active  = action == 1
            self.right_flipper.active = action == 1

        self.left_flipper.update()
        self.right_flipper.update()

        # Update multiplier zones
        for zone in self.multiplier_zones:
            result = zone.update(dt)
            if result == "expired" and not self._any_ball_in_zone(zone):
                reward += R_MULTIPLIER_EXPIRE

        # Update multiplier timer
        if self.multiplier_active:
            self.multiplier_timer -= dt
            if self.multiplier_timer <= 0:
                self.multiplier_active = False
                self.multiplier        = 1.0

        # Update all balls
        balls_to_remove = []
        for ball in self.balls:
            ball_reward = self._update_ball(ball, cfg)
            reward     += ball_reward

            # Check drain
            if ball.y > TABLE_H + BALL_RADIUS:
                balls_to_remove.append(ball)

        # Remove drained balls
        for ball in balls_to_remove:
            self.balls.remove(ball)
            if len(self.balls) == 0 or \
               (self.multiball and len(self.balls) == 1):
                if self.multiball and len(self.balls) == 1:
                    self.multiball         = False
                    self.multiplier_active = False
                    self.multiplier        = 1.0
                else:
                    self.lives -= 1
                    if self.lives <= 0:
                        reward   += R_BALL_LAST_LIFE
                        self.done = True
                    else:
                        reward += R_BALL_DRAINED
                        self._launch_ball()

        # Apply score
        scored          = reward * self.multiplier \
                          if self.multiplier_active else reward
        self.score     += max(0, scored)
        self.total_reward += reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if self.window is None:
            self._init_pygame()

        self.window.fill(BG_COLOR)
        self._draw_table()
        self._draw_panel()

        if self.show_rules:
            self._draw_rules_overlay()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

    # ── Ball Update ───────────────────────────────────────────────────────────

    def _update_ball(self, ball, cfg):
        reward = 0.0
        ball.update()

        # Wall collisions
        if ball.x - BALL_RADIUS < 0:
            ball.x   = BALL_RADIUS
            ball.vx  = abs(ball.vx) * WALL_BOUNCE
            reward  += self._wall_hit("left")
        if ball.x + BALL_RADIUS > TABLE_W:
            ball.x   = TABLE_W - BALL_RADIUS
            ball.vx  = -abs(ball.vx) * WALL_BOUNCE
            reward  += self._wall_hit("right")
        if ball.y - BALL_RADIUS < 0:
            ball.y   = BALL_RADIUS
            ball.vy  = abs(ball.vy) * WALL_BOUNCE
            reward  += self._wall_hit("top")

        # Top corner angles — redirect toward center
        if ball.y < 120 and ball.x < 80:
            ball.vx = abs(ball.vx)
            ball.vy = abs(ball.vy)
        if ball.y < 120 and ball.x > TABLE_W - 80:
            ball.vx = -abs(ball.vx)
            ball.vy = abs(ball.vy)

        # Bumper collisions
        for bumper in self.bumpers:
            if bumper.check_collision(ball):
                if bumper.kind == "hot":
                    reward += R_HOT_BUMPER
                elif bumper.kind == "forge":
                    reward         += R_FORGE_BUMPER
                    self.forge_charge = min(4,
                        self.forge_charge + 0.5)
                    if self.forge_charge >= 4:
                        reward += self._trigger_forge()
                else:
                    reward += R_STANDARD_BUMPER

        # Direct Forge hit
        forge_x, forge_y = TABLE_W // 2, 180
        dx   = ball.x - forge_x
        dy   = ball.y - forge_y
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 30 + BALL_RADIUS:
            if dist == 0:
                dist = 0.001
            nx      = dx / dist
            ny      = dy / dist
            speed   = math.sqrt(ball.vx**2 + ball.vy**2)
            ball.vx = nx * speed * BUMPER_BOUNCE
            ball.vy = ny * speed * BUMPER_BOUNCE
            ball.x  = forge_x + nx * (30 + BALL_RADIUS + 1)
            ball.y  = forge_y + ny * (30 + BALL_RADIUS + 1)
            self.forge_charge += 1
            reward += R_FORGE_BUMPER
            if self.forge_charge >= 4:
                reward += self._trigger_forge()

        # Hole collisions
        for hole in self.holes:
            if hole.check_collision(ball):
                reward     += R_HOLE
                other_holes = [h for h in self.holes
                               if h.hole_id != hole.hole_id]
                if other_holes:
                    exit_hole = random.choice(other_holes)
                    ball.x    = exit_hole.x
                    ball.y    = exit_hole.y
                    angle     = random.uniform(
                        math.pi * 0.75, math.pi * 1.25)
                    speed     = random.uniform(8, 12)
                    ball.vx   = math.cos(angle) * speed
                    ball.vy   = math.sin(angle) * speed
                    exit_hole.pulse = 15
                break

        # Slingshot collisions
        for sling in self.slingshots:
            if sling.check_collision(ball, cfg["slingshot_power"]):
                reward += R_SLINGSHOT

        # Flipper collisions
        self.left_flipper.check_collision(ball)
        self.right_flipper.check_collision(ball)

        # Ramp collision
        ramp_reward = self._check_ramp(ball, cfg)
        reward     += ramp_reward

        # Multiplier zone collisions
        for zone in self.multiplier_zones:
            if zone.check_collision(ball):
                reward                += R_MULTIPLIER_PAD
                self.multiplier_active = True
                self.multiplier        = cfg["multiplier"]
                self.multiplier_timer  = 10.0
                zone.active            = False
                zone.timer             = random.uniform(8, 20)

        # Multiball score multiplier
        if self.multiball:
            reward *= 2

        return reward

    def _wall_hit(self, wall):
        if wall == self.last_wall:
            self.wall_hit_count += 1
            if self.wall_hit_count >= 3:
                self.wall_hit_count = 0
                return R_WALL_REPEAT
        else:
            self.last_wall      = wall
            self.wall_hit_count = 1
        return 0.0

    def _trigger_forge(self):
        """Trigger multi-ball and reset forge."""
        self.forge_charge = 0
        self.multiball    = True

        # Launch 2 extra balls
        for _ in range(2):
            angle = random.uniform(
                math.pi * 0.6, math.pi * 0.9)
            speed = random.uniform(10, 14)
            new_ball = Ball(
                TABLE_W // 2,
                200,
                math.cos(angle) * speed,
                math.sin(angle) * speed
            )
            self.balls.append(new_ball)

        return R_FORGE_CHARGED

    def _check_ramp(self, ball, cfg):
        """Check if ball enters or completes the ramp."""
        # Ramp entry zone — left side mid table
        entry_x1, entry_y1 = 30,  420
        entry_x2, entry_y2 = 80,  480

        if (entry_x1 <= ball.x <= entry_x2 and
                entry_y1 <= ball.y <= entry_y2 and
                not self.ramp_active):
            speed = math.sqrt(ball.vx**2 + ball.vy**2)
            if speed >= cfg["ramp_speed"]:
                self.ramp_active = True
                self.ramp_ball   = ball
                ball.vx          =  2.0
                ball.vy          = -14.0
                return 0.0

        # Ramp exit zone — top area
        if self.ramp_active and self.ramp_ball == ball:
            if ball.y < 100:
                self.ramp_active = False
                self.ramp_ball   = None
                ball.vx          = random.uniform(-3, 3)
                ball.vy          = random.uniform(4, 8)
                return R_RAMP

        return 0.0

    def _any_ball_in_zone(self, zone):
        for ball in self.balls:
            if zone.check_collision(ball):
                return True
        return False

    # ── Table Builder ─────────────────────────────────────────────────────────

    def _build_table(self):
        """Build all table elements with randomization."""
        self._build_bumpers()
        self._build_holes()
        self._build_slingshots()
        self._build_multiplier_zones()
        self._build_flippers()

    def _build_bumpers(self):
        self.bumpers = []

        # Standard bumper positions in bonus zone
        positions = [
            (120, 220), (200, 200), (280, 220), (360, 200),
            (160, 270), (240, 250), (320, 270),
            (130, 320), (210, 300), (290, 300), (370, 320),
        ]

        # Randomly assign 3 hot bumpers
        hot_indices   = random.sample(range(len(positions)), 3)

        # Two forge bumpers — fixed positions flanking The Forge
        forge_positions = [(190, 230), (310, 230)]

        for i, (x, y) in enumerate(positions):
            kind = "hot" if i in hot_indices else "standard"
            self.bumpers.append(Bumper(x, y, kind))

        for x, y in forge_positions:
            self.bumpers.append(Bumper(x, y, "forge"))

    def _build_holes(self):
        self.holes = [
            Hole(100, 380, 0),
            Hole(400, 380, 1),
            Hole(100, 520, 2),
            Hole(400, 520, 3),
        ]

    def _build_slingshots(self):
        self.slingshots = [
            Slingshot(
                [(60, 580), (110, 640), (60, 640)],
                "left"
            ),
            Slingshot(
                [(440, 580), (390, 640), (440, 640)],
                "right"
            ),
        ]

    def _build_multiplier_zones(self):
        self.multiplier_zones = [
            MultiplierZone(130, 450, 100, 40, 0),
            MultiplierZone(270, 450, 100, 40, 1),
        ]

    def _build_flippers(self):
        self.left_flipper  = Flipper(140, 720, "left")
        self.right_flipper = Flipper(360, 720, "right")

    def _launch_ball(self):
        """Launch ball from right side plunger."""
        angle = random.uniform(
            math.pi * 0.85, math.pi * 0.95)
        speed = random.uniform(12, 15)
        ball  = Ball(
            TABLE_W - 30,
            650,
            math.cos(angle) * speed,
            math.sin(angle) * speed
        )
        self.balls.append(ball)

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self):
        if not self.balls:
            ball_x, ball_y   = 0.5, 0.5
            ball_vx, ball_vy = 0.0, 0.0
        else:
            ball             = self.balls[0]
            ball_x           = ball.x / TABLE_W
            ball_y           = ball.y / TABLE_H
            ball_vx          = ball.vx / MAX_BALL_SPEED
            ball_vy          = ball.vy / MAX_BALL_SPEED

        return np.array([
            ball_x,
            ball_y,
            ball_vx,
            ball_vy,
            float(self.left_flipper.active  if self.left_flipper  else 0),
            float(self.right_flipper.active if self.right_flipper else 0),
            self.forge_charge / 4.0,
            self.multiplier / 3.0,
            float(self.multiplier_zones[0].active if self.multiplier_zones else 0),
            float(self.multiplier_zones[1].active if self.multiplier_zones else 0),
            self.lives / 5.0,
            len(self.balls) / 3.0,
        ], dtype=np.float32)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_table(self):
        """Draw the full pinball table."""
        # Table background
        pygame.draw.rect(self.window, (18, 18, 30),
                         pygame.Rect(0, 0, TABLE_W, TABLE_H))

        # Table border
        pygame.draw.rect(self.window, CERULEAN,
                         pygame.Rect(0, 0, TABLE_W, TABLE_H), 3)

        # Top corner angle guides
        pygame.draw.line(self.window, DARK_GRAY,
                         (0, 100), (80, 0), 4)
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W, 100), (TABLE_W - 80, 0), 4)

        # Plunger lane
        pygame.draw.rect(self.window, DARK_GRAY,
                         pygame.Rect(TABLE_W - 45, 600,
                                     45, 200), 2)

        # Ramp
        self._draw_ramp()

        # The Forge
        self._draw_forge()

        # Bumpers
        for bumper in self.bumpers:
            self._draw_bumper(bumper)

        # Holes
        for hole in self.holes:
            self._draw_hole(hole)
            if hole.pulse > 0:
                hole.pulse -= 1

        # Slingshots
        for sling in self.slingshots:
            self._draw_slingshot(sling)
            if sling.hit_timer > 0:
                sling.hit_timer -= 1

        # Multiplier zones
        for zone in self.multiplier_zones:
            self._draw_multiplier_zone(zone)

        # Flippers
        self._draw_flipper(self.left_flipper)
        self._draw_flipper(self.right_flipper)

        # Balls
        for ball in self.balls:
            pygame.draw.circle(self.window, WHITE,
                               ball.pos(), BALL_RADIUS)
            pygame.draw.circle(self.window, GRAY,
                               ball.pos(), BALL_RADIUS, 2)

        # Rules button
        self._draw_rules_button()

        # Multiplier indicator
        if self.multiplier_active:
            t   = self.font_md.render(
                f"{DIFFICULTIES[self.difficulty]['multiplier']}x ACTIVE!",
                True, GOLD)
            self.window.blit(t, t.get_rect(
                center=(TABLE_W // 2, TABLE_H - 40)))

    def _draw_forge(self):
        cx, cy  = TABLE_W // 2, 180
        charge  = int(self.forge_charge)
        color   = FORGE_COLORS[min(charge, 4)]

        # Outer ring showing charge
        for i in range(4):
            ring_color = FORGE_COLORS[i + 1] \
                         if i < charge else DARK_GRAY
            pygame.draw.circle(self.window, ring_color,
                               (cx, cy), 30 + i * 5, 3)

        # Forge body
        pygame.draw.circle(self.window, color, (cx, cy), 28)
        pygame.draw.circle(self.window, WHITE, (cx, cy), 28, 2)

        # Anvil symbol
        lbl = self.font_lg.render("⚒", True,
                                   BLACK if charge >= 3 else WHITE)
        self.window.blit(lbl, lbl.get_rect(center=(cx, cy)))

        # Charge label
        lbl2 = self.font_sm.render(
            f"FORGE {charge}/4", True, color)
        self.window.blit(lbl2, lbl2.get_rect(
            center=(cx, cy + 48)))

    def _draw_bumper(self, bumper):
        if bumper.kind == "hot":
            color = CERULEAN_LT
        elif bumper.kind == "forge":
            charge = int(self.forge_charge)
            color  = FORGE_COLORS[min(charge, 4)]
        else:
            color = GRAY

        if bumper.hit_timer > 0:
            color = WHITE
            bumper.hit_timer -= 1

        pygame.draw.circle(self.window, color,
                           (bumper.x, bumper.y), bumper.radius)
        pygame.draw.circle(self.window, WHITE,
                           (bumper.x, bumper.y), bumper.radius, 2)

        if bumper.kind == "forge":
            lbl = self.font_sm.render("⚒", True, BLACK)
            self.window.blit(lbl, lbl.get_rect(
                center=(bumper.x, bumper.y)))

    def _draw_hole(self, hole):
        pulse_color = CERULEAN if hole.pulse == 0 else WHITE
        pygame.draw.circle(self.window, BLACK,
                           (hole.x, hole.y), hole.radius)
        pygame.draw.circle(self.window, pulse_color,
                           (hole.x, hole.y), hole.radius, 3)

    def _draw_slingshot(self, sling):
        color = WHITE if sling.hit_timer > 0 else DARK_GRAY
        pygame.draw.polygon(self.window, color, sling.points)
        pygame.draw.polygon(self.window, CERULEAN, sling.points, 2)

    def _draw_multiplier_zone(self, zone):
        if not zone.active and not zone.warning:
            # Faint outline only
            pygame.draw.rect(self.window, DARK_GRAY,
                             pygame.Rect(zone.x, zone.y,
                                         zone.w, zone.h), 1,
                             border_radius=6)
            return

        pulse_val = int((math.sin(zone.pulse) + 1) * 60)

        if zone.warning:
            color = (0, max(60, pulse_val),
                     max(80, pulse_val + 20))
        else:
            color = (0, min(255, 123 + pulse_val),
                     min(255, 167 + pulse_val))

        pygame.draw.rect(self.window, color,
                         pygame.Rect(zone.x, zone.y,
                                     zone.w, zone.h),
                         border_radius=6)
        pygame.draw.rect(self.window, CERULEAN_LT,
                         pygame.Rect(zone.x, zone.y,
                                     zone.w, zone.h),
                         2, border_radius=6)

        if zone.active:
            cfg = DIFFICULTIES[self.difficulty]
            lbl = self.font_sm.render(
                f"{cfg['multiplier']}x", True, WHITE)
            self.window.blit(lbl, lbl.get_rect(
                center=(zone.x + zone.w // 2,
                        zone.y + zone.h // 2)))

    def _draw_flipper(self, flipper):
        tip = flipper.get_tip()
        pygame.draw.line(self.window, CERULEAN,
                         (int(flipper.x), int(flipper.y)),
                         (int(tip[0]), int(tip[1])), 10)
        pygame.draw.circle(self.window, CERULEAN,
                           (int(flipper.x), int(flipper.y)), 8)

    def _draw_ramp(self):
        # Ramp track — left side
        ramp_color = CERULEAN_LT if self.ramp_active else CERULEAN
        points     = [(30, 500), (80, 480),
                      (80, 420), (30, 440)]
        pygame.draw.lines(self.window, ramp_color,
                          False, points, 4)
        lbl = self.font_sm.render("RAMP", True, ramp_color)
        self.window.blit(lbl, (10, 460))

    def _draw_rules_button(self):
        btn = pygame.Rect(TABLE_W - 80, TABLE_H - 36, 74, 28)
        pygame.draw.rect(self.window, (30, 30, 55),
                         btn, border_radius=6)
        pygame.draw.rect(self.window, CERULEAN,
                         btn, 1, border_radius=6)
        lbl = self.font_sm.render("📋 Rules", True, GRAY)
        self.window.blit(lbl, lbl.get_rect(center=btn.center))

    def _draw_panel(self):
        """Draw the right side score/info panel."""
        panel = pygame.Rect(TABLE_W, 0, PANEL_W, TABLE_H)
        pygame.draw.rect(self.window, PANEL_BG, panel)
        pygame.draw.line(self.window, CERULEAN,
                         (TABLE_W, 0), (TABLE_W, TABLE_H), 2)

        y = 20
        # Title
        t = self.font_lg.render("⚡ DATAFORGE", True, CERULEAN)
        self.window.blit(t, t.get_rect(
            center=(TABLE_W + PANEL_W // 2, y)))
        y += 30

        t2 = self.font_sm.render("PINBALL", True, CERULEAN_LT)
        self.window.blit(t2, t2.get_rect(
            center=(TABLE_W + PANEL_W // 2, y)))
        y += 30

        # Score
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W + 10, y),
                         (TABLE_W + PANEL_W - 10, y), 1)
        y += 10
        s = self.font_md.render("SCORE", True, GRAY)
        self.window.blit(s, (TABLE_W + 10, y))
        y += 22
        score_lbl = self.font_lg.render(
            f"{int(self.score):,}", True, WHITE)
        self.window.blit(score_lbl, (TABLE_W + 10, y))
        y += 36

        # Lives
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W + 10, y),
                         (TABLE_W + PANEL_W - 10, y), 1)
        y += 10
        l = self.font_md.render("LIVES", True, GRAY)
        self.window.blit(l, (TABLE_W + 10, y))
        y += 22
        cfg   = DIFFICULTIES[self.difficulty]
        balls = "●" * self.lives + "○" * (cfg["lives"] - self.lives)
        bl    = self.font_md.render(balls, True, CERULEAN)
        self.window.blit(bl, (TABLE_W + 10, y))
        y += 36

        # Forge charge
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W + 10, y),
                         (TABLE_W + PANEL_W - 10, y), 1)
        y += 10
        fc = self.font_md.render("FORGE", True, GRAY)
        self.window.blit(fc, (TABLE_W + 10, y))
        y += 22
        charge     = int(self.forge_charge)
        forge_bars = "▰" * charge + "▱" * (4 - charge)
        fc2        = self.font_md.render(
            forge_bars, True,
            FORGE_COLORS[min(charge, 4)])
        self.window.blit(fc2, (TABLE_W + 10, y))
        y += 36

        # Multiplier
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W + 10, y),
                         (TABLE_W + PANEL_W - 10, y), 1)
        y += 10
        mx = self.font_md.render("MULTIPLIER", True, GRAY)
        self.window.blit(mx, (TABLE_W + 10, y))
        y += 22
        if self.multiplier_active:
            ml = self.font_lg.render(
                f"{cfg['multiplier']}x", True, GOLD)
        else:
            ml = self.font_md.render("1x", True, DARK_GRAY)
        self.window.blit(ml, (TABLE_W + 10, y))
        y += 36

        # Multi-ball indicator
        if self.multiball:
            pygame.draw.line(self.window, DARK_GRAY,
                             (TABLE_W + 10, y),
                             (TABLE_W + PANEL_W - 10, y), 1)
            y += 10
            mb = self.font_md.render("MULTI-BALL!", True, YELLOW)
            self.window.blit(mb, (TABLE_W + 10, y))
            y += 22
            bc = self.font_sm.render(
                f"{len(self.balls)} balls active", True, GOLD)
            self.window.blit(bc, (TABLE_W + 10, y))
            y += 30

        # Difficulty
        pygame.draw.line(self.window, DARK_GRAY,
                         (TABLE_W + 10, y),
                         (TABLE_W + PANEL_W - 10, y), 1)
        y += 10
        dl = self.font_sm.render(
            self.difficulty, True, CERULEAN_LT)
        self.window.blit(dl, (TABLE_W + 10, y))

    def _draw_rules_overlay(self):
        """Draw rules popup overlay."""
        overlay = pygame.Surface(
            (TABLE_W, TABLE_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.window.blit(overlay, (0, 0))

        popup_w = 420
        popup_h = 480
        popup_x = (TABLE_W - popup_w) // 2
        popup_y = (TABLE_H - popup_h) // 2
        popup   = pygame.Rect(popup_x, popup_y,
                               popup_w, popup_h)

        pygame.draw.rect(self.window, (20, 20, 40),
                         popup, border_radius=14)
        pygame.draw.rect(self.window, CERULEAN,
                         popup, 2, border_radius=14)

        y = popup_y + 16
        title = self.font_lg.render(
            "📋 How Points Work", True, CERULEAN_LT)
        self.window.blit(title, title.get_rect(
            center=(TABLE_W // 2, y)))
        y += 32

        # Rules rows
        for label, value in RULES:
            col = GREEN if value.startswith("+") else RED
            lbl = self.font_sm.render(label, True, GRAY)
            val = self.font_sm.render(value, True, col)
            self.window.blit(lbl, (popup_x + 16, y))
            self.window.blit(val, (popup_x + popup_w - 
                                   val.get_width() - 16, y))
            y += 20

        y += 10

        # Controls
        title2 = self.font_md.render(
            "Controls", True, CERULEAN_LT)
        self.window.blit(title2, (popup_x + 16, y))
        y += 22

        cfg      = DIFFICULTIES[self.difficulty]
        controls = CONTROLS_STANDARD \
                   if cfg["dual_flipper"] else CONTROLS_SIMPLE
        for key, action in controls:
            kl = self.font_sm.render(key,    True, GOLD)
            al = self.font_sm.render(action, True, GRAY)
            self.window.blit(kl, (popup_x + 16, y))
            self.window.blit(al, (popup_x + 180,  y))
            y += 18

        y += 10
        btn = pygame.Rect(TABLE_W // 2 - 60,
                          popup_y + popup_h - 50,
                          120, 36)
        pygame.draw.rect(self.window, CERULEAN,
                         btn, border_radius=8)
        got_it = self.font_md.render("Got it!", True, WHITE)
        self.window.blit(got_it, got_it.get_rect(
            center=btn.center))

    # ── Human Play ────────────────────────────────────────────────────────────

    def play_human_episode(self):
        """
        Run a full human-controlled pinball episode.
        Returns final score and result dict.
        """
        self.render()
        cfg = DIFFICULTIES[self.difficulty]

        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Rules button
                    btn = pygame.Rect(
                        TABLE_W - 80, TABLE_H - 36, 74, 28)
                    if btn.collidepoint(event.pos):
                        self.show_rules = not self.show_rules

                    # Rules overlay Got it button
                    if self.show_rules:
                        got_btn = pygame.Rect(
                            TABLE_W // 2 - 60,
                            TABLE_H // 2 + 190, 120, 36)
                        if got_btn.collidepoint(event.pos):
                            self.show_rules = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.show_rules = False

                    if not self.show_rules:
                        if cfg["dual_flipper"]:
                            if event.key == pygame.K_LEFT:
                                self.left_flipper.active  = True
                            if event.key == pygame.K_RIGHT:
                                self.right_flipper.active = True
                        else:
                            if event.key == pygame.K_SPACE:
                                self.left_flipper.active  = True
                                self.right_flipper.active = True

                if event.type == pygame.KEYUP:
                    if cfg["dual_flipper"]:
                        if event.key == pygame.K_LEFT:
                            self.left_flipper.active  = False
                        if event.key == pygame.K_RIGHT:
                            self.right_flipper.active = False
                    else:
                        if event.key == pygame.K_SPACE:
                            self.left_flipper.active  = False
                            self.right_flipper.active = False

            if not self.show_rules:
                self.step(0)   # physics update with no action
                               # flipper state already set by keys

            self.render()

        return {
            "score":          self.score,
            "total_reward":   self.total_reward,
            "won":            False,   # pinball has no win — just high score
            "forge_charges":  0,       # tracked via rewards
            "difficulty":     self.difficulty,
        }

    # ── Difficulty Screen ─────────────────────────────────────────────────────

    def _difficulty_screen(self):
        W = 520
        H = 420
        pygame.display.set_mode((W, H))
        pygame.display.set_caption(
            "DataForge Pinball — Select Difficulty")

        title_font = pygame.font.SysFont("segoeui", 28, bold=True)
        btn_font   = pygame.font.SysFont("segoeui", 18, bold=True)
        desc_font  = pygame.font.SysFont("segoeui", 13)

        btn_colors = {
            "Simple":   (46, 160,  67),
            "Standard": (210, 140,   0),
            "Complex":  (200,  30,  30),
        }

        buttons = {}
        btn_h   = 70
        btn_w   = 420
        start_y = 130

        for i, (name, cfg) in enumerate(DIFFICULTIES.items()):
            rect = pygame.Rect(
                (W - btn_w) // 2,
                start_y + i * (btn_h + 14),
                btn_w, btn_h
            )
            buttons[name] = {"rect": rect, "cfg": cfg}

        while True:
            mouse = pygame.mouse.get_pos()
            self.window.fill(BG_COLOR)

            title = title_font.render(
                "⚡ DataForge Pinball", True, CERULEAN_LT)
            self.window.blit(title, title.get_rect(
                center=(W // 2, 60)))

            sub = desc_font.render(
                "Select difficulty to begin", True, GRAY)
            self.window.blit(sub, sub.get_rect(
                center=(W // 2, 95)))

            for name, data in buttons.items():
                rect    = data["rect"]
                hovered = rect.collidepoint(mouse)
                color   = (80, 80, 120) if hovered \
                          else btn_colors[name]

                pygame.draw.rect(self.window, color,
                                 rect, border_radius=10)
                pygame.draw.rect(self.window, WHITE,
                                 rect, 2, border_radius=10)

                lbl  = btn_font.render(name, True, WHITE)
                desc = desc_font.render(
                    data["cfg"]["desc"], True, WHITE)

                self.window.blit(lbl, lbl.get_rect(
                    center=(rect.centerx, rect.centery - 12)))
                self.window.blit(desc, desc.get_rect(
                    center=(rect.centerx, rect.centery + 14)))

            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN \
                        and event.button == 1:
                    for name, data in buttons.items():
                        if data["rect"].collidepoint(event.pos):
                            return name

    # ── Pygame Init ───────────────────────────────────────────────────────────

    def _init_pygame_basic(self):
        if self.window is None:
            pygame.init()
            self.window  = pygame.display.set_mode((520, 420))
            self.clock   = pygame.time.Clock()
            self.font_lg = pygame.font.SysFont("segoeui", 20, bold=True)
            self.font_md = pygame.font.SysFont("segoeui", 15)
            self.font_sm = pygame.font.SysFont("segoeui", 12)

    def _init_pygame(self):
        if self.window is None:
            pygame.init()
        self.window  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(
            f"DataForge Pinball ({self.difficulty})")
        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_md = pygame.font.SysFont("segoeui", 15)
        self.font_sm = pygame.font.SysFont("segoeui", 12)
