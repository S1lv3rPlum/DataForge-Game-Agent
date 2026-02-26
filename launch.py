# launch.py
# DataForge Game Agent — Main Launcher
# Version 2.0
# Now with turn-based dual agent support and watch toggle
#
# Run with:  python launch.py

import sys
import os
import time
import pygame

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from games.registry import GAME_REGISTRY

# ── Display Settings ──────────────────────────────────────────────────────────
W, H         = 900, 620
FPS          = 60
BG_COLOR     = (15,  15,  25)
HEADER_COLOR = (20,  20,  40)
TEXT_WHITE   = (240, 240, 240)
TEXT_GRAY    = (160, 160, 180)
TEXT_DIM     = (100, 100, 120)
ACCENT_TEAL  = (78,  205, 196)
ACCENT_RED   = (255, 107, 107)
ACCENT_GOLD  = (255, 230, 109)

# ── Agent Options ─────────────────────────────────────────────────────────────
AGENTS = ["Random Agent", "DQN Learning Agent", "Both", "Play Yourself"]

AGENT_COLORS = {
    "Random Agent":       (255, 107, 107),
    "DQN Learning Agent": (78,  205, 196),
    "Both":               (255, 230, 109),
    "Play Yourself":      (168, 255, 120),
}

AGENT_DESCRIPTIONS = {
    "Random Agent":       "Clicks randomly — no learning",
    "DQN Learning Agent": "Learns from experience using deep RL",
    "Both":               "Both agents take turns — watch either one live",
    "Play Yourself":      "You vs AI — beat it before it beats you!",
}


# ── Top Level Functions (must be outside class for multiprocessing) ───────────

def run_random_process(game_name="Minesweeper"):
    import threading
    from agent.dashboard import Dashboard, watch_request
    from agent.random_agent import run_headless

    stop_flag = {"done": False}
    dash      = Dashboard(agent_name="Random Agent", show_window=True)

    def agent_thread():
        run_headless(game_name=game_name,
                     dash=dash,
                     stop_flag=stop_flag,
                     watch_request=watch_request)
        stop_flag["done"] = True

    t = threading.Thread(target=agent_thread, daemon=True)
    t.start()
    dash.main_loop(stop_flag)
    dash.close()


def run_learning_process(game_name="Minesweeper"):
    import threading
    from agent.dashboard import Dashboard, watch_request
    from agent.learning_agent import run_headless

    stop_flag = {"done": False}
    dash      = Dashboard(agent_name="DQN Learning Agent",
                          show_window=True)

    def agent_thread():
        run_headless(game_name=game_name,
                     dash=dash,
                     stop_flag=stop_flag,
                     watch_request=watch_request)
        stop_flag["done"] = True

    t = threading.Thread(target=agent_thread, daemon=True)
    t.start()
    dash.main_loop(stop_flag)
    dash.close()


def run_both_agents(game_name="Minesweeper"):
    import threading
    from agent.dashboard import Dashboard, watch_request
    from agent.learning_agent import DQNAgent, STEP_DELAY, TARGET_UPDATE, SAVE_EVERY

    cfg      = GAME_REGISTRY[game_name]
    EnvClass = cfg["env_class"]

    stop_flag     = {"done": False}
    random_dash   = Dashboard(agent_name="Random Agent",
                              show_window=True)
    learning_dash = Dashboard(agent_name="DQN Learning Agent",
                              show_window=False)

    # Get difficulty once
    temp_env    = EnvClass(render_mode="human")
    temp_obs, _ = temp_env.reset()
    chosen_diff = temp_env.difficulty
    state_size  = temp_env.rows * temp_env.cols
    action_size = temp_env.action_space.n
    temp_env.close()

    dqn_agent = DQNAgent(state_size, action_size, difficulty=chosen_diff)
    dqn_agent.load()

    print(f"\n🎮 Both Agents — {game_name} ({chosen_diff})")
    print(f"Use dashboard buttons to watch an agent.\n")

    episode_count = {"random": 0, "learning": 0}
    TOTAL_EPISODES = 3000

    def both_thread():
        for turn in range(TOTAL_EPISODES):
            if stop_flag["done"] or \
               watch_request.get("agent") == "MENU":
                break

            watching = watch_request.get("agent")

            # Random episode
            episode_count["random"] += 1
            render_random = watching == "Random Agent"

            r_env    = EnvClass(
                render_mode="human" if render_random else None,
                difficulty=chosen_diff
            )
            r_obs, _ = r_env.reset()
            r_done   = False
            r_reward = 0
            r_steps  = 0

            while not r_done:
                if render_random:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            watch_request["agent"] = None
                            render_random = False

                action          = r_env.action_space.sample()
                r_obs, reward, r_done, _, _ = r_env.step(action)
                r_reward       += reward
                r_steps        += 1

                if render_random:
                    time.sleep(STEP_DELAY)

            r_env.close()

            random_dash.update(
                episode       = episode_count["random"],
                reward        = r_reward,
                won           = r_env.won,
                steps         = r_steps,
                correct_flags = r_env.correct_flags,
                pct_cleared   = r_env.safe_revealed / r_env.safe_total
                                if r_env.safe_total > 0 else 0.0
            )

            print(f"[Random]   Ep {episode_count['random']:4d} | "
                  f"{'WIN 🎉' if r_env.won else 'LOSS 💥'} | "
                  f"Steps: {r_steps:3d} | Reward: {r_reward:6.1f}")

            if stop_flag["done"] or \
               watch_request.get("agent") == "MENU":
                break

            # Learning episode
            episode_count["learning"] += 1
            watching        = watch_request.get("agent")
            render_learning = watching == "DQN Learning Agent"

            l_env    = EnvClass(
                render_mode="human" if render_learning else None,
                difficulty=chosen_diff
            )
            l_obs, _ = l_env.reset()
            l_state  = l_obs
            l_done   = False
            l_reward = 0
            l_steps  = 0

            while not l_done:
                if render_learning:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            watch_request["agent"] = None
                            render_learning = False

                action = dqn_agent.select_action(l_state, l_env)
                l_next, reward, l_done, _, _ = l_env.step(action)

                dqn_agent.memory.push(
                    l_state, action, reward, l_next, l_done)
                dqn_agent.learn()

                l_state   = l_next
                l_reward += reward
                l_steps  += 1

                if render_learning:
                    time.sleep(STEP_DELAY)

            l_env.close()
            dqn_agent.update_epsilon()

            if episode_count["learning"] % TARGET_UPDATE == 0:
                dqn_agent.update_target_network()
            if episode_count["learning"] % SAVE_EVERY == 0:
                dqn_agent.save()

            learning_dash.update(
                episode       = episode_count["learning"],
                reward        = l_reward,
                won           = l_env.won,
                steps         = l_steps,
                correct_flags = l_env.correct_flags,
                pct_cleared   = l_env.safe_revealed / l_env.safe_total
                                if l_env.safe_total > 0 else 0.0
            )

            print(f"[Learning] Ep {episode_count['learning']:4d} | "
                  f"{'WIN 🎉' if l_env.won else 'LOSS 💥'} | "
                  f"Steps: {l_steps:3d} | Reward: {l_reward:6.1f} | "
                  f"ε: {dqn_agent.epsilon:.3f}")

        dqn_agent.save()
        stop_flag["done"] = True
        print("\nBoth agents finished!")

    t = threading.Thread(target=both_thread, daemon=True)
    t.start()
    random_dash.main_loop(stop_flag)
    random_dash.close()
    learning_dash.close()
    watch_request["agent"] = None


# ── Launcher Class ────────────────────────────────────────────────────────────
def run_human_mode(game_name="Minesweeper"):
    import threading
    from agent.dashboard import Dashboard, watch_request
    from agent.learning_agent import DQNAgent, TARGET_UPDATE, SAVE_EVERY

    cfg      = GAME_REGISTRY[game_name]
    EnvClass = cfg["env_class"]

    stop_flag     = {"done": False}
    human_dash    = Dashboard(agent_name="Human",
                              show_window=True,
                              human_mode=True)
    random_dash   = Dashboard(agent_name="Random Agent",
                              show_window=False)
    learning_dash = Dashboard(agent_name="DQN Learning Agent",
                              show_window=False)

    # Get difficulty via human env
    human_env    = EnvClass(render_mode="human")
    obs, _       = human_env.reset()
    chosen_diff  = human_env.difficulty
    state_size   = human_env.rows * human_env.cols
    action_size  = human_env.action_space.n

    dqn_agent = DQNAgent(state_size, action_size, difficulty=chosen_diff)
    dqn_agent.load()

    print(f"\n👤 Human vs AI — {game_name} ({chosen_diff})")
    print(f"Left click = reveal  |  Right click = flag\n")

    ai_snapshot   = {
        "Random Agent": {
            "wins": 0, "episodes": 0,
            "avg_reward": 0.0, "total_reward": 0.0
        },
        "DQN Learning Agent": {
            "wins": 0, "episodes": 0,
            "avg_reward": 0.0, "total_reward": 0.0
        },
    }
    snapshot_lock = threading.Lock()
    ai_episode    = {"random": 0, "learning": 0}

    def ai_worker():
        r_env = EnvClass(render_mode=None, difficulty=chosen_diff)
        l_env = EnvClass(render_mode=None, difficulty=chosen_diff)

        while not stop_flag["done"]:
            # Random episode
            r_obs, _ = r_env.reset()
            r_done   = False
            r_reward = 0
            r_steps  = 0

            while not r_done:
                action          = r_env.action_space.sample()
                r_obs, reward, r_done, _, _ = r_env.step(action)
                r_reward       += reward
                r_steps        += 1

            ai_episode["random"] += 1

            with snapshot_lock:
                snap = ai_snapshot["Random Agent"]
                snap["episodes"]     += 1
                snap["total_reward"] += r_reward
                snap["avg_reward"]    = (snap["total_reward"] /
                                         snap["episodes"])
                if r_env.won:
                    snap["wins"] += 1

            random_dash.update(
                episode       = ai_episode["random"],
                reward        = r_reward,
                won           = r_env.won,
                steps         = r_steps,
                correct_flags = r_env.correct_flags,
                pct_cleared   = r_env.safe_revealed / r_env.safe_total
                                if r_env.safe_total > 0 else 0.0
            )

            # Learning episode
            l_obs, _ = l_env.reset()
            l_state  = l_obs
            l_done   = False
            l_reward = 0
            l_steps  = 0

            while not l_done:
                action = dqn_agent.select_action(l_state, l_env)
                l_next, reward, l_done, _, _ = l_env.step(action)
                dqn_agent.memory.push(
                    l_state, action, reward, l_next, l_done)
                dqn_agent.learn()
                l_state   = l_next
                l_reward += reward
                l_steps  += 1

            dqn_agent.update_epsilon()
            ai_episode["learning"] += 1

            if ai_episode["learning"] % TARGET_UPDATE == 0:
                dqn_agent.update_target_network()
            if ai_episode["learning"] % SAVE_EVERY == 0:
                dqn_agent.save()

            with snapshot_lock:
                snap = ai_snapshot["DQN Learning Agent"]
                snap["episodes"]     += 1
                snap["total_reward"] += l_reward
                snap["avg_reward"]    = (snap["total_reward"] /
                                         snap["episodes"])
                if l_env.won:
                    snap["wins"] += 1

            learning_dash.update(
                episode       = ai_episode["learning"],
                reward        = l_reward,
                won           = l_env.won,
                steps         = l_steps,
                correct_flags = l_env.correct_flags,
                pct_cleared   = l_env.safe_revealed / l_env.safe_total
                                if l_env.safe_total > 0 else 0.0
            )

        r_env.close()
        l_env.close()
        dqn_agent.save()

    # Start AI thread
    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    # Human game loop runs in main thread
    human_episode = 0
    playing       = True

    while playing and not stop_flag["done"]:
        if watch_request.get("agent") == "MENU":
            break

        # Reset snapshot for this round
        with snapshot_lock:
            for snap in ai_snapshot.values():
                snap["wins"]         = 0
                snap["episodes"]     = 0
                snap["avg_reward"]   = 0.0
                snap["total_reward"] = 0.0

        human_env.difficulty = chosen_diff
        human_env.done       = False
        human_env.won        = False
        obs, _               = human_env.reset()

        with snapshot_lock:
            current_snap = {k: dict(v) for k, v in ai_snapshot.items()}

        stats, action = human_env.play_human_episode(
            ai_snapshot=current_snap)

        human_episode += 1

        human_dash.update(
            episode       = human_episode,
            reward        = stats["reward"],
            won           = stats["won"],
            steps         = stats["steps"],
            correct_flags = stats["correct_flags"],
            pct_cleared   = stats["safe_revealed"] / stats["safe_total"]
                            if stats["safe_total"] > 0 else 0.0
        )

        # Refresh dashboard after human game
        try:
            human_dash._draw()
        except Exception:
            pass

        print(f"[Human]    Ep {human_episode:4d} | "
              f"{'WIN 🎉' if stats['won'] else 'LOSS 💥'} | "
              f"Steps: {stats['steps']:3d} | "
              f"Reward: {stats['reward']:6.1f}")

        if action == "menu":
            playing = False

    stop_flag["done"] = True
    human_env.close()
    human_dash.close()
    random_dash.close()
    learning_dash.close()
    watch_request["agent"] = None
    print("\nHuman mode finished!")
    
class Launcher:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((W, H))
        pygame.display.set_caption("DataForge Game Agent — Launcher")
        self.clock   = pygame.time.Clock()

        self.font_xl  = pygame.font.SysFont("segoeui", 32, bold=True)
        self.font_lg  = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_md  = pygame.font.SysFont("segoeui", 16)
        self.font_sm  = pygame.font.SysFont("segoeui", 13)
        self.font_ico = pygame.font.SysFont("segoeuiemoji", 28)

        self.selected_game  = None
        self.selected_agent = None
        self.hovered_card   = None
        self.hovered_agent  = None

    def run(self):
        self.selected_game = self._game_selection_screen()
        if not self.selected_game:
            pygame.quit()
            return "quit"

        self.selected_agent = self._agent_selection_screen()
        if not self.selected_agent:
            pygame.quit()
            return None

        self._launch(self.selected_game, self.selected_agent)
        pygame.quit()
        return None

    # ── Game Selection ────────────────────────────────────────────────────────

    def _game_selection_screen(self):
        games    = list(GAME_REGISTRY.items())
        cards    = self._build_game_cards(games)
        selected = None

        while selected is None:
            mouse = pygame.mouse.get_pos()
            self.hovered_card = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, rect in cards:
                        if rect.collidepoint(event.pos):
                            selected = name

            for name, rect in cards:
                if rect.collidepoint(mouse):
                    self.hovered_card = name

            self._draw_game_screen(games, cards)
            self.clock.tick(FPS)

        return selected

    def _build_game_cards(self, games):
        cards   = []
        card_w  = 200
        card_h  = 200
        cols    = min(len(games), 4)
        total_w = cols * card_w + (cols - 1) * 20
        start_x = (W - total_w) // 2
        start_y = 160

        for i, (name, _) in enumerate(games):
            x    = start_x + i * (card_w + 20)
            rect = pygame.Rect(x, start_y, card_w, card_h)
            cards.append((name, rect))

        return cards

    def _draw_game_screen(self, games, cards):
        self.screen.fill(BG_COLOR)
        self._draw_header("Select a Game")
        self._draw_footer("Click a game to continue")

        for name, rect in cards:
            cfg     = GAME_REGISTRY[name]
            hovered = self.hovered_card == name
            self._draw_game_card(rect, name, cfg, hovered)

        pygame.display.flip()

    def _draw_game_card(self, rect, name, cfg, hovered):
        color  = tuple(min(255, c + 20) for c in cfg["card_color"]) \
                 if hovered else cfg["card_color"]
        accent = cfg["accent"]

        pygame.draw.rect(self.screen, color, rect, border_radius=14)
        pygame.draw.rect(self.screen, accent, rect, 2, border_radius=14)

        if hovered:
            glow = pygame.Surface((rect.w + 8, rect.h + 8), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*accent, 40),
                             glow.get_rect(), border_radius=16)
            self.screen.blit(glow, (rect.x - 4, rect.y - 4))

        icon_rect = pygame.Rect(rect.x + 10, rect.y + 10,
                                rect.w - 20, 110)
        pygame.draw.rect(self.screen, (10, 10, 20),
                         icon_rect, border_radius=10)

        icon_lines = cfg["icon_chars"]
        for i, line in enumerate(icon_lines):
            lbl = self.font_ico.render(line, True, TEXT_WHITE)
            self.screen.blit(lbl, lbl.get_rect(
                center=(icon_rect.centerx,
                        icon_rect.y + 16 + i * 22)))

        name_lbl = self.font_lg.render(name, True, TEXT_WHITE)
        self.screen.blit(name_lbl, name_lbl.get_rect(
            center=(rect.centerx, rect.y + 132)))

        desc_lbl = self.font_sm.render(cfg["description"], True, TEXT_GRAY)
        self.screen.blit(desc_lbl, desc_lbl.get_rect(
            center=(rect.centerx, rect.y + 158)))

    # ── Agent Selection ───────────────────────────────────────────────────────

    def _agent_selection_screen(self):
        buttons  = self._build_agent_buttons()
        selected = None

        while selected is None:
            mouse = pygame.mouse.get_pos()
            self.hovered_agent = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, rect in buttons:
                        if rect.collidepoint(event.pos):
                            selected = name
                    back = pygame.Rect(20, H - 50, 100, 34)
                    if back.collidepoint(event.pos):
                        return None

            for name, rect in buttons:
                if rect.collidepoint(mouse):
                    self.hovered_agent = name

            self._draw_agent_screen(buttons)
            self.clock.tick(FPS)

        return selected

    def _build_agent_buttons(self):
        buttons = []
        btn_w   = 200
        btn_h   = 110
        spacing = 16
        total_w = len(AGENTS) * btn_w + (len(AGENTS) - 1) * spacing
        start_x = (W - total_w) // 2
        y       = 260

        for i, name in enumerate(AGENTS):
            x    = start_x + i * (btn_w + spacing)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            buttons.append((name, rect))

        return buttons

    def _draw_agent_screen(self, buttons):
        self.screen.fill(BG_COLOR)
        self._draw_header(f"Select Agent — {self.selected_game}")
        self._draw_footer("ESC or Back to return to game selection")

        recap = self.font_md.render(
            f"Game: {self.selected_game}", True, TEXT_GRAY)
        self.screen.blit(recap, recap.get_rect(center=(W // 2, 175)))

        for name, rect in buttons:
            hovered = self.hovered_agent == name
            color   = AGENT_COLORS[name]
            bg      = tuple(min(255, c // 3 + 20) for c in color)
            if hovered:
                bg = tuple(min(255, c // 2 + 20) for c in color)

            pygame.draw.rect(self.screen, bg, rect, border_radius=12)
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=12)

            if hovered:
                glow = pygame.Surface((rect.w + 8, rect.h + 8),
                                      pygame.SRCALPHA)
                pygame.draw.rect(glow, (*color, 40),
                                 glow.get_rect(), border_radius=14)
                self.screen.blit(glow, (rect.x - 4, rect.y - 4))

            name_lbl = self.font_lg.render(name, True, TEXT_WHITE)
            self.screen.blit(name_lbl, name_lbl.get_rect(
                center=(rect.centerx, rect.centery - 22)))

            # Word wrap description
            words    = AGENT_DESCRIPTIONS[name].split()
            lines    = []
            line     = ""
            for word in words:
                test = line + " " + word if line else word
                if self.font_sm.size(test)[0] < rect.w - 16:
                    line = test
                else:
                    lines.append(line)
                    line = word
            if line:
                lines.append(line)

            for i, ln in enumerate(lines):
                dl = self.font_sm.render(ln, True, TEXT_GRAY)
                self.screen.blit(dl, dl.get_rect(
                    center=(rect.centerx,
                            rect.centery + 2 + i * 16)))

            pygame.draw.circle(self.screen, color,
                               (rect.centerx, rect.centery + 34), 6)

        back_rect = pygame.Rect(20, H - 50, 100, 34)
        pygame.draw.rect(self.screen, (40, 40, 60),
                         back_rect, border_radius=8)
        pygame.draw.rect(self.screen, TEXT_DIM,
                         back_rect, 1, border_radius=8)
        back_lbl = self.font_sm.render("← Back", True, TEXT_GRAY)
        self.screen.blit(back_lbl, back_lbl.get_rect(
            center=back_rect.center))

        pygame.display.flip()

    # ── Launch ────────────────────────────────────────────────────────────────

    def _launch(self, game, agent):
        pygame.quit()
        pygame.display.quit()

        if agent == "Both":
            run_both_agents(game)
        elif agent == "Random Agent":
            run_random_process(game)
        elif agent == "DQN Learning Agent":
            run_learning_process(game)
        elif agent == "Play Yourself":
            run_human_mode(game)

    # ── Shared UI ─────────────────────────────────────────────────────────────

    def _draw_header(self, subtitle):
        pygame.draw.rect(self.screen, HEADER_COLOR,
                         pygame.Rect(0, 0, W, 130))
        pygame.draw.line(self.screen, ACCENT_TEAL,
                         (0, 130), (W, 130), 2)

        brand = self.font_xl.render(
            "⚡ DataForge Game Agent", True, ACCENT_TEAL)
        self.screen.blit(brand, brand.get_rect(center=(W // 2, 55)))

        sub = self.font_md.render(subtitle, True, TEXT_GRAY)
        self.screen.blit(sub, sub.get_rect(center=(W // 2, 95)))

    def _draw_footer(self, hint):
        pygame.draw.rect(self.screen, HEADER_COLOR,
                         pygame.Rect(0, H - 40, W, 40))
        pygame.draw.line(self.screen, (40, 40, 60),
                         (0, H - 40), (W, H - 40), 1)

        hint_lbl = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(hint_lbl, hint_lbl.get_rect(
            center=(W // 2, H - 20)))

# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    while True:
        launcher = Launcher()
        result   = launcher.run()
        if result == "quit":
            break
