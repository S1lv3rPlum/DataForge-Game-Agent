# launch.py
# DataForge Game Agent — Main Launcher
# Version 1.0
#
# The single entry point for everything.
# Run with:  python launch.py

import sys
import os
import subprocess
import pygame
import time

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

# Agent options
AGENTS = ["Random Agent", "DQN Learning Agent", "Both"]

AGENT_COLORS = {
    "Random Agent":      (255, 107, 107),
    "DQN Learning Agent":(78,  205, 196),
    "Both":              (255, 230, 109),
}

AGENT_DESCRIPTIONS = {
    "Random Agent":       "Clicks randomly — no learning",
    "DQN Learning Agent": "Learns from experience using deep RL",
    "Both":               "Run both simultaneously — shared dashboard",
}


class Launcher:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((W, H))
        pygame.display.set_caption("DataForge Game Agent — Launcher")
        self.clock   = pygame.time.Clock()

        # Fonts
        self.font_xl  = pygame.font.SysFont("segoeui", 32, bold=True)
        self.font_lg  = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_md  = pygame.font.SysFont("segoeui", 16)
        self.font_sm  = pygame.font.SysFont("segoeui", 13)
        self.font_ico = pygame.font.SysFont("segoeuiemoji", 28)

        self.selected_game   = None
        self.selected_agent  = None
        self.hovered_card    = None
        self.hovered_agent   = None
        self.hovered_btn     = None

    def run(self):
    """Main launcher loop."""
    self.selected_game  = self._game_selection_screen()
    if not self.selected_game:
        pygame.quit()
        return "quit"

    self.selected_agent = self._agent_selection_screen()
    if not self.selected_agent:
        # Back was pressed — restart launcher
        pygame.quit()
        return None

    self._launch(self.selected_game, self.selected_agent)
    pygame.quit()
    return None

    # ── Game Selection Screen ─────────────────────────────────────────────────

    def _game_selection_screen(self):
        """Show game cards and return selected game name."""
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
        """Calculate card positions."""
        cards     = []
        card_w    = 200
        card_h    = 240
        cols      = min(len(games), 4)
        total_w   = cols * card_w + (cols - 1) * 20
        start_x   = (W - total_w) // 2
        start_y   = 160

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
        """Draw a single game card with icon and info."""
        color  = tuple(min(255, c + 20) for c in cfg["card_color"]) \
                 if hovered else cfg["card_color"]
        accent = cfg["accent"]

        # Card background
        pygame.draw.rect(self.screen, color, rect, border_radius=14)
        pygame.draw.rect(self.screen, accent, rect, 2, border_radius=14)

        # Hover glow
        if hovered:
            glow = pygame.Surface((rect.w + 8, rect.h + 8), pygame.SRCALPHA)
            pygame.draw.rect(glow, (*accent, 40),
                             glow.get_rect(), border_radius=16)
            self.screen.blit(glow, (rect.x - 4, rect.y - 4))

        # Icon area
        icon_rect = pygame.Rect(rect.x + 10, rect.y + 10,
                                rect.w - 20, 130)
        pygame.draw.rect(self.screen, (10, 10, 20),
                         icon_rect, border_radius=10)

        # Draw icon characters
        icon_lines = cfg["icon_chars"]
        for i, line in enumerate(icon_lines):
            lbl = self.font_ico.render(line, True, TEXT_WHITE)
            self.screen.blit(lbl, lbl.get_rect(
                center=(icon_rect.centerx,
                        icon_rect.y + 18 + i * 26)))

        # Game name
        name_lbl = self.font_lg.render(name, True, TEXT_WHITE)
        self.screen.blit(name_lbl, name_lbl.get_rect(
            center=(rect.centerx, rect.y + 155)))

        # Description
        desc_lbl = self.font_sm.render(cfg["description"], True, TEXT_GRAY)
        self.screen.blit(desc_lbl, desc_lbl.get_rect(
            center=(rect.centerx, rect.y + 178)))

        # Difficulty pills
        pill_y = rect.y + 200
        diffs  = cfg.get("difficulties", [])
        total_pill_w = len(diffs) * 58 + (len(diffs) - 1) * 6
        pill_x = rect.centerx - total_pill_w // 2

        for diff in diffs:
            pill = pygame.Rect(pill_x, pill_y, 54, 20)
            pygame.draw.rect(self.screen, (40, 40, 70),
                             pill, border_radius=10)
            pygame.draw.rect(self.screen, accent, pill, 1, border_radius=10)
            dlbl = self.font_sm.render(diff[:3], True, accent)
            self.screen.blit(dlbl, dlbl.get_rect(center=pill.center))
            pill_x += 60

    # ── Agent Selection Screen ────────────────────────────────────────────────

    def _agent_selection_screen(self):
        """Show agent options and return selected agent."""
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
                    # Back button
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
        btn_w   = 260
        btn_h   = 110
        spacing = 24
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

        # Game info recap
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
                center=(rect.centerx, rect.centery - 18)))

            desc_lbl = self.font_sm.render(
                AGENT_DESCRIPTIONS[name], True, TEXT_GRAY)
            self.screen.blit(desc_lbl, desc_lbl.get_rect(
                center=(rect.centerx, rect.centery + 8)))

            # Color indicator dot
            pygame.draw.circle(self.screen, color,
                                (rect.centerx, rect.centery + 34), 6)

        # Back button
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
        """Launch the selected game and agent(s)."""
        pygame.quit()

        scripts = {
            "Random Agent":       "agent/random_agent.py",
            "DQN Learning Agent": "agent/learning_agent.py",
        }

        if agent == "Both":
            print(f"\n🚀 Launching Both Agents for {game}...")
            print("Opening two windows — Random Agent and DQN Learning Agent")
            print("Both will share the live dashboard automatically.\n")

            # Launch random agent in a new terminal window
            subprocess.Popen(
                ["cmd", "/c", "start", "cmd", "/k",
                 f"cd /d {os.getcwd()} && "
                 f"venv\\Scripts\\activate && "
                 f"python agent/random_agent.py"],
                shell=False
            )

            time.sleep(2)   # small delay so windows don't overlap perfectly

            # Launch learning agent in this process
            os.system(f"python agent/learning_agent.py")

        else:
            script = scripts.get(agent)
            if script:
                print(f"\n🚀 Launching {agent} for {game}...")
                os.system(f"python {script}")

    # ── Shared UI Elements ────────────────────────────────────────────────────

    def _draw_header(self, subtitle):
        """Draw the top header bar."""
        pygame.draw.rect(self.screen, HEADER_COLOR,
                         pygame.Rect(0, 0, W, 130))
        pygame.draw.line(self.screen, ACCENT_TEAL, (0, 130), (W, 130), 2)

        # DataForge branding
        brand = self.font_xl.render("⚡ DataForge Game Agent", True, ACCENT_TEAL)
        self.screen.blit(brand, brand.get_rect(center=(W // 2, 55)))

        sub = self.font_md.render(subtitle, True, TEXT_GRAY)
        self.screen.blit(sub, sub.get_rect(center=(W // 2, 95)))

    def _draw_footer(self, hint):
        """Draw the bottom footer bar."""
        pygame.draw.rect(self.screen, HEADER_COLOR,
                         pygame.Rect(0, H - 40, W, 40))
        pygame.draw.line(self.screen, (40, 40, 60), (0, H - 40), (W, H - 40), 1)

        hint_lbl = self.font_sm.render(hint, True, TEXT_DIM)
        self.screen.blit(hint_lbl, hint_lbl.get_rect(center=(W // 2, H - 20)))


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    while True:
        launcher = Launcher()
        result   = launcher.run()
        if result == "quit":
            break
