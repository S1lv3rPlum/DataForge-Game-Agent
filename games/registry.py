# registry.py
# Game Registry for DataForge Game Agent
# Add new games here — they appear automatically in the launcher
#
# To add a new game:
# 1. Create games/yourgame/yourgame_env.py
# 2. Add an entry to GAME_REGISTRY below
# 3. Set active=True when ready to show in launcher

from games.minesweeper.minesweeper_env import MinesweeperEnv

# Pinball imported but commented out until ready for launcher
# from games.pinball.pinball_env import PinballEnv

GAME_REGISTRY = {
    "Minesweeper": {
        "env_class":   MinesweeperEnv,
        "description": "Classic mine-finding puzzle",
        "difficulties": ["Beginner", "Medium", "Hard"],
        "icon_chars":  ["💣 💣", "🚩 💣", "💣 🚩"],
        "card_color":  (46,  46,  80),
        "accent":      (255, 100, 100),
        "rules": [
            ("Reveal safe cell",     "+1"),
            ("Correct flag",         "+5"),
            ("Wrong flag",           "-3"),
            ("Remove correct flag",  "-2"),
            ("Hit a mine",          "-10"),
            ("Win the game",        "+20"),
            ("Each step",           "-0.1"),
            ("Repeated action",     "-0.5"),
        ],
        "controls": [
            ("Left click",  "Reveal cell"),
            ("Right click", "Flag / unflag"),
        ],
    },

    # ── Pinball (uncomment when ready to add to launcher) ─────────────────
    # "Pinball": {
    #     "env_class":   PinballEnv,
    #     "description": "Arcade pinball with AI learning",
    #     "difficulties": ["Simple", "Standard", "Complex"],
    #     "icon_chars":  ["⚡ ●", "⚒ ●", "● ⚡"],
    #     "card_color":  (20,  40,  80),
    #     "accent":      (0,  123, 167),
    #     "rules": [
    #         ("Standard bumper",     "+10"),
    #         ("Hot bumper",          "+25"),
    #         ("Forge bumper",        "+15 + charge"),
    #         ("Forge fully charged", "+50"),
    #         ("Slingshot",           "+5"),
    #         ("Hole entry",          "+20"),
    #         ("Ramp completed",      "+30"),
    #         ("Multiplier pad",      "+15 + multiplier"),
    #         ("Multi-ball active",   "all x2"),
    #         ("Ball drained",        "-50"),
    #         ("Last life drained",   "-100"),
    #         ("Same wall x3",        "-5"),
    #         ("Multiplier expired",  "-10"),
    #     ],
    #     "controls": [
    #         ("Spacebar",      "Both flippers (Simple)"),
    #         ("← Left arrow",  "Left flipper (Standard+)"),
    #         ("→ Right arrow", "Right flipper (Standard+)"),
    #     ],
    # },
}
