# registry.py
# Game Registry — Version 1.0
# This is where all available games register themselves.
# To add a new game in the future:
#   1. Create a folder under games/
#   2. Build your environment following the Gymnasium interface
#   3. Add an entry to GAME_REGISTRY below — nothing else changes!

from games.minesweeper.minesweeper_env import MinesweeperEnv

# ── Game Registry ─────────────────────────────────────────────────────────────
# Each entry defines everything the launcher needs to know about a game.
# icon_chars = list of strings used to draw the pixel art icon on the card

GAME_REGISTRY = {
    "Minesweeper": {
        "env_class":   MinesweeperEnv,
        "description": "Classic mine-finding puzzle",
        "difficulties": ["Beginner", "Medium", "Hard"],
        "icon_chars":  [
            "  💣💣💣  ",
            " 💣   💣 ",
            "💣  💥  💣",
            " 💣   💣 ",
            "  💣💣💣  ",
        ],
        "card_color":  (46, 46, 80),
        "accent":      (255, 100, 100),
    },
    # ── Add future games below this line ─────────────────────────────────────
    # "Super Mario": {
    #     "env_class":   MarioEnv,
    #     "description": "Classic NES platformer",
    #     "difficulties": ["World 1-1", "World 1-2"],
    #     "icon_chars":  [...],
    #     "card_color":  (180, 60, 60),
    #     "accent":      (255, 200, 0),
    # },
}
