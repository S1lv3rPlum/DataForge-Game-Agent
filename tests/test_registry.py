# tests/test_registry.py
# Tests for the game registry
# Run with: pytest tests/test_registry.py -v

import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from games.registry import GAME_REGISTRY

class TestRegistry:

    def test_registry_not_empty(self):
        assert len(GAME_REGISTRY) > 0

    def test_minesweeper_registered(self):
        assert "Minesweeper" in GAME_REGISTRY

    def test_all_games_have_required_fields(self):
        required = ["env_class", "description", "difficulties",
                    "icon_chars", "card_color", "accent"]
        for name, cfg in GAME_REGISTRY.items():
            for field in required:
                assert field in cfg, \
                    f"{name} missing field: {field}"

    def test_all_env_classes_instantiate(self):
        for name, cfg in GAME_REGISTRY.items():
            EnvClass = cfg["env_class"]
            env      = EnvClass(render_mode=None,
                                difficulty=cfg["difficulties"][0])
            obs, _   = env.reset()
            assert obs is not None
            env.close()

    def test_all_difficulties_valid(self):
        for name, cfg in GAME_REGISTRY.items():
            assert len(cfg["difficulties"]) > 0
            for diff in cfg["difficulties"]:
                assert isinstance(diff, str)

    def test_card_colors_are_rgb(self):
        for name, cfg in GAME_REGISTRY.items():
            assert len(cfg["card_color"]) == 3
            for val in cfg["card_color"]:
                assert 0 <= val <= 255

    def test_icon_chars_not_empty(self):
        for name, cfg in GAME_REGISTRY.items():
            assert len(cfg["icon_chars"]) > 0
