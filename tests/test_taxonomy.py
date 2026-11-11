"""
test_taxonomy.py
────────────────────────────────────────────────────────────────
Plain-assert checks for claude-32: the 8 DICT categories, in the
game's order, and the 5 profiles written against them.
Run from the repo root:

    python -m tests.test_taxonomy

Prints one line per check and "RESULT: OK" when all pass.
────────────────────────────────────────────────────────────────
"""

import os
import re

from environment.cybermorph_env import CyberMorphEnv
from training_data import ATTACK_TYPES, DERIVED_WEAKNESS


EXPECTED = [
    "Phishing",
    "Smishing",
    "Vishing",
    "Social Engineering",
    "Credential Theft",
    "Public Wi-Fi Attack",
    "Malware Infection",
    "Ransomware",
]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_CONFIG = os.path.join(os.path.dirname(REPO), "cyber-morph-game", "util", "Config.gd")

# Words of the old taxonomy. Built from pieces so this file does not contain them.
OLD_WORDS = ["rog" + "ue", "malver" + "tising", "dd" + "os", "drive" + "by", "mi" + "tm", "password" + "_attack"]


def check_attack_types():
    assert ATTACK_TYPES == EXPECTED, "ATTACK_TYPES is not the 8 categories in the game's order"
    print("ok  ATTACK_TYPES is the 8 categories in order")


def check_matches_the_game():
    # Drift guard: compare with Config.THREAT_CATEGORIES when the game repo is next door.
    if not os.path.exists(GAME_CONFIG):
        print("--  game Config.gd not found next to this repo, drift check skipped")
        return
    with open(GAME_CONFIG, encoding="utf-8") as f:
        text = f.read()
    block = re.search(r"THREAT_CATEGORIES[^=]*=\s*\[(.*?)\]", text, re.S)
    assert block, "could not find THREAT_CATEGORIES in Config.gd"
    game_list = re.findall(r'"([^"]+)"', block.group(1))
    assert game_list == ATTACK_TYPES, f"game list {game_list} differs from ATTACK_TYPES"
    print("ok  ATTACK_TYPES equals Config.THREAT_CATEGORIES in the game")


def check_profiles():
    assert len(DERIVED_WEAKNESS) == 5, "expected 5 profiles"
    for key, profile in DERIVED_WEAKNESS.items():
        for field in ("weakness", "fp_rate"):
            values = profile[field]
            assert len(values) == 8, f"{key} {field} does not have 8 numbers"
            assert all(0.0 <= v <= 1.0 for v in values), f"{key} {field} left 0 to 1"
        # No safe cards exist for Credential Theft (4) and Public Wi-Fi (5): fixed value.
        assert profile["fp_rate"][4] == profile["fp_rate"][5] == 0.05, f"{key} fp_rate 4 and 5 must be 0.05"
    print("ok  5 profiles, 8 numbers each, all in 0 to 1, fixed fp_rate where no safe cards exist")


def check_profile_stories():
    by_name = {p["username"]: p for p in DERIVED_WEAKNESS.values()}
    names = {"JuanDelaCruz": 0, "MariaClara": 4, "KardingGamer": 5, "BenjiPH": 7}
    for name, index in names.items():
        weakness = by_name[name]["weakness"]
        peak = max(range(8), key=lambda i: weakness[i])
        if name == "JuanDelaCruz":
            assert peak == 0, "Juan should be weakest on Phishing"
        else:
            assert peak == index, f"{name} should be weakest on {ATTACK_TYPES[index]}"
    rosa = by_name["TeacherRosa"]
    assert max(rosa["fp_rate"]) >= 0.5, "Rosa should be an over-blocker"
    print("ok  each profile's weakest category matches its story")


def check_env_names():
    env = CyberMorphEnv()
    env.reset(seed=3)
    for action in range(16):
        info = env.step(action)[4]
        assert info["attack_name"] == ATTACK_TYPES[action // 2], f"wrong name for action {action}"
    print("ok  step() reports the category name in info")


def check_no_old_words():
    pattern = re.compile("|".join(OLD_WORDS), re.I)
    found = []
    for folder, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in ("venv", ".git", "__pycache__")]
        for name in files:
            # test_load_exports.py uses an old name on purpose: it checks that old exports are still read.
            if name.endswith(".py") and name != "test_load_exports.py":
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    if pattern.search(f.read()):
                        found.append(os.path.relpath(path, REPO))
    assert not found, f"old category words still in: {found}"
    print("ok  no old category words left in the .py files")


if __name__ == "__main__":
    check_attack_types()
    check_matches_the_game()
    check_profiles()
    check_profile_stories()
    check_env_names()
    check_no_old_words()
    print("RESULT: OK")
