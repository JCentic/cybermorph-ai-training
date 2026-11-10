"""
test_load_exports.py
────────────────────────────────────────────────────────────────
Plain-assert checks for analysis/load_exports.py (claude-34, step 5.5).
Run from the repo root:

    python -m tests.test_load_exports

Prints one line per check and "RESULT: OK" when all pass. Uses temporary
files only; it never reads or writes data/.
────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import tempfile

from analysis import load_exports as le
from training_data import ATTACK_TYPES


def row(log_id, category="Phishing", legit=False, level="beginner", correct=True, tutorial=False, map_name="Home", person="AAAAAA"):
    return {
        "participant_code": person, "session_id": "s1", "attack_log_id": log_id, "threat_type": category,
        "is_legitimate_item": legit, "scenario_difficulty": level, "player_action": "blocked", "is_correct": correct,
        "response_time_ms": 900, "is_tutorial": tutorial, "is_baseline": False, "phase": "run", "map_name": map_name,
        "skill_level_at_time": "beginner", "build_version": "2.6 (build 17)", "logged_at": "2026-11-01T00:00:00Z",
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_types_from_csv_and_json():
    with tempfile.TemporaryDirectory() as folder:
        sample = [row("a1", legit=True, correct=False), row("a2")]
        write_csv(os.path.join(folder, "one.csv"), sample)
        with open(os.path.join(folder, "two.json"), "w", encoding="utf-8") as f:
            json.dump({"participant_code": "BBBBBB", "rows": [row("b1", person="BBBBBB")]}, f)
        rows, read, failed = le.load_rows(folder)
        assert read == ["one.csv", "two.json"] and not failed, (read, failed)
        assert len(rows) == 3
        first = rows[0]
        assert first["is_legitimate_item"] is True and first["is_correct"] is False, "csv True/False words must become booleans"
        assert first["response_time_ms"] == 900 and isinstance(first["response_time_ms"], int)
    print("ok  csv words and json booleans both become real booleans and whole numbers")


def check_unreadable_file_is_reported_not_fatal():
    with tempfile.TemporaryDirectory() as folder:
        write_csv(os.path.join(folder, "good.csv"), [row("a1")])
        with open(os.path.join(folder, "bad.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        with open(os.path.join(folder, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("ignored")
        rows, read, failed = le.load_rows(folder)
        assert read == ["good.csv"] and len(failed) == 1 and failed[0].startswith("bad.json"), (read, failed)
        assert len(rows) == 1
    print("ok  a bad file is listed and skipped, other file types are ignored")


def check_missing_folder_is_empty():
    assert le.load_rows(os.path.join(tempfile.gettempdir(), "no-such-folder-cm")) == ([], [], [])
    print("ok  a missing folder gives nothing and no crash")


def check_clean_rows():
    rows = [
        le.normalise(row("a1")),
        le.normalise(row("a1")),  # the same answer again
        le.normalise(row("a2", tutorial=True)),
        le.normalise(row("a3", category="")),
        le.normalise(row("a4", category="Rogue Software")),  # an old-taxonomy name
        le.normalise(row("a5", category="Ransomware")),
        le.normalise(row("")),  # no id: always kept, like decode_export.py
        le.normalise(row("")),
    ]
    kept, dropped = le.clean_rows(rows)
    assert [r["attack_log_id"] for r in kept] == ["a1", "a5", "", ""], [r["attack_log_id"] for r in kept]
    assert dropped == {le.DROP_DUPLICATE: 1, le.DROP_TUTORIAL: 1, le.DROP_CATEGORY: 2}, dropped
    print("ok  duplicates, tutorial rows and unknown or empty categories are dropped and counted")


def check_summary_counts():
    rows = [le.normalise(r) for r in [
        row("1", "Phishing", False, "beginner", map_name="Home", person="P1"),
        row("2", "Phishing", True, "expert", map_name="Office", person="P1"),
        row("3", "Ransomware", False, "intermediate", map_name="Public Park", person="P2"),
    ]]
    s = le.summarise(rows)
    assert s["decisions"] == 3 and s["people"] == {"P1": 2, "P2": 1}
    assert s["reached"] == {"Home": 1, "Office": 1, "Internet Cafe": 0, "Public Park": 1}, s["reached"]
    ph = s["categories"]["Phishing"]
    assert (ph["total"], ph["threat"], ph["safe"]) == (2, 1, 1) and ph["levels"] == {"beginner": 1, "intermediate": 0, "expert": 1}
    assert list(s["categories"]) == ATTACK_TYPES, "categories must come in the game's order, all 8, even with no rows"
    assert s["thin"] == ATTACK_TYPES, "with 3 rows every category is under the minimum"
    print("ok  people, maps reached, threat and safe cards and card levels are counted; all 8 categories listed in order")


def check_report_text():
    kept, dropped = le.clean_rows([le.normalise(row("1", person="SECRET1"))])
    text = le.format_report(le.summarise(kept), ["x.csv"], [], dropped)
    assert "SECRET1" not in text, "participant codes must not be printed"
    assert "<- under %d" % le.MIN_DECISIONS in text and "not yet" in text
    assert "Phishing" in text and "Ransomware" in text
    print("ok  the report flags thin categories, says 'not yet' for small data, and prints no participant code")


def check_enough_data_says_yes():
    rows = []
    for i in range(le.TARGET_PEOPLE):
        for k, name in enumerate(ATTACK_TYPES):
            for j in range(le.MIN_DECISIONS // le.TARGET_PEOPLE + 1):
                rows.append(le.normalise(row("%d-%d-%d" % (i, k, j), name, False, "beginner", map_name="Public Park", person="P%d" % i)))
    kept, dropped = le.clean_rows(rows)
    text = le.format_report(le.summarise(kept), ["x.csv"], [], dropped)
    assert "Enough for real profiles: yes" in text, text
    assert "Every category has at least" in text
    print("ok  enough people, last-map players and decisions say yes")


def check_main_with_missing_folder():
    assert le.main([os.path.join(tempfile.gettempdir(), "no-such-folder-cm")]) == 1
    print("ok  main() with no folder returns 1")


if __name__ == "__main__":
    check_types_from_csv_and_json()
    check_unreadable_file_is_reported_not_fatal()
    check_missing_folder_is_empty()
    check_clean_rows()
    check_summary_counts()
    check_report_text()
    check_enough_data_says_yes()
    check_main_with_missing_folder()
    print("RESULT: OK")
