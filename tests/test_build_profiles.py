"""
test_build_profiles.py
────────────────────────────────────────────────────────────────
Plain-assert checks for analysis/build_profiles.py (claude-34, step 5.5).
Run from the repo root:

    python -m tests.test_build_profiles

Prints one line per check and "RESULT: OK" when all pass. Uses made-up rows only.
────────────────────────────────────────────────────────────────
"""

from analysis import build_profiles as bp
from analysis import load_exports as le
from training_data import ATTACK_TYPES


def row(person, category, legit, correct, action="blocked", ms=3000):
    return le.normalise({
        "participant_code": person, "attack_log_id": "", "threat_type": category,
        "is_legitimate_item": legit, "is_correct": correct, "player_action": action,
        "response_time_ms": ms, "map_name": "Home", "phase": "run",
    })


def rows_for(person, wrong_threats, right_threats, wrong_safe=0, right_safe=0, category="Phishing"):
    rows = []
    rows += [row(person, category, False, False) for _ in range(wrong_threats)]
    rows += [row(person, category, False, True) for _ in range(right_threats)]
    rows += [row(person, category, True, False) for _ in range(wrong_safe)]
    rows += [row(person, category, True, True) for _ in range(right_safe)]
    return rows


def check_counts():
    people = bp.count_by_person(rows_for("A", 2, 8, 1, 4))
    cell = people["A"]["Phishing"]
    assert cell == {"threat": 10, "threat_wrong": 2, "safe": 5, "safe_wrong": 1}, cell
    assert people["A"]["Ransomware"]["threat"] == 0
    print("ok  threat and safe cards are counted as seen and wrong")


def check_shrinking():
    assert bp.shrunk_rate(0, 0, 0.4) == 0.4, "no cards: the group rate"
    few = bp.shrunk_rate(3, 3, 0.2)  # 3 cards, all wrong
    many = bp.shrunk_rate(300, 300, 0.2)
    assert 0.2 < few < 0.6 < many, (few, many)
    assert many > 0.97, "a lot of data hardly moves"
    print("ok  few cards lean on the group rate, many cards speak for themselves")


def check_groups():
    rows = []
    for index, wrong in enumerate([2, 5, 10, 20, 30, 40]):
        rows += rows_for("P%d" % index, wrong, 50 - wrong)
    rows += rows_for("tiny", 1, 1)  # 2 cards: too few to rank
    people = bp.count_by_person(rows)
    groups = bp.split_into_groups(people)
    assert "tiny" not in sum(groups.values(), []), "a person with almost no cards is not ranked"
    assert sorted(groups["expert"]) == ["P0", "P1"], groups
    assert sorted(groups["beginner"]) == ["P4", "P5"], groups
    assert sorted(sum(groups.values(), [])) == ["P%d" % i for i in range(6)]
    print("ok  people are cut into beginner, intermediate and expert by accuracy; tiny samples are left out")


def check_profile_shape():
    rows = rows_for("A", 20, 20) + rows_for("B", 5, 35) + rows_for("C", 1, 39)
    report = bp.build(rows)
    assert set(report["profiles"]) == {"beginner", "intermediate", "expert"}
    for name, profile in report["profiles"].items():
        assert len(profile["weakness"]) == 8 and len(profile["fp_rate"]) == 8, name
        assert all(0.0 <= v <= 1.0 for v in profile["weakness"] + profile["fp_rate"]), name
        assert profile["fp_rate"][ATTACK_TYPES.index("Credential Theft")] == bp.NO_SAFE_CARDS_FP
        assert profile["source"] == "real"
    assert report["profiles"]["beginner"]["weakness"][0] > report["profiles"]["expert"]["weakness"][0]
    print("ok  each real profile has 8 weakness and 8 false-positive numbers in 0 to 1, in the game's order")


def check_missed_alerts_are_not_response_times():
    rows = [row("A", "Phishing", False, True, "blocked", 2000), row("A", "Phishing", False, False, "ignored", 5000)]
    assert bp.response_times_of(rows, ["A"]) == [2000], "a missed alert is the alert's timeout, not thinking time"
    print("ok  missed alerts are left out of the response time")


def check_report_text():
    rows = rows_for("A", 20, 20) + rows_for("B", 5, 35) + rows_for("C", 1, 39)
    text = bp.format_report(bp.build(rows))
    assert "real beginner" in text and "invented" in text and "Average chance to fail" in text
    print("ok  the report puts the real and the invented profiles side by side")


if __name__ == "__main__":
    check_counts()
    check_shrinking()
    check_groups()
    check_profile_shape()
    check_missed_alerts_are_not_response_times()
    check_report_text()
    print("RESULT: OK")
