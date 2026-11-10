"""
load_exports.py
────────────────────────────────────────────────────────────────
Reads the testers' play data (claude-34, step 5.5) and says what is in it.

The game's "Share my play data" log reaches this repo as files in data/
(that folder is in .gitignore: it has participant codes, never commit it):

  - .csv  made by  decode_export.py --csv   (one row per decision)
  - .json made by  decode_export.py --json  (a list of the same rows)

Run from the repo root:

    python -m analysis.load_exports            # reads data/
    python -m analysis.load_exports some_dir   # reads another folder

What it does, in order:
  1. reads every .csv and .json file in the folder;
  2. keeps an answer that arrives twice (same attack_log_id) once;
  3. drops tutorial rows and rows with a category the game does not know;
  4. prints the people, decisions and maps reached, and per category the
     decisions (threat and safe cards, and per card level);
  5. says plainly which categories have fewer than MIN_DECISIONS decisions,
     and whether the plan's target number of testers is reached.

Standard library only. build_profiles.py (next step) imports load_rows()
and clean_rows() from here.
────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import sys

from training_data import ATTACK_TYPES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = os.path.join(REPO, "data")

# The maps in the game's order (Config / MapCatalog); map k offers the first 2k categories.
MAP_ORDER = ["Home", "Office", "Internet Cafe", "Public Park"]
LEVELS = ["beginner", "intermediate", "expert"]

# claude-34: below about 100 decisions a category is a comparison, not a training source.
MIN_DECISIONS = 100
# claude-34 / finish-progress 5.5: about 15 to 20 testers, at least 8 through the last map.
TARGET_PEOPLE = 15
TARGET_LAST_MAP = 8

DROP_TUTORIAL = "tutorial row"
DROP_CATEGORY = "no category or an unknown one"
DROP_DUPLICATE = "answer that arrived twice"


def _to_bool(value):
    """The csv holds the words True and False, the json holds real booleans."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _to_int(value):
    """A whole number, or None when the cell is empty or not a number."""
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def normalise(row):
    """One row with real types. Missing columns become empty text or None; nothing is invented."""
    return {
        "participant_code": str(row.get("participant_code", "")).strip(),
        "session_id": str(row.get("session_id", "")).strip(),
        "attack_log_id": str(row.get("attack_log_id", "")).strip(),
        "threat_type": str(row.get("threat_type", "")).strip(),
        "is_legitimate_item": _to_bool(row.get("is_legitimate_item", False)),
        "scenario_difficulty": str(row.get("scenario_difficulty", "")).strip(),
        "player_action": str(row.get("player_action", "")).strip(),
        "is_correct": _to_bool(row.get("is_correct", False)),
        "response_time_ms": _to_int(row.get("response_time_ms", "")),
        "is_tutorial": _to_bool(row.get("is_tutorial", False)),
        "is_baseline": _to_bool(row.get("is_baseline", False)),
        "phase": str(row.get("phase", "")).strip(),
        "map_name": str(row.get("map_name", "")).strip(),
        "skill_level_at_time": str(row.get("skill_level_at_time", "")).strip(),
        "build_version": str(row.get("build_version", "")).strip(),
        "logged_at": str(row.get("logged_at", "")).strip(),
    }


def read_file(path):
    """The rows of one .csv or .json file, normalised. Raises ValueError for a file that is not a list of rows."""
    with open(path, encoding="utf-8", newline="") as f:
        if path.lower().endswith(".csv"):
            return [normalise(row) for row in csv.DictReader(f)]
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("rows", [])  # one export as the game makes it: {"participant_code", "rows"}
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("not a list of rows")
    return [normalise(row) for row in data]


def load_rows(folder=DEFAULT_FOLDER):
    """Every row of every .csv and .json in the folder, in file-name order. Returns (rows, files_read, files_failed)."""
    rows, read, failed = [], [], []
    if not os.path.isdir(folder):
        return rows, read, failed
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".csv", ".json")):
            continue
        try:
            rows.extend(read_file(os.path.join(folder, name)))
            read.append(name)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed.append("%s (%s)" % (name, error))
    return rows, read, failed


def clean_rows(rows):
    """Drops duplicates, tutorial rows and rows with no known category. Returns (kept, dropped) where dropped counts each reason."""
    dropped = {DROP_DUPLICATE: 0, DROP_TUTORIAL: 0, DROP_CATEGORY: 0}
    seen, kept = set(), []
    for row in rows:
        log_id = row["attack_log_id"]
        if log_id:
            if log_id in seen:
                dropped[DROP_DUPLICATE] += 1
                continue
            seen.add(log_id)
        if row["is_tutorial"]:
            dropped[DROP_TUTORIAL] += 1
            continue
        if row["threat_type"] not in ATTACK_TYPES:
            dropped[DROP_CATEGORY] += 1
            continue
        kept.append(row)
    return kept, dropped


def summarise(rows):
    """The numbers of the report, from clean rows."""
    people = {}
    for row in rows:
        people[row["participant_code"]] = people.get(row["participant_code"], 0) + 1
    reached = {m: len({r["participant_code"] for r in rows if r["map_name"] == m}) for m in MAP_ORDER}
    categories = {}
    for name in ATTACK_TYPES:
        mine = [r for r in rows if r["threat_type"] == name]
        categories[name] = {
            "total": len(mine),
            "threat": sum(1 for r in mine if not r["is_legitimate_item"]),
            "safe": sum(1 for r in mine if r["is_legitimate_item"]),
            "levels": {lv: sum(1 for r in mine if r["scenario_difficulty"] == lv) for lv in LEVELS},
        }
    return {
        "decisions": len(rows),
        "people": people,
        "reached": reached,
        "categories": categories,
        "phases": {p: sum(1 for r in rows if r["phase"] == p) for p in sorted({r["phase"] for r in rows})},
        "builds": {b: sum(1 for r in rows if r["build_version"] == b) for b in sorted({r["build_version"] for r in rows})},
        "thin": [name for name, c in categories.items() if c["total"] < MIN_DECISIONS],
    }


def format_report(summary, files_read, files_failed, dropped):
    """The text printed by main(). Participant codes are not printed, only how many decisions each person has."""
    lines = []
    lines.append("files read: %d (%s)" % (len(files_read), ", ".join(files_read) or "none"))
    for item in files_failed:
        lines.append("could not read: " + item)
    for reason, count in dropped.items():
        if count:
            lines.append("dropped %d: %s" % (count, reason))
    people = summary["people"]
    lines.append("")
    lines.append("decisions kept: %d from %d people" % (summary["decisions"], len(people)))
    lines.append("decisions per person: " + (", ".join(str(n) for n in sorted(people.values(), reverse=True)) or "none"))
    lines.append("people who played each map: " + ", ".join("%s %d" % (m, n) for m, n in summary["reached"].items()))
    lines.append("by phase: " + ", ".join("%s %d" % (p or "(none)", n) for p, n in summary["phases"].items()))
    lines.append("by build: " + ", ".join("%s %d" % (b or "(none)", n) for b, n in summary["builds"].items()))
    lines.append("")
    lines.append("%-20s %6s %7s %5s   %9s %12s %6s" % ("category", "total", "threat", "safe", "beginner", "intermediate", "expert"))
    for name, c in summary["categories"].items():
        flag = "   <- under %d" % MIN_DECISIONS if c["total"] < MIN_DECISIONS else ""
        lv = c["levels"]
        lines.append("%-20s %6d %7d %5d   %9d %12d %6d%s" % (name, c["total"], c["threat"], c["safe"], lv["beginner"], lv["intermediate"], lv["expert"], flag))
    lines.append("")
    if summary["thin"]:
        lines.append("Under %d decisions (a comparison, not a training source): %s." % (MIN_DECISIONS, ", ".join(summary["thin"])))
    else:
        lines.append("Every category has at least %d decisions." % MIN_DECISIONS)
    last = summary["reached"].get(MAP_ORDER[-1], 0)
    lines.append("People: %d of about %d wanted. Through the last map: %d of %d wanted." % (len(people), TARGET_PEOPLE, last, TARGET_LAST_MAP))
    enough = len(people) >= TARGET_PEOPLE and last >= TARGET_LAST_MAP and not summary["thin"]
    lines.append("Enough for real profiles: " + ("yes" if enough else "not yet, keep the five invented profiles and use this as a comparison"))
    return "\n".join(lines)


def main(argv):
    folder = argv[0] if argv else DEFAULT_FOLDER
    if not os.path.isdir(folder):
        print("No data folder at %s. Put the exports there (decode_export.py --fetch --csv ...)." % folder)
        return 1
    rows, read, failed = load_rows(folder)
    kept, dropped = clean_rows(rows)
    print(format_report(summarise(kept), read, failed, dropped))
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
