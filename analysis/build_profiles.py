"""
build_profiles.py
────────────────────────────────────────────────────────────────
Step 5.5 of the plan (claude-34): turns the testers' real play data into
three "real" player profiles, and compares them with the five invented
profiles that the training uses today.

Run from the repo root:

    python -m analysis.build_profiles              # reads data/, prints the report
    python -m analysis.build_profiles --write      # also writes data/profiles_real.json

What it does, in simple words:
  1. loads the clean decisions (analysis/load_exports.py);
  2. for every person and every attack type, counts how many threat cards
     the person got wrong, and how many safe cards the person wrongly blocked;
  3. "shrinks" a person's rate toward the group's rate when the person saw
     only a few cards (a rate from 3 cards is a guess, not a fact);
  4. sorts the people by how well they did and cuts them into three groups
     (beginner, intermediate, expert): the "real" profiles;
  5. prints the real profiles next to the invented ones, so we can see how far
     the invented ones are from real players.

It never changes training_data.py. The training keeps using the five invented
profiles until there is enough data (the loader says so). Standard library only.
A missed alert (player_action "ignored") is NOT a response time: the game
records the alert's timeout (5000 ms) for it, so it is left out of the time.
────────────────────────────────────────────────────────────────
"""

import json
import os
import sys

from analysis.load_exports import DEFAULT_FOLDER, MAP_ORDER, clean_rows, load_rows  # noqa: F401
from training_data import ATTACK_TYPES, DERIVED_WEAKNESS

# How many "imaginary cards" of the group's rate a person starts with. A person with 3 cards is pulled
# strongly toward the group; a person with 100 cards hardly moves. (Beta prior.)
PRIOR_WEIGHT = 8.0
# A category with no safe cards at all (Credential Theft, Public Wi-Fi) gets this fixed false-positive rate,
# the same as the invented profiles use.
NO_SAFE_CARDS_FP = 0.05
# The three groups, from weakest to strongest player. The names are the game's levels.
GROUP_NAMES = ["beginner", "intermediate", "expert"]
IGNORED = "ignored"


def blank_counts():
    """The counts of one person (or one group): per attack type, threat and safe cards seen and got wrong."""
    return {name: {"threat": 0, "threat_wrong": 0, "safe": 0, "safe_wrong": 0} for name in ATTACK_TYPES}


def count_by_person(rows):
    """{participant_code: counts}. A threat card is wrong when the player failed it (let it through or missed it);
    a safe card is wrong when the player blocked it."""
    people = {}
    for row in rows:
        counts = people.setdefault(row["participant_code"], blank_counts())
        cell = counts[row["threat_type"]]
        if row["is_legitimate_item"]:
            cell["safe"] += 1
            cell["safe_wrong"] += 0 if row["is_correct"] else 1
        else:
            cell["threat"] += 1
            cell["threat_wrong"] += 0 if row["is_correct"] else 1
    return people


def shrunk_rate(wrong, seen, group_rate, weight=PRIOR_WEIGHT):
    """(wrong + weight * group_rate) / (seen + weight). With no cards seen, it is the group rate."""
    return (wrong + weight * group_rate) / (seen + weight)


def person_accuracy(counts):
    """Share of all this person's cards answered right (0 to 1), or None with no cards."""
    seen = sum(c["threat"] + c["safe"] for c in counts.values())
    wrong = sum(c["threat_wrong"] + c["safe_wrong"] for c in counts.values())
    return None if seen == 0 else 1.0 - wrong / seen


def split_into_groups(people, minimum_cards=30):
    """Sorts the people from the weakest to the strongest by accuracy and cuts them into three groups of about the same size.
    A person with fewer than `minimum_cards` cards is left out (too little to rank). Returns {group_name: [codes]}."""
    ranked = []
    for code, counts in people.items():
        seen = sum(c["threat"] + c["safe"] for c in counts.values())
        accuracy = person_accuracy(counts)
        if accuracy is not None and seen >= minimum_cards:
            ranked.append((accuracy, code))
    ranked.sort()
    groups = {name: [] for name in GROUP_NAMES}
    total = len(ranked)
    for position, (_accuracy, code) in enumerate(ranked):
        index = min(position * len(GROUP_NAMES) // max(total, 1), len(GROUP_NAMES) - 1)
        groups[GROUP_NAMES[index]].append(code)
    return groups


def pool(people, codes):
    """The counts of a whole group: every person's counts added up."""
    total = blank_counts()
    for code in codes:
        for name in ATTACK_TYPES:
            for key in total[name]:
                total[name][key] += people[code][name][key]
    return total


def overall_rates(counts):
    """(threat_wrong_rate, safe_wrong_rate) over all categories of `counts`, or None for a part with no cards."""
    threat = sum(c["threat"] for c in counts.values())
    safe = sum(c["safe"] for c in counts.values())
    threat_wrong = sum(c["threat_wrong"] for c in counts.values())
    safe_wrong = sum(c["safe_wrong"] for c in counts.values())
    return (threat_wrong / threat if threat else None, safe_wrong / safe if safe else None)


def profile_of(counts, everyone, response_times):
    """One profile in the shape of DERIVED_WEAKNESS: 8 weakness numbers and 8 false-positive numbers in ATTACK_TYPES order, plus the mean response time.
    A category with few cards in this group is pulled toward the group's own overall rate (the group's average over all categories)."""
    group_threat, group_safe = overall_rates(counts)
    all_threat, all_safe = overall_rates(everyone)
    group_threat = all_threat if group_threat is None else group_threat
    group_safe = all_safe if group_safe is None else group_safe
    weakness, fp_rate = [], []
    for name in ATTACK_TYPES:
        cell = counts[name]
        weakness.append(round(shrunk_rate(cell["threat_wrong"], cell["threat"], group_threat if group_threat is not None else 0.5), 3))
        if cell["safe"] == 0 and everyone[name]["safe"] == 0:
            fp_rate.append(NO_SAFE_CARDS_FP)  # no safe card of this type exists in the game
        else:
            fp_rate.append(round(shrunk_rate(cell["safe_wrong"], cell["safe"], group_safe if group_safe is not None else 0.2), 3))
    mean_time = round(sum(response_times) / len(response_times)) if response_times else None
    return {"weakness": weakness, "fp_rate": fp_rate, "avg_response_time_ms": mean_time}


def response_times_of(rows, codes):
    """The real thinking times (ms) of these people: a missed alert is left out, because its time is just the alert's timeout."""
    wanted = set(codes)
    return [r["response_time_ms"] for r in rows
            if r["participant_code"] in wanted and r["player_action"] != IGNORED and r["response_time_ms"] and r["response_time_ms"] > 0]


def build(rows):
    """All the numbers of the report. Returns a dict: groups, profiles, counts per group, the people count."""
    people = count_by_person(rows)
    groups = split_into_groups(people)
    everyone = pool(people, list(people))
    profiles, sizes = {}, {}
    for name in GROUP_NAMES:
        codes = groups[name]
        if not codes:
            continue
        counts = pool(people, codes)
        profile = profile_of(counts, everyone, response_times_of(rows, codes))
        profile["source"] = "real"
        profile["people"] = len(codes)
        profile["cards"] = sum(c["threat"] + c["safe"] for c in counts.values())
        profile["cards_per_category"] = [counts[n]["threat"] + counts[n]["safe"] for n in ATTACK_TYPES]
        profiles[name] = profile
        sizes[name] = len(codes)
    return {"people": len(people), "groups": sizes, "profiles": profiles, "overall": overall_rates(everyone)}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def calibration_notes(report):
    """Plain sentences about how far the invented profiles are from the real ones."""
    notes = []
    real = report["profiles"]
    if not real:
        return ["There is no real profile to compare with yet."]
    real_means = {name: mean(p["weakness"]) for name, p in real.items()}
    invented_means = {key: mean(p["weakness"]) for key, p in DERIVED_WEAKNESS.items()}
    real_average = mean(list(real_means.values()))
    invented_average = mean(list(invented_means.values()))
    notes.append("Average chance to fail a threat card: real players %.0f%%, invented profiles %.0f%%." % (real_average * 100, invented_average * 100))
    if invented_average > real_average + 0.1:
        scale = real_average / invented_average
        notes.append("The invented players fail much more than the real ones. A scale of about %.2f on every invented weakness number would bring the averages together." % scale)
    elif invented_average < real_average - 0.1:
        notes.append("The invented players fail much less than the real ones.")
    else:
        notes.append("The averages are close.")
    worst_real = max(real_means.values())
    worst_invented = max(invented_means.values())
    notes.append("The weakest real group fails %.0f%% of threat cards, the weakest invented profile %.0f%%." % (worst_real * 100, worst_invented * 100))
    thin = [ATTACK_TYPES[i] for i in range(len(ATTACK_TYPES)) if min(p["cards_per_category"][i] for p in real.values()) < 20]
    if thin:
        notes.append("Few cards for some group in: %s. Those numbers lean on the group average and are guesses." % ", ".join(thin))
    return notes


def format_report(report):
    lines = []
    lines.append("people: %d (groups: %s)" % (report["people"], ", ".join("%s %d" % (n, s) for n, s in report["groups"].items()) or "none"))
    header = "%-22s" % "weakness (threat wrong)" + "".join("%7s" % n[:6] for n in ATTACK_TYPES) + "   mean"
    lines.append("")
    lines.append(header)
    for name, p in report["profiles"].items():
        lines.append("%-22s" % ("real " + name) + "".join("%7.2f" % v for v in p["weakness"]) + "   %.2f" % mean(p["weakness"]))
    for key, p in DERIVED_WEAKNESS.items():
        lines.append("%-22s" % ("invented " + p["username"][:12]) + "".join("%7.2f" % v for v in p["weakness"]) + "   %.2f" % mean(p["weakness"]))
    lines.append("")
    lines.append("false positives (safe card blocked)")
    for name, p in report["profiles"].items():
        lines.append("%-22s" % ("real " + name) + "".join("%7.2f" % v for v in p["fp_rate"]))
    lines.append("")
    for name, p in report["profiles"].items():
        lines.append("real %-12s people %d, cards %d, mean response %s ms" % (name, p["people"], p["cards"], p["avg_response_time_ms"]))
    lines.append("")
    lines.extend(calibration_notes(report))
    return "\n".join(lines)


def main(argv):
    write = "--write" in argv
    folder = DEFAULT_FOLDER
    rows, read, failed = load_rows(folder)
    kept, _dropped = clean_rows(rows)
    if not kept:
        print("No data in %s. Put the exports there first (decode_export.py --fetch --csv ...)." % folder)
        return 1
    report = build(kept)
    print(format_report(report))
    if write:
        path = os.path.join(folder, "profiles_real.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"attack_types": ATTACK_TYPES, "profiles": report["profiles"]}, f, indent=1)
        print("\nwritten: %s (not used by the training yet)" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
