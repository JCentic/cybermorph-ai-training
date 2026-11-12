"""
simulate_players.py
────────────────────────────────────────────────────────────────
SIMULATED PLAYERS BUILT FROM THE REAL TESTERS' STATISTICS (not invented numbers, not fake rows)

Why: the five invented training players are not based on data, and the testers' data is uneven
(some categories, like Malware and Ransomware, have under 50 cards in total). Training on a
rate taken from 20 cards would teach the agent noise.

What it does, in simple words:
  1. Some testers are HELD OUT (never used here, kept for the final test). Default: QHMNY3, 7DS5GT, K8DFGB.
  2. From the other testers it fits a small statistical model (a logistic model):
         chance of a wrong answer = sigmoid( how weak THIS PLAYER is  +  how hard THIS CATEGORY is )
     Threat cards and safe cards (wrongly blocking a safe card) get a model each.
     A category with few cards still gets a number, because the model borrows from the players'
     overall level and the other categories, but it is UNCERTAIN, and the model knows how uncertain.
  3. It makes N simulated players. For each one it draws a player level from the real spread of levels,
     and the difficulty of every category from what the data supports, WIDE where the data is thin.
     So the agent practises on many plausible versions of Malware and Ransomware, not on one guess.
  4. It writes data/profiles_simulated.json (the training players) and
     data/profiles_holdout.json (the held-out real testers, as individual profiles, for the final test).

Every simulated player is labelled "source": "simulated". Nothing here is written to a tester CSV,
and nothing here is a real person. In the defense, say: "simulated players whose weak spots and error
rates are fitted to our real testers' data; the final test uses real testers the training never saw."

Run from the repo root (venv active):

    python -m analysis.simulate_players                 # report only
    python -m analysis.simulate_players --write         # also writes the two json files
Standard library + numpy.
────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys

import numpy as np

from analysis.build_profiles import (NO_SAFE_CARDS_FP, count_by_person, overall_rates, pool, profile_of,
                                     response_times_of)
from analysis.load_exports import DEFAULT_FOLDER, clean_rows, load_rows
from training_data import ATTACK_TYPES

DEFAULT_HOLDOUT = ["QHMNY3", "7DS5GT", "K8DFGB"]
MIN_CARDS = 30            # a tester with fewer cards is left out of the fit
RIDGE_PLAYER = 1.0        # pulls player levels toward the average (bigger = more pull)
RIDGE_CATEGORY = 0.5      # pulls category difficulty toward the average
N_PLAYERS = 60
LEVEL_SPREAD_FLOOR = 0.45  # the smallest spread of player levels we allow (the real testers are only a few people)
PROFILE_FILE = os.path.join("data", "profiles_simulated.json")
HOLDOUT_FILE = os.path.join("data", "profiles_holdout.json")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(people_counts, kind):
    """Fits  P(wrong) = sigmoid(player_level[p] + category_hardness[c])  by Newton steps with a ridge penalty.
    people_counts: {code: counts}. kind: "threat" or "safe".
    Returns (codes, categories_used, theta, covariance). theta = [player levels..., category hardness...]."""
    codes = sorted(people_counts)
    cats = [i for i, n in enumerate(ATTACK_TYPES) if sum(people_counts[c][n][kind] for c in codes) > 0]
    n_p, n_c = len(codes), len(cats)
    seen = np.zeros((n_p, n_c))
    wrong = np.zeros((n_p, n_c))
    for i, code in enumerate(codes):
        for j, ci in enumerate(cats):
            cell = people_counts[code][ATTACK_TYPES[ci]]
            seen[i, j] = cell[kind]
            wrong[i, j] = cell[kind + "_wrong"]
    base = np.log(max(wrong.sum(), 1.0) / max(seen.sum() - wrong.sum(), 1.0))  # the average log-odds
    theta = np.concatenate([np.zeros(n_p), np.full(n_c, base)])
    penalty = np.concatenate([np.full(n_p, RIDGE_PLAYER), np.full(n_c, RIDGE_CATEGORY)])
    centre = np.concatenate([np.zeros(n_p), np.full(n_c, base)])
    for _ in range(60):
        eta = theta[:n_p, None] + theta[None, n_p:]
        p = sigmoid(eta)
        resid = wrong - seen * p                      # score
        w = seen * p * (1 - p)                        # curvature
        grad = np.concatenate([resid.sum(axis=1), resid.sum(axis=0)]) - penalty * (theta - centre)
        hess = np.zeros((n_p + n_c, n_p + n_c))
        hess[:n_p, :n_p] = np.diag(w.sum(axis=1))
        hess[n_p:, n_p:] = np.diag(w.sum(axis=0))
        hess[:n_p, n_p:] = w
        hess[n_p:, :n_p] = w.T
        hess += np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        theta = theta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = theta[:n_p, None] + theta[None, n_p:]
    p = sigmoid(eta)
    w = seen * p * (1 - p)
    hess = np.zeros((n_p + n_c, n_p + n_c))
    hess[:n_p, :n_p] = np.diag(w.sum(axis=1))
    hess[n_p:, n_p:] = np.diag(w.sum(axis=0))
    hess[:n_p, n_p:] = w
    hess[n_p:, :n_p] = w.T
    hess += np.diag(penalty)
    return codes, cats, theta, np.linalg.inv(hess)


def simulate(people_counts, n_players, rng):
    """Draws `n_players` (weakness list, fp list). Returns (players, report)."""
    out_w = np.full((n_players, 8), 0.0)
    out_f = np.full((n_players, 8), NO_SAFE_CARDS_FP)
    report = {}
    for kind, target in (("threat", out_w), ("safe", out_f)):
        codes, cats, theta, cov = fit_logistic(people_counts, kind)
        n_p = len(codes)
        levels = theta[:n_p]
        spread = max(float(np.std(levels, ddof=1)), LEVEL_SPREAD_FLOOR)
        mean_level = float(np.mean(levels))
        # The uncertainty of the category hardness (only the category part of the covariance).
        cat_mean = theta[n_p:]
        cat_cov = cov[n_p:, n_p:]
        chol = np.linalg.cholesky(cat_cov + 1e-9 * np.eye(len(cats)))
        for k in range(n_players):
            level = rng.normal(mean_level, spread)
            hard = cat_mean + chol @ rng.standard_normal(len(cats))
            # The player level in the fit has a mean near 0 (ridge); levels were measured relative to it.
            for j, ci in enumerate(cats):
                target[k, ci] = sigmoid(level + hard[j])
        report[kind] = {"cats": cats, "mean": sigmoid(mean_level + cat_mean), "sd_cat": np.sqrt(np.diag(cat_cov)), "spread": spread}
    for k in range(n_players):
        out_w[k] = np.clip(out_w[k], 0.02, 0.95)
        out_f[k] = np.clip(out_f[k], 0.02, 0.95)
    return out_w, out_f, report


def main(argv):
    parser = argparse.ArgumentParser(description="Simulated players fitted to the real testers' statistics.")
    parser.add_argument("--folder", default=DEFAULT_FOLDER)
    parser.add_argument("--holdout", nargs="*", default=DEFAULT_HOLDOUT, help="tester codes kept out of the fit, for the final test")
    parser.add_argument("--players", type=int, default=N_PLAYERS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    rows = clean_rows(load_rows(args.folder)[0])[0]
    people = count_by_person(rows)
    everyone = pool(people, list(people))
    cards = {c: sum(v["threat"] + v["safe"] for v in counts.values()) for c, counts in people.items()}
    held = [c for c in args.holdout if c in people]
    fit_people = {c: v for c, v in people.items() if c not in held and cards[c] >= MIN_CARDS}
    print("held out (never used in the fit):", ", ".join("%s (%d cards)" % (c, cards[c]) for c in held))
    print("used to fit the model:", ", ".join("%s (%d)" % (c, cards[c]) for c in sorted(fit_people)))

    rng = np.random.default_rng(args.seed)
    w, f, report = simulate(fit_people, args.players, rng)

    print("\ncategory                threat cards | fitted failure rate   sd of its log-odds | simulated pool: mean [10%..90%]")
    pooled = pool(fit_people, list(fit_people))
    for i, name in enumerate(ATTACK_TYPES):
        r = report["threat"]
        j = r["cats"].index(i)
        print("  %-20s %6d       | %.2f                    %.2f              | %.2f [%.2f..%.2f]" % (
            name, pooled[name]["threat"], r["mean"][j], r["sd_cat"][j], w[:, i].mean(), np.percentile(w[:, i], 10), np.percentile(w[:, i], 90)))
    print("\nsafe cards (wrongly blocking):")
    for i, name in enumerate(ATTACK_TYPES):
        r = report["safe"]
        if i not in r["cats"]:
            print("  %-20s no safe cards in the game (fixed %.2f)" % (name, NO_SAFE_CARDS_FP))
            continue
        j = r["cats"].index(i)
        print("  %-20s %6d       | %.2f                    %.2f              | %.2f [%.2f..%.2f]" % (
            name, pooled[name]["safe"], r["mean"][j], r["sd_cat"][j], f[:, i].mean(), np.percentile(f[:, i], 10), np.percentile(f[:, i], 90)))
    print("\nplayer-level spread (log-odds): threat %.2f, safe %.2f   (the invented players were far more spread out)" % (report["threat"]["spread"], report["safe"]["spread"]))
    print("mean weakness of the simulated players: %.2f (real testers overall: %.2f)" % (w[:, :].mean(), overall_rates(everyone)[0]))

    if args.write:
        times = response_times_of(rows, list(people))
        mean_time = int(np.mean(times)) if times else 3000
        profiles = {}
        for k in range(args.players):
            profiles["sim-%03d" % k] = {
                "username": "sim %03d" % k, "description": "simulated from the real testers' statistics",
                "weakness": [round(float(x), 3) for x in w[k]], "fp_rate": [round(float(x), 3) for x in f[k]],
                "avg_response_time_ms": mean_time, "source": "simulated"}
        with open(PROFILE_FILE, "w", encoding="utf-8") as fh:
            json.dump({"attack_types": ATTACK_TYPES, "held_out": held, "fit_on": sorted(fit_people), "profiles": profiles}, fh, indent=1)
        everyone_counts = pool(people, list(people))
        holdout = {}
        for code in held:
            profile = profile_of(people[code], everyone_counts, response_times_of(rows, [code]))
            profile.update({"source": "real", "people": 1, "cards": cards[code]})
            holdout["real-" + code] = profile
        with open(HOLDOUT_FILE, "w", encoding="utf-8") as fh:
            json.dump({"attack_types": ATTACK_TYPES, "profiles": holdout}, fh, indent=1)
        print("\nwritten: %s (%d simulated players) and %s (%d held-out real testers)" % (PROFILE_FILE, args.players, HOLDOUT_FILE, len(holdout)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
