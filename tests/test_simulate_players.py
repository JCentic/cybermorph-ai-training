"""
test_simulate_players.py - checks of analysis/simulate_players.py. Run from the repo root:  python -m tests.test_simulate_players
"""

import numpy as np

from analysis.build_profiles import blank_counts
from analysis.simulate_players import fit_logistic, simulate
from training_data import ATTACK_TYPES


def make_people(rng, n_people=8, cards=150):
    """Counts made from known rates: player level l, category hardness h, P(wrong) = sigmoid(l + h)."""
    hardness = np.linspace(-1.2, 0.6, 8)
    levels = np.linspace(-0.8, 0.8, n_people)
    people = {}
    for p in range(n_people):
        counts = blank_counts()
        for c, name in enumerate(ATTACK_TYPES):
            rate = 1 / (1 + np.exp(-(levels[p] + hardness[c])))
            counts[name]["threat"] = cards
            counts[name]["threat_wrong"] = int(rng.binomial(cards, rate))
            if c not in (4, 5):
                counts[name]["safe"] = cards
                counts[name]["safe_wrong"] = int(rng.binomial(cards, rate))
        people["p%d" % p] = counts
    return people, hardness


def check_fit_recovers_known_category_rates():
    rng = np.random.default_rng(1)
    people, hardness = make_people(rng)
    codes, cats, theta, cov = fit_logistic(people, "threat")
    fitted = theta[len(codes):]
    order_true, order_fit = np.argsort(hardness), np.argsort(fitted)
    assert list(order_true) == list(order_fit), "the fit orders the categories by difficulty like the truth"
    assert np.corrcoef(hardness, fitted)[0, 1] > 0.98
    print("ok  the fit recovers the order and size of the category difficulty")


def check_thin_categories_are_more_uncertain():
    rng = np.random.default_rng(2)
    people, _ = make_people(rng)
    for counts in people.values():
        counts[ATTACK_TYPES[7]]["threat"] = 3
        counts[ATTACK_TYPES[7]]["threat_wrong"] = 1
    _, cats, theta, cov = fit_logistic(people, "threat")
    n_p = len(people)
    sd = np.sqrt(np.diag(cov))[n_p:]
    assert sd[7] > 1.5 * sd[0], "a category with few cards must have a wider uncertainty"
    print("ok  a category with few cards is more uncertain than one with many")


def check_simulated_players_are_valid():
    rng = np.random.default_rng(3)
    people, _ = make_people(rng)
    w, f, _ = simulate(people, 40, np.random.default_rng(4))
    assert w.shape == (40, 8) and f.shape == (40, 8)
    assert np.all((w >= 0.02) & (w <= 0.95)) and np.all((f >= 0.02) & (f <= 0.95))
    assert np.allclose(f[:, 4], 0.05) and np.allclose(f[:, 5], 0.05), "no safe cards exist for Credential Theft and Public Wi-Fi"
    assert w.std(axis=0).min() > 0.01, "the players differ from each other"
    print("ok  simulated players are in range, differ from each other, and keep the fixed rate where no safe card exists")


if __name__ == "__main__":
    check_fit_recovers_known_category_rates()
    check_thin_categories_are_more_uncertain()
    check_simulated_players_are_valid()
    print("RESULT: OK")
