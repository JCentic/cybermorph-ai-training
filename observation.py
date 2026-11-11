"""
observation.py
────────────────────────────────────────────────────────────────
WHAT THE AGENT SEES (claude-33): estimates built from answers.

Pure Python: no gym, no numpy, no files. The same rule is ported to
GDScript for the phone (claude-28) and tested against fixtures made
from THIS file, so training and play see the same numbers.

THE WINDOW is the list of the player's most recent answers. One answer:

    (category, is_threat, correct, response_ms)

    category     0 to 7, an index into ATTACK_TYPES (training_data.py)
    is_threat    True for a threat card, False for a safe card
    correct      True if the player answered right
                 (threat card: handled it; safe card: did not block it)
    response_ms  how long the answer took

THE OBSERVATION, 20 numbers, each in 0 to 1:
    0-7    estimated weakness per category (threat cards answered wrong)
    8-15   estimated false-positive rate per category (safe cards blocked)
    16     mean response time over RESPONSE_CAP_MS
    17     recent accuracy: right answers among the last RECENT, over that count
    18     streak: right answers in a row at the end, over STREAK_CAP (capped at 1)
    19     map stage over MAX_STAGE (stage k offers the first 2k categories)

ESTIMATE of a category: (failures + prior * PRIOR_WEIGHT) / (seen + PRIOR_WEIGHT)
    The prior is worth PRIOR_WEIGHT pretend cards, so a category with no
    answers shows the prior, and real answers take over as they pile up.
────────────────────────────────────────────────────────────────
"""

N_CATEGORIES = 8
N_ACTIONS = 16          # category x (threat, safe); action // 2 is the category
MAX_STAGE = 4           # Home 1, Office 2, Internet Cafe 3, Public Park 4
WINDOW_SIZE = 50        # the game keeps the last 50 answers
PRIOR_WEIGHT = 3        # TUNABLE: pretend cards the prior is worth
RESPONSE_CAP_MS = 6000.0
RECENT = 5
STREAK_CAP = 4

# Credential Theft (4) and Public Wi-Fi Attack (5) have no safe cards in the game.
NO_SAFE_CARDS = (4, 5)

# Cold-start defaults: the mean of the five profiles in training_data.py,
# rounded to 2 decimals (tests/test_observation.py checks this).
PRIOR_WEAKNESS = (0.26, 0.29, 0.30, 0.31, 0.34, 0.41, 0.23, 0.38)
PRIOR_FP = (0.21, 0.17, 0.15, 0.18, 0.05, 0.05, 0.13, 0.11)
PRIOR_RESPONSE_MS = 2600.0


def estimate(failures, seen, prior, weight=PRIOR_WEIGHT):
    """Shrunk estimate: with seen == 0 it is the prior."""
    return (failures + prior * weight) / (seen + weight)


def action_mask(stage):
    """16 booleans, True where the action is valid on this map stage.

    An action is invalid when its category is not on the map yet (stage k
    offers categories 0 to 2k - 1), or when it is a safe card for a
    category that has none.
    """
    offered = 2 * stage
    mask = []
    for action in range(N_ACTIONS):
        category = action // 2
        is_threat = action % 2 == 0
        valid = category < offered and (is_threat or category not in NO_SAFE_CARDS)
        mask.append(valid)
    return mask


def _clip01(value):
    return max(0.0, min(1.0, value))


# --- OBSERVATION VERSION 2 (29 numbers): the 20 above, plus what a coach needs to PLAN ---
#    20-27  practice count per category: threat cards of that category among the last WINDOW_SIZE answers, over PRACTICE_CAP (capped at 1).
#           Tells the agent how sure each estimate is and which categories were not practised yet.
#    28     cards left in this run over the cards of the whole run (1.0 at the start, 0 at the end). The game knows it: it is the time left over the map's time.
PRACTICE_CAP = 10
OBS2_SIZE = 29


def build_observation_v2(window, stage, cards_left_share):
    """The 29 numbers of version 2: build_observation() plus the practice counts and the share of the run left."""
    obs = build_observation(window, stage)
    recent = list(window)[-WINDOW_SIZE:]
    practice = [0] * N_CATEGORIES
    for category, is_threat, _correct, _response_ms in recent:
        if is_threat:
            practice[category] += 1
    obs.extend(_clip01(count / PRACTICE_CAP) for count in practice)
    obs.append(_clip01(cards_left_share))
    return obs


def build_observation(window, stage):
    """The 20 numbers for this window and stage, as a list of floats."""
    recent = list(window)[-WINDOW_SIZE:]

    threat_seen = [0] * N_CATEGORIES
    threat_failed = [0] * N_CATEGORIES
    safe_seen = [0] * N_CATEGORIES
    safe_blocked = [0] * N_CATEGORIES
    for category, is_threat, correct, _response_ms in recent:
        if is_threat:
            threat_seen[category] += 1
            threat_failed[category] += 0 if correct else 1
        else:
            safe_seen[category] += 1
            safe_blocked[category] += 0 if correct else 1

    obs = []
    for c in range(N_CATEGORIES):
        obs.append(estimate(threat_failed[c], threat_seen[c], PRIOR_WEAKNESS[c]))
    for c in range(N_CATEGORIES):
        obs.append(estimate(safe_blocked[c], safe_seen[c], PRIOR_FP[c]))

    if recent:
        mean_ms = sum(entry[3] for entry in recent) / len(recent)
    else:
        mean_ms = PRIOR_RESPONSE_MS
    obs.append(mean_ms / RESPONSE_CAP_MS)

    last = recent[-RECENT:]
    obs.append(sum(1 for entry in last if entry[2]) / len(last) if last else 0.5)

    streak = 0
    for entry in reversed(recent):
        if not entry[2]:
            break
        streak += 1
    obs.append(streak / STREAK_CAP)

    obs.append(stage / MAX_STAGE)
    return [_clip01(value) for value in obs]
