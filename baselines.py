"""
baselines.py
────────────────────────────────────────────────────────────────
THE NON-RL WAYS OF CHOOSING THE NEXT CARD (the rivals of the RL agent)

The research question is "why do we use reinforcement learning for the game?".
To answer it, the RL agent must beat, or at least be honestly compared with,
simple ways of choosing cards that need no learning. Each policy here does one
job: given what the game can see (the 20 numbers of observation.py and the
map stage), return one of the 16 actions (category x threat-or-safe).

Every policy only chooses VALID actions (a category the map offers, and no safe
card for Credential Theft and Public Wi-Fi), so it is compared fairly with
the RL agent, which has to learn the same rules.

THE POLICIES
  RandomPolicy         a random valid card. The "no idea" floor.
  StandardPattern      the same fixed order for everyone: the categories take turns,
                       two threat cards and then one safe card where a safe card
                       exists. This is the game WITHOUT the AI (the standard pattern).
  GreedyWeakest        always the card the player seems weakest at (highest estimate).
                       The simplest "adaptive" rule. Never changes its mind while the
                       estimate stays the highest, so it can repeat one category.
  GreedyWithVariety    like GreedyWeakest, but never the same category three times in a
                       row. The strong hand-written rival: if RL cannot beat this one,
                       we must say so.
  RLPolicy             a trained Stable-Baselines3 model.

The estimates are the observation numbers 0 to 7 (threat) and 8 to 15
(safe-card blocking), exactly what the RL agent sees. Standard library + numpy.
────────────────────────────────────────────────────────────────
"""

import numpy as np

from observation import N_CATEGORIES, NO_SAFE_CARDS, action_mask


def action_of(category, is_threat):
    """The action number for a category and a card kind (the env's rule: category x 2, threat first)."""
    return category * 2 + (0 if is_threat else 1)


class Policy:
    """Base class: a name, a reset for a new episode, and act()."""
    name = "policy"

    def reset(self, rng):
        """Called at the start of every episode. `rng` is a numpy Generator for any random choice."""
        self.rng = rng

    def act(self, obs, stage):
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def act(self, obs, stage):
        valid = np.flatnonzero(action_mask(stage))
        return int(valid[int(self.rng.integers(len(valid)))])


class StandardPattern(Policy):
    name = "standard pattern"

    def reset(self, rng):
        super().reset(rng)
        self.step = 0

    def act(self, obs, stage):
        offered = 2 * stage
        category = self.step % offered
        # Two threat cards for every safe card, where a safe card exists for this category.
        round_number = self.step // offered
        is_threat = not (round_number % 3 == 2 and category not in NO_SAFE_CARDS)
        self.step += 1
        return action_of(category, is_threat)


class GreedyWeakest(Policy):
    name = "greedy weakest"

    def reset(self, rng):
        super().reset(rng)
        self.history = []

    def _scores(self, obs, stage):
        """16 numbers: the estimate for each action (weakness for a threat card, safe-card blocking for a safe card). Invalid actions get -1."""
        scores = np.full(16, -1.0)
        mask = action_mask(stage)
        for category in range(N_CATEGORIES):
            if mask[action_of(category, True)]:
                scores[action_of(category, True)] = float(obs[category])
            if mask[action_of(category, False)]:
                scores[action_of(category, False)] = float(obs[8 + category])
        return scores

    def act(self, obs, stage):
        scores = self._scores(obs, stage)
        best = np.flatnonzero(scores == scores.max())
        action = int(best[int(self.rng.integers(len(best)))])
        self.history.append(action // 2)
        return action


class GreedyWithVariety(GreedyWeakest):
    name = "greedy with variety"

    def act(self, obs, stage):
        scores = self._scores(obs, stage)
        if len(self.history) >= 2 and self.history[-1] == self.history[-2]:
            blocked = self.history[-1]
            scores[action_of(blocked, True)] = -1.0
            scores[action_of(blocked, False)] = -1.0
        best = np.flatnonzero(scores == scores.max())
        action = int(best[int(self.rng.integers(len(best)))])
        self.history.append(action // 2)
        return action


class RLPolicy(Policy):
    """A trained Stable-Baselines3 model (PPO, DQN or A2C). It chooses the best action it knows, with no random exploring."""

    def __init__(self, model, label):
        self.model = model
        self.name = label

    def act(self, obs, stage):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def all_baselines():
    """The four rivals that need no learning, in the order they are reported."""
    return [RandomPolicy(), StandardPattern(), GreedyWeakest(), GreedyWithVariety()]
