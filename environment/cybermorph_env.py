"""
cybermorph_env.py
────────────────────────────────────────────────────────────────
GYMNASIUM ENVIRONMENT FOR CYBERMORPH RL TRAINING

The fake CyberMorph player the AI (PPO, DQN, A2C) trains against,
without the real Godot game running.

WHAT THIS DOES:
  1. Simulates a fake player from one of the profiles in training_data.py.
     The player's TRUE weakness numbers stay hidden inside the env.
  2. Picks a map stage (1 to 4) for each episode: stage k offers the
     first 2k attack categories, like the game's maps.
  3. Starts every episode with the game's judging cards (9 cards, 3 per
     level) so the agent's first estimates already have a few answers.
  4. Shows the agent only ESTIMATES built from the player's answers
     (see observation.py), never the true numbers.
  5. Computes the reward and ends the episode.

OBSERVATION SPACE (20 floats, 0 to 1): defined in observation.py.

ACTION SPACE (Discrete 16):
  attack_type = action // 2   (an index into ATTACK_TYPES)
  is_threat   = action % 2 == 0   (0 = threat card, 1 = safe card)
  See ATTACK_TYPES in training_data.py for the canonical order.
  An action is INVALID when its category is not on the stage's map, or it
  is a safe card for Credential Theft or Public Wi-Fi (the game has none).
  An invalid action costs a step, pays 0 and changes nothing.

REWARD STRUCTURE (unchanged, "weakness-targeting"):
  - Positive reward when the player fails the card
  - Bonus: AI uses diverse attacks (not just one type)
  - Penalty: AI repeats same attack too much
────────────────────────────────────────────────────────────────
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Training data holds the player profiles and the canonical category order
from training_data import ATTACK_TYPES, DERIVED_WEAKNESS
from observation import (
    MAX_STAGE,
    NO_SAFE_CARDS,
    RESPONSE_CAP_MS,
    WINDOW_SIZE,
    action_mask,
    build_observation,
    build_observation_v2,
)

# How many cards the agent chooses in one episode, per map stage. It is the number of cards a real run of that map has, so the agent is trained
# on runs that look like the real ones (before 2026-11-11 every episode was up to 100 cards, far longer than a real run).
# Where the numbers come from: a map's time divided by the time between two cards (the 8 to 10 second cooldown plus the time to answer, about 12 s):
#   Home 90 s -> 8, Office 180 s -> 15, Internet Cafe 270 s -> 22, Public Park 360 s -> 30.
# The testers' real runs agree: at most 7, 14, 21 and 25 cards (analysis of data/, 2026-11-11). The 9 judging cards before a run are not counted here.
RUN_CARDS_BY_STAGE = {1: 8, 2: 15, 3: 22, 4: 30}

# --- THE LEARNING MODEL (an ASSUMPTION, switched on only for the policy comparison; training does not use it) ---
# A real player gets better at what they practise: every card shows the right answer and a short lesson. With learning=True the simulated player does the same:
#   - after a valid card of category c, the player's weakness for c (threat card) or false-positive rate for c (safe card) moves toward a floor by PRACTISE_RATE of the gap;
#   - between cards the player forgets a little: every valid step every number drifts back toward where it was at the start of the episode by FORGET_RATE of the gap.
# These three numbers are NOT measured. They are chosen to be modest (a few cards of one kind help a little, ten cards of one kind help a lot) and are written here and in RESULTS.md as an
# assumption. The comparison of policies must be shown for more than one setting (see evaluate_policies.py --learning-rates) so the conclusion does not depend on one guess.
PRACTISE_RATE = 0.06
FORGET_RATE = 0.01
WEAKNESS_FLOOR = 0.05
FP_FLOOR = 0.02

# --- THE TWO REWARDS ---
# "weakness"    (Model 1, the first reward) pays the agent when the simulated player FAILS the card. It finds weak spots, but a coach rewarded for failures keeps
#               hitting the same weak spot and the player does not improve (RESULTS.md, A6).
# "improvement" (Model 2) pays the agent when the player's average weakness FALLS after the card (needs learning=True), plus the same variety bonus and repetition penalty.
#               This is what a coach is for: the player gets better. IMPROVEMENT_SCALE turns a tiny drop (about 0.002 to 0.01 per card) into a reward of a size like the old one.
REWARD_MODES = ("weakness", "improvement")
IMPROVEMENT_SCALE = 50.0
START_CREDITS = 500      # inside the simulation only; the agent never sees it
JUDGING_LEVELS = 3       # the game's judging session: 3 levels ...
JUDGING_CARDS_PER_LEVEL = 3   # ... of 3 cards each (JudgingSettings), at least 1 safe per level
RESPONSE_SPREAD = 0.2    # a player's answer time varies by about 20 percent


class CyberMorphEnv(gym.Env):
    """
    USAGE:
        env = CyberMorphEnv()
        obs, info = env.reset(seed=42)
        obs, info = env.reset(options={"stage": 4, "profile": "dummy-player-003"})

        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}  # No rendering needed for training

    def __init__(self, profiles=None, learning=False, practise_rate=None, forget_rate=None, reward_mode="weakness", observation_version=1):
        # observation_version 1: the 20 numbers of observation.py. Version 2: 29 numbers (adds the practice counts and the share of the run left) so the agent can plan.
        assert observation_version in (1, 2), observation_version
        self.observation_version = observation_version
        assert reward_mode in REWARD_MODES, reward_mode
        assert reward_mode == "weakness" or learning, "the improvement reward needs learning=True (a player who never improves can give no improvement reward)"
        self.reward_mode = reward_mode
        # `learning`: True makes the simulated player improve with practice (see the learning model above). Default False: the training world, unchanged.
        self.learning = learning
        self.practise_rate = PRACTISE_RATE if practise_rate is None else practise_rate
        self.forget_rate = FORGET_RATE if forget_rate is None else forget_rate
        # `profiles`: the simulated players of this environment (a dict like DERIVED_WEAKNESS). None means the five invented ones (training_data.py). The evaluation can be given
        # other players that the agent never trained on (see training_common.load_held_out_profiles).
        # Box(shape=(20,)): 20 values in 0 to 1 (see observation.py)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(20 if observation_version == 1 else 29,), dtype=np.float32)

        # Discrete(16): attack_type (8) x threat_or_safe (2)
        self.action_space = spaces.Discrete(16)

        self.player_profiles = profiles if profiles is not None else DERIVED_WEAKNESS

        # Hidden truth of the simulated player (set in reset)
        self.current_profile = None
        self.weakness_vector = None  # 8 failure rates, never shown to the agent
        self.fp_vector = None        # 8 false-positive rates, never shown
        self.avg_response_ms = None

        # What the agent can learn from: the player's recent answers
        self.window = []
        self.stage = 1

        # Tracking for the current episode
        self.episode_step = 0
        self.attack_history = []  # category of every VALID action
        self.credits = START_CREDITS

        print("[OK] CyberMorphEnv initialized")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Action space: {self.action_space}")
        print(f"  Loaded {len(self.player_profiles)} player profiles")

    def reset(self, seed=None, options=None):
        """
        START A NEW EPISODE

        options (both optional, used for evaluation and the graphs):
            "stage":   1 to 4, the map stage (default: drawn, equal weights)
            "profile": a key of DERIVED_WEAKNESS (default: drawn)
        """
        super().reset(seed=seed)
        options = options or {}

        # Every draw below comes from self.np_random, so a seed repeats the episode.
        profile_keys = list(self.player_profiles.keys())
        if "profile" in options:
            profile_key = options["profile"]
            assert profile_key in self.player_profiles, f"unknown profile {profile_key}"
        else:
            profile_key = profile_keys[int(self.np_random.integers(len(profile_keys)))]

        if "stage" in options:
            self.stage = int(options["stage"])
            assert 1 <= self.stage <= MAX_STAGE, f"stage must be 1 to {MAX_STAGE}"
        else:
            self.stage = int(self.np_random.integers(1, MAX_STAGE + 1))

        profile_data = self.player_profiles[profile_key]
        self.current_profile = profile_key
        base_weakness = np.array(profile_data["weakness"], dtype=np.float32)
        base_fp_rate = np.array(profile_data["fp_rate"], dtype=np.float32)
        self.avg_response_ms = float(profile_data["avg_response_time_ms"])

        # Small noise so two episodes of the same profile differ. Two separate
        # draws: weakness and fp_rate must not move together.
        noise_scale = 0.05
        weakness_noise = self.np_random.normal(0, noise_scale, size=8)
        fp_noise = self.np_random.normal(0, noise_scale, size=8)
        self.weakness_vector = np.clip(base_weakness + weakness_noise, 0.0, 1.0).astype(np.float32)
        self.fp_vector = np.clip(base_fp_rate + fp_noise, 0.0, 1.0).astype(np.float32)

        self.episode_step = 0
        self.attack_history = []
        self.credits = START_CREDITS
        self.window = []
        # Where the player started (the forgetting pulls back toward it, and the skill gain is measured from it).
        self.start_weakness = self.weakness_vector.copy()
        self.start_fp = self.fp_vector.copy()

        # The game opens every run with a judging session; the agent's first
        # estimates are built from it.
        self._play_judging_cards()

        info = {
            "profile": self.current_profile,
            "stage": self.stage,
            "episode_step": self.episode_step,
            "start_mean_weakness": self.mean_weakness(self.start_weakness),
        }
        return self._get_obs(), info

    def _play_judging_cards(self):
        """Simulate the 9 judging cards (JudgingPattern): per level, the stage's
        categories take turns in a random order, with at least one safe card."""
        offered = 2 * self.stage
        for _level in range(JUDGING_LEVELS):
            order = [int(c) for c in self.np_random.permutation(offered)]
            cards = [order[i % offered] for i in range(JUDGING_CARDS_PER_LEVEL)]
            is_threat = [True] * JUDGING_CARDS_PER_LEVEL
            can_be_safe = [i for i, c in enumerate(cards) if c not in NO_SAFE_CARDS]
            if can_be_safe:
                is_threat[can_be_safe[int(self.np_random.integers(len(can_be_safe)))]] = False
            for category, threat in zip(cards, is_threat):
                self._answer(category, threat)

    def _answer(self, attack_type, is_threat):
        """The simulated player answers one card. Records it in the window and
        returns True when the player FAILED it (the 'attack succeeds')."""
        # Threat card: the player fails with probability weakness.
        # Safe card: the player fails by wrongly blocking it, probability fp_rate.
        fail_prob = self.weakness_vector[attack_type] if is_threat else self.fp_vector[attack_type]
        attack_succeeds = bool(self.np_random.random() < fail_prob)

        spread = self.avg_response_ms * RESPONSE_SPREAD
        response_ms = float(np.clip(self.np_random.normal(self.avg_response_ms, spread), 300.0, RESPONSE_CAP_MS))

        self.window.append((attack_type, is_threat, not attack_succeeds, response_ms))
        del self.window[:-WINDOW_SIZE]
        return attack_succeeds

    def mean_weakness(self, vector):
        """The average weakness over the categories this map offers (the first 2 x stage). Lower is a better player."""
        offered = 2 * self.stage
        return float(np.mean(vector[:offered]))

    def _learn(self, attack_type, is_threat):
        """The learning model (see the constants at the top): practise what was just answered, forget a little of everything."""
        if is_threat:
            gap = self.weakness_vector[attack_type] - WEAKNESS_FLOOR
            self.weakness_vector[attack_type] -= self.practise_rate * max(gap, 0.0)
        else:
            gap = self.fp_vector[attack_type] - FP_FLOOR
            self.fp_vector[attack_type] -= self.practise_rate * max(gap, 0.0)
        self.weakness_vector += self.forget_rate * (self.start_weakness - self.weakness_vector)
        self.fp_vector += self.forget_rate * (self.start_fp - self.fp_vector)
        np.clip(self.weakness_vector, 0.0, 1.0, out=self.weakness_vector)
        np.clip(self.fp_vector, 0.0, 1.0, out=self.fp_vector)

    def _get_obs(self):
        """The 20 numbers the agent sees: estimates from the window, not the truth."""
        if self.observation_version == 2:
            left = 1.0 - self.episode_step / RUN_CARDS_BY_STAGE[self.stage]
            return np.array(build_observation_v2(self.window, self.stage, left), dtype=np.float32)
        return np.array(build_observation(self.window, self.stage), dtype=np.float32)

    def step(self, action):
        """
        ONE TRAINING STEP

        Returns: observation, reward, terminated, truncated, info
        """
        action = int(action)
        attack_type = action // 2
        is_threat = action % 2 == 0

        self.episode_step += 1
        truncated = self.episode_step >= RUN_CARDS_BY_STAGE[self.stage]

        # Invalid action: costs the step, pays nothing, changes nothing.
        if not action_mask(self.stage)[action]:
            info = {
                "attack_type": attack_type,
                "attack_name": ATTACK_TYPES[attack_type],
                "invalid": True,
                "stage": self.stage,
                "reward": 0.0,
            }
            return self._get_obs(), np.float32(0.0), False, truncated, info

        # The diversity score counts valid actions only
        self.attack_history.append(attack_type)

        attack_succeeds = self._answer(attack_type, is_threat)
        weakness_before = self.mean_weakness(self.weakness_vector)
        if self.learning:
            self._learn(attack_type, is_threat)
        self._improvement = weakness_before - self.mean_weakness(self.weakness_vector)

        # Player loses credits when attacked, gains them when defending.
        # Credits only decide when an episode ends; the agent never sees them.
        self.credits += -50 if attack_succeeds else 30
        self.credits = int(np.clip(self.credits, 0, START_CREDITS))

        reward = self._compute_reward(attack_succeeds, attack_type)
        terminated = self.credits <= 0

        info = {
            "attack_type": attack_type,
            "attack_name": ATTACK_TYPES[attack_type],
            "attack_succeeds": attack_succeeds,
            "is_threat": is_threat,
            "invalid": False,
            "stage": self.stage,
            "credits": self.credits,
            "reward": reward,
            "mean_weakness": self.mean_weakness(self.weakness_vector),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, attack_succeeds, attack_type):
        """
        THE REWARD FOR THIS STEP ("weakness-targeting", decision D33-4)

          + 1.0 when the player fails the card, 0.05 when not
          + diversity bonus: unique categories used so far / 8 * 0.2
          - repetition penalty: (share of the last 20 valid steps in this
            category)^2 * 0.2
        Clipped to 0 to 2.

        Known quirk, kept on purpose: with a one-step history the repetition
        share is 1.0, so the first step of an episode pays 0.
        """
        if self.reward_mode == "improvement":
            base_reward = max(self._improvement, 0.0) * IMPROVEMENT_SCALE
        else:
            base_reward = 1.0 if attack_succeeds else 0.05

        diversity = len(set(self.attack_history)) / 8.0
        diversity_bonus = diversity * 0.2

        recent_window = 20
        recent_history = self.attack_history[-recent_window:]
        repetition_count = recent_history.count(attack_type)
        repetition_ratio = repetition_count / len(recent_history) if recent_history else 0.0
        repetition_penalty = (repetition_ratio ** 2) * 0.2

        total_reward = base_reward + diversity_bonus - repetition_penalty
        return np.float32(np.clip(total_reward, 0.0, 2.0))
