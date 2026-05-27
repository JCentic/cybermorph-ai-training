"""
cybermorph_env.py
────────────────────────────────────────────────────────────────
GYMNASIUM ENVIRONMENT FOR CYBERMORPH RL TRAINING

This file implements the fake CyberMorph player simulation.
The AI (PPO, DQN, A2C) trains against this environment without
the real Godot game running.

WHAT THIS DOES:
  1. Simulates a fake player with customizable weaknesses
  2. Responds to AI actions (attacks) with probabilistic outcomes
  3. Tracks player state (weaknesses, false positives, confidence)
  4. Computes rewards for successful exploitation
  5. Resets episodes with different player profiles

OBSERVATION SPACE (20 floats):
  [0-7]   weakness_vector     — player fails each attack type with this prob
  [8-15]  fp_vector           — player false-positive rate per attack
  [16]    avg_response_time   — normalized 0-1
  [17]    credit_ratio        — current_credits / 500
  [18]    confidence_proxy    — correct_answers / 5
  [19]    consecutive_safe    — safe_events_in_a_row / 4

ACTION SPACE (Discrete 16):
  0=phishing_threat      1=phishing_safe
  2=malvertising_threat  3=malvertising_safe
  4=rogue_soft_threat    5=rogue_soft_safe
  6=driveby_threat       7=driveby_safe
  8=password_threat      9=password_safe
  10=malware_threat      11=malware_safe
  12=mitm_threat         13=mitm_safe
  14=ddos_threat         15=ddos_safe

REWARD STRUCTURE:
  - Positive reward: AI successfully exploits a weakness
  - Bonus: AI uses diverse attacks (not just one type)
  - Penalty: AI repeats same attack too much
────────────────────────────────────────────────────────────────
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import training data — contains real dummy player profiles
# This lets us initialize episodes with realistic player weaknesses
from training_data import DERIVED_WEAKNESS, TRAINING_EPISODE_SEEDS


class CyberMorphEnv(gym.Env):
    """
    ────────────────────────────────────────────────────────────
    CYBERMORPH RL TRAINING ENVIRONMENT
    
    A Gymnasium-compliant environment that simulates fake players
    for reinforcement learning algorithm training.
    
    USAGE:
        env = CyberMorphEnv()
        obs, info = env.reset()
        
        for step in range(300000):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
    
    ────────────────────────────────────────────────────────────
    """
    
    metadata = {"render_modes": []}  # No rendering needed for training
    
    def __init__(self):
        """
        INITIALIZE THE ENVIRONMENT
        
        This runs ONCE when you create the environment.
        It sets up the structure: action space, observation space,
        and loads the training data profiles.
        
        Do NOT put episode-specific initialization here.
        That belongs in reset().
        """
        
        # ─────────────────────────────────────────────────────────
        # DEFINE THE OBSERVATION SPACE
        # ─────────────────────────────────────────────────────────
        # Box(shape=(20,), dtype=float32) means:
        #   - 20 continuous values
        #   - Each value is a 32-bit float
        #   - Values are normalized 0.0 to 1.0
        #
        # This matches our 20-element state array design
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32
        )
        
        # ─────────────────────────────────────────────────────────
        # DEFINE THE ACTION SPACE
        # ─────────────────────────────────────────────────────────
        # Discrete(16) means:
        #   - 16 possible actions (0 through 15)
        #   - Each represents: attack_type (8) x threat_or_safe (2)
        #
        # Action mapping (must match Godot exactly):
        #   [phishing, rogue_soft, pwd_attack, malvertising,
        #    malware, mitm, ddos, driveby] x [threat, safe]
        
        self.action_space = spaces.Discrete(16)
        
        # ─────────────────────────────────────────────────────────
        # LOAD TRAINING DATA PROFILES
        # ─────────────────────────────────────────────────────────
        # These are from training_data.py — real dummy player
        # weakness vectors computed from session logs.
        # This keeps training realistic instead of purely random.
        
        self.player_profiles = DERIVED_WEAKNESS
        self.episode_seeds = TRAINING_EPISODE_SEEDS
        
        # ─────────────────────────────────────────────────────────
        # EPISODE STATE (initialized in reset())
        # ─────────────────────────────────────────────────────────
        
        # Player state — updated during training
        self.current_profile = None  # Which player we're training against
        self.weakness_vector = None  # 8 attack types, failure rates
        self.fp_vector = None        # 8 attack types, false positive rates
        self.avg_response_time = None
        
        # Tracking for the current episode
        self.episode_step = 0
        self.episode_rewards = []
        self.attack_history = []  # Track which attacks were used
        self.credits = 500  # Starting credits
        self.confidence = 0.5  # Starting confidence (0-1)
        self.consecutive_safe_events = 0
        
        print("✓ CyberMorphEnv initialized")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Action space: {self.action_space}")
        print(f"  Loaded {len(self.player_profiles)} player profiles")

    def reset(self, seed=None, options=None):
        """
        RESET THE ENVIRONMENT FOR A NEW EPISODE
        
        This is called at the START OF EVERY EPISODE.
        It picks a new player profile and initializes fresh state.
        
        Gymnasium requires specific return format:
            return observation, info_dict
        
        Args:
            seed: random seed for reproducibility (optional)
            options: additional options (optional)
            
        Returns:
            observation: initial 20-element state array
            info: metadata dict about the episode
        """
        
        # ─────────────────────────────────────────────────────────
        # SET RANDOM SEED (for reproducibility)
        # ─────────────────────────────────────────────────────────
        # If a seed is provided, use it. This ensures that if we
        # run with seed=42, we always get the same sequence of
        # random decisions. This is CRITICAL for algorithm comparison.
        
        super().reset(seed=seed)
        
        # ─────────────────────────────────────────────────────────
        # PICK A RANDOM PLAYER PROFILE
        # ─────────────────────────────────────────────────────────
        # Instead of random weaknesses, pick from our training data.
        # This makes training more realistic — the AI learns to
        # handle different player archetypes (beginners, advanced, etc)
        
        profile_key = random.choice(list(self.player_profiles.keys()))
        profile_data = self.player_profiles[profile_key]
        
        # Extract the base weaknesses from training data
        self.current_profile = profile_key
        base_weakness = np.array(profile_data["weakness"], dtype=np.float32)
        base_fp_rate = np.array(profile_data["fp_rate"], dtype=np.float32)
        base_response_time = profile_data["avg_response_time_ms"]
        
        # ─────────────────────────────────────────────────────────
        # ADD SMALL RANDOM NOISE TO WEAKNESSES
        # ─────────────────────────────────────────────────────────
        # This ensures each episode is slightly different even with
        # the same player profile. Otherwise every Maria episode
        # would be identical.
        #
        # Noise scale: ±0.05 means:
        #   If base weakness is 0.60, episode weakness is 0.55-0.65
        #   This gives variety without destroying the profile's
        #   characteristic weaknesses.
        
        noise_scale = 0.05
        noise = np.random.normal(0, noise_scale, size=8)
        self.weakness_vector = np.clip(base_weakness + noise, 0.0, 1.0).astype(np.float32)
        self.fp_vector = np.clip(base_fp_rate + noise, 0.0, 1.0).astype(np.float32)
        
        # Normalize response time to 0-1 range
        # (base_response_time in milliseconds, normalize by 6000ms max)
        self.avg_response_time = np.clip(base_response_time / 6000.0, 0.0, 1.0)
        
        # ─────────────────────────────────────────────────────────
        # RESET EPISODE TRACKING VARIABLES
        # ─────────────────────────────────────────────────────────
        # These track what happened during THIS episode.
        # They get reset here, then updated by step() each training step.
        
        self.episode_step = 0
        self.episode_rewards = []
        self.attack_history = []  # Track which attacks (0-15) were used
        
        # Reset player state to starting conditions
        self.credits = 500  # Starting credits
        self.confidence = 0.5  # Neutral confidence
        self.consecutive_safe_events = 0
        
        # ─────────────────────────────────────────────────────────
        # BUILD INITIAL OBSERVATION
        # ─────────────────────────────────────────────────────────
        # This calls _get_obs() which builds the 20-element state
        # from current weakness vectors, credits, confidence, etc.
        # We'll implement _get_obs() in the next phase.
        
        obs = self._get_obs()
        
        # ─────────────────────────────────────────────────────────
        # RETURN REQUIRED GYMNASIUM FORMAT
        # ─────────────────────────────────────────────────────────
        # Gymnasium requires: (observation, info_dict)
        # The info dict can contain metadata about the episode.
        
        info = {
            "profile": self.current_profile,
            "episode_step": self.episode_step,
        }
        
        return obs, info


    def _get_obs(self):
        """
        BUILD THE 20-ELEMENT OBSERVATION ARRAY
        
        This helper method constructs the current state that the
        AI "sees" during training. It's called by reset() and step().
        
        The 20-element array format:
        
          [0-7]   weakness_vector     (8 attack types)
          [8-15]  fp_vector           (8 attack types)
          [16]    avg_response_time   (normalized 0-1)
          [17]    credit_ratio        (current / 500)
          [18]    confidence_proxy    (correct / 5 recent events)
          [19]    consecutive_safe    (safe events in a row / 4)
        
        This represents EVERYTHING the AI needs to know about the
        player's current state to make a good attack decision.
        
        Returns:
            obs: np.array of shape (20,) with dtype float32
        """
        
        # ─────────────────────────────────────────────────────────
        # BUILD THE OBSERVATION ARRAY
        # ─────────────────────────────────────────────────────────
        
        obs = np.zeros(20, dtype=np.float32)
        
        # [0-7]: Copy weakness vector directly
        # These are the probabilities the player fails each attack
        obs[0:8] = self.weakness_vector
        
        # [8-15]: Copy false positive rates
        # How often the player blocks SAFE events (false alarms)
        obs[8:16] = self.fp_vector
        
        # [16]: Average response time (already normalized in reset)
        obs[16] = np.float32(self.avg_response_time)
        
        # [17]: Credit ratio normalized to 0-1
        # If player has 500 credits = 1.0
        # If player has 250 credits = 0.5
        # If player has 0 credits = 0.0
        obs[17] = np.float32(min(self.credits / 500.0, 1.0))
        
        # [18]: Confidence proxy — recent correct answers
        # We track this as just the current confidence (0-1)
        # In a full implementation, this would be recent success rate
        obs[18] = np.float32(self.confidence)
        
        # [19]: Consecutive safe events (normalized to 0-1)
        # If player has blocked 4 safe events in a row = 1.0
        # If player has blocked 0 safe events = 0.0
        obs[19] = np.float32(min(self.consecutive_safe_events / 4.0, 1.0))
        
        return obs


    def step(self, action):
        """
        EXECUTE ONE TRAINING STEP
        
        The AI picks an action (attack), and we:
          1. Simulate the player's response
          2. Compute reward for the attack
          3. Update player state (credits, confidence, etc.)
          4. Return new observation + reward + done signal
        
        This is called 300,000 times per training run.
        
        Args:
            action: int in range [0, 15] representing which attack to perform
                    0-7:   8 attack types × threat/safe actions
                    8-15:  continuing the pattern
        
        Returns:
            observation: new 20-element state after action
            reward: float, how well the action performed
            terminated: bool, episode is complete (rare in training)
            truncated: bool, episode hit max steps (we ignore this)
            info: dict with metadata about the step
        """
        
        # ─────────────────────────────────────────────────────────
        # PARSE THE ACTION INTO ATTACK TYPE AND THREAT/SAFE
        # ─────────────────────────────────────────────────────────
        # Action space is 16 discrete values: 8 attacks × 2 variants
        #
        # Action to attack type mapping:
        #   0-1:   phishing (0=threat, 1=safe)
        #   2-3:   malvertising (2=threat, 3=safe)
        #   4-5:   rogue_software (4=threat, 5=safe)
        #   6-7:   driveby (6=threat, 7=safe)
        #   8-9:   password_attack (8=threat, 9=safe)
        #   10-11: malware (10=threat, 11=safe)
        #   12-13: mitm (12=threat, 13=safe)
        #   14-15: ddos (14=threat, 15=safe)
        
        attack_type = action // 2  # Which of 8 attack types (0-7)
        is_threat = action % 2 == 0  # True if threat action, False if safe
        
        # ─────────────────────────────────────────────────────────
        # TRACK ATTACK HISTORY (for diversity scoring)
        # ─────────────────────────────────────────────────────────
        # We record which attacks were used to compute diversity later
        self.attack_history.append(attack_type)
        
        # ─────────────────────────────────────────────────────────
        # SIMULATE PLAYER RESPONSE
        # ─────────────────────────────────────────────────────────
        # Based on the player's weakness, decide if attack succeeds
        #
        # If is_threat=True (real threat):
        #   Success = player FAILS to block it
        #   Probability = weakness[attack_type]
        #   Example: If weakness[phishing]=0.7, 70% chance we succeed
        #
        # If is_threat=False (harmless event):
        #   Success = player DOESN'T falsely block it
        #   Probability = 1 - fp_rate[attack_type]
        #   Example: If fp_rate[phishing]=0.3, 70% chance we succeed
        
        if is_threat:
            # Real threat — succeed if player is weak to this attack
            success_prob = self.weakness_vector[attack_type]
        else:
            # Safe event — succeed if player doesn't false positive
            success_prob = 1.0 - self.fp_vector[attack_type]
        
        # Roll the dice: does this attack succeed?
        attack_succeeds = np.random.random() < success_prob
        
        # ─────────────────────────────────────────────────────────
        # UPDATE PLAYER STATE BASED ON OUTCOME
        # ─────────────────────────────────────────────────────────
        
        if attack_succeeds:
            # Attack was successful
            # - Player loses credits (we "exploited" them)
            # - Their confidence decreases (they're getting stressed)
            # - Consecutive safe events counter resets
            self.credits -= 50  # Loss when attacked
            self.confidence = max(0.0, self.confidence - 0.1)
            self.consecutive_safe_events = 0
        else:
            # Attack failed (player blocked it or didn't fall for it)
            # - Player gains credits (they defended well)
            # - Their confidence increases
            # - Add to consecutive safe events counter
            self.credits += 30  # Gain when they defend
            self.confidence = min(1.0, self.confidence + 0.05)
            self.consecutive_safe_events += 1
        
        # Clamp credits to realistic range (can't go negative or too high)
        self.credits = np.clip(self.credits, 0, 500)
        
        # ─────────────────────────────────────────────────────────
        # COMPUTE REWARD
        # ─────────────────────────────────────────────────────────
        # This is the learning signal. We call _compute_reward()
        # which will be implemented in Phase 5.
        #
        # The reward should:
        #   + Be high when we successfully exploit weaknesses
        #   + Have bonus for using diverse attacks (not just one)
        #   - Have penalty if we keep attacking same type
        # This pushes the AI to learn adaptive strategy, not just spam.
        
        reward = self._compute_reward(attack_succeeds, attack_type)
        
        # ─────────────────────────────────────────────────────────
        # DETERMINE IF EPISODE IS DONE
        # ─────────────────────────────────────────────────────────
        # In our setup, an episode/session ends when:
        #   1. The player runs out of credits (credits <= 0) -> Terminated
        #   2. Or the session reaches its step limit (100 steps) -> Truncated
        
        terminated = self.credits <= 0
        truncated = self.episode_step >= 100
        
        # ─────────────────────────────────────────────────────────
        # INCREMENT STEP COUNTER
        # ─────────────────────────────────────────────────────────
        self.episode_step += 1
        
        # ─────────────────────────────────────────────────────────
        # BUILD NEW OBSERVATION
        # ─────────────────────────────────────────────────────────
        # The AI sees the updated state after this action
        obs = self._get_obs()
        
        # ─────────────────────────────────────────────────────────
        # BUILD INFO DICT FOR DEBUGGING/MONITORING
        # ─────────────────────────────────────────────────────────
        info = {
            "attack_type": attack_type,
            "attack_succeeds": attack_succeeds,
            "is_threat": is_threat,
            "credits": self.credits,
            "confidence": self.confidence,
            "reward": reward,
        }
        
        # ─────────────────────────────────────────────────────────
        # RETURN GYMNASIUM FORMAT
        # ─────────────────────────────────────────────────────────
        # Required format: (obs, reward, terminated, truncated, info)
        
        return obs, reward, terminated, truncated, info


    def _compute_reward(self, attack_succeeds, attack_type):
        """
        COMPUTE THE REWARD FOR THIS STEP
        
        This is THE MOST IMPORTANT METHOD for training quality.
        The reward is the learning signal that teaches the AI.
        
        REWARD PHILOSOPHY:
          ✓ High reward for successful exploitation (core goal)
          ✓ Bonus for diverse attacks (not one-trick spam)
          ✗ Penalty for repetition (forces strategic variety)
        
        This creates a balance:
          - PPO should learn to exploit weaknesses (high reward success)
          - PPO should learn STRATEGY, not find one attack and spam it
          - DQN tends to exploit relentlessly; this penalty helps PPO differ
          - A2C converges fast; this encourages exploration
        
        Args:
            attack_succeeds: bool, did the attack work?
            attack_type: int [0-7], which of 8 attack types was used
        
        Returns:
            reward: float, the learning signal
        """
        
        # ─────────────────────────────────────────────────────────
        # PART 1: BASE REWARD (successful exploitation)
        # ─────────────────────────────────────────────────────────
        # This is the PRIMARY signal: "did you exploit the player?"
        #
        # Success = full reward
        # Partial success = small reward for trying
        # This teaches the AI that successful attacks are good
        
        if attack_succeeds:
            base_reward = 1.0  # Full reward for success
        else:
            base_reward = 0.05  # Small reward for attempt (encourages exploration)
        
        # ─────────────────────────────────────────────────────────
        # PART 2: DIVERSITY BONUS (learn strategy, not one-trick)
        # ─────────────────────────────────────────────────────────
        # Count unique attack types used so far in this episode.
        # More variety = higher bonus.
        #
        # This prevents the AI from finding ONE high-reward attack
        # and spamming it forever. Example:
        #   - Bad AI: finds "password_attack beats Maria" → uses only that
        #   - Good AI: learns all 8 attacks, adapts based on player state
        #
        # How it works:
        #   unique_types = set(attack_history)
        #   diversity = len(unique_types) / 8  (0.0 to 1.0)
        #   bonus = diversity * 0.2  (scales the bonus magnitude)
        
        unique_attack_types = len(set(self.attack_history))
        diversity = unique_attack_types / 8.0  # 0=no diversity, 1=all 8 types used
        diversity_bonus = diversity * 0.2  # Scale bonus magnitude
        
        # ─────────────────────────────────────────────────────────
        # PART 3: REPETITION PENALTY (force strategic variety)
        # ─────────────────────────────────────────────────────────
        # If the AI keeps using the same attack, penalize it.
        #
        # How it works:
        #   Count how many times THIS attack type was used recently
        #   If used too much in a row, reduce reward
        #
        # This prevents:
        #   PPO: from settling into boring local optima
        #   DQN: from pure exploitation of one action
        #   A2C: from converging too narrowly
        #
        # Window = last 20 steps (recent history)
        # If this attack was used 15 times in last 20 steps,
        # that's 75% repetition → strong penalty
        
        recent_window = 20
        if len(self.attack_history) > recent_window:
            recent_history = self.attack_history[-recent_window:]
        else:
            recent_history = self.attack_history
        
        repetition_count = recent_history.count(attack_type)
        repetition_ratio = repetition_count / len(recent_history) if recent_history else 0.0
        
        # Penalty increases as repetition increases
        # At 50% repetition (using same attack 10 out of 20 times): penalty = 0.05
        # At 80% repetition: penalty = 0.16
        repetition_penalty = (repetition_ratio ** 2) * 0.2
        
        # ─────────────────────────────────────────────────────────
        # PART 4: COMBINE SIGNALS INTO FINAL REWARD
        # ─────────────────────────────────────────────────────────
        # Total reward = base + diversity_bonus - repetition_penalty
        #
        # Example scenarios:
        #
        # Scenario A: Successful attack, diverse history, no repetition
        #   base=1.0 + diversity_bonus=0.2 - repetition_penalty=0.01 = 1.19 ✓
        #
        # Scenario B: Failed attack, diverse history, no repetition
        #   base=0.05 + diversity_bonus=0.2 - repetition_penalty=0.01 = 0.24
        #
        # Scenario C: Successful attack, poor diversity, high repetition
        #   base=1.0 + diversity_bonus=0.025 - repetition_penalty=0.16 = 0.865
        #
        # Notice: Scenario C rewards success but encourages the AI to
        # "stop spamming this attack and learn new ones"
        
        total_reward = base_reward + diversity_bonus - repetition_penalty
        
        # Clamp to reasonable range (avoid exploding rewards)
        total_reward = np.clip(total_reward, 0.0, 2.0)
        
        return np.float32(total_reward)


# ════════════════════════════════════════════════════════════════
# ENVIRONMENT COMPLETE
# ════════════════════════════════════════════════════════════════
#
# USAGE EXAMPLE (what the training script does):
#
#   from environment.cybermorph_env import CyberMorphEnv
#   env = CyberMorphEnv()
#
#   obs, info = env.reset(seed=42)  # Start new episode
#   print(f"Observation shape: {obs.shape}")  # (20,)
#
#   for step in range(300000):
#       action = agent.predict(obs)  # AI decides (0-15)
#       obs, reward, done, truncated, info = env.step(action)
#
#       if step % 10000 == 0:
#           print(f"Step {step}: reward={reward:.3f}")
#
#       if done:
#           obs, info = env.reset(seed=seed)
#
# ════════════════════════════════════════════════════════════════
# WHAT THIS ENVIRONMENT DOES
# ════════════════════════════════════════════════════════════════
#
# ✓ Simulates realistic player profiles (from training_data.py)
# ✓ Adds episode variety (noise on weaknesses each reset)
# ✓ Simulates probabilistic player responses
# ✓ Tracks diverse state (weaknesses, confidence, credits)
# ✓ Computes reward balancing exploitation + strategy
# ✓ Follows Gymnasium standard interface
#
# ════════════════════════════════════════════════════════════════
# NEXT STEPS
# ════════════════════════════════════════════════════════════════
#
# Step 1.4: Create profiles/__init__.py (done)
# Step 1.5: Write profiles/player_profiles.py (defines intermediate profile)
# Step 1.6: Write train_ppo.py (trains PPO using this environment)
# Step 1.7: Write train_dqn.py
# Step 1.8: Write train_a2c.py
# Step 1.9: Write run_all.py
# Step 1.10: Write plot_metrics.py
#
# Then Phase 2: Run the comparison!
#
# ════════════════════════════════════════════════════════════════

