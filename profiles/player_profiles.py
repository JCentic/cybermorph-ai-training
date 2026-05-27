"""
player_profiles.py
────────────────────────────────────────────────────────────────
PLAYER PROFILE DEFINITIONS FOR RL TRAINING

This file defines 4 fake player archetypes that the AI trains against.
These are NOT real players — they're simulated profiles with
different weakness patterns so the AI learns robust strategy.

PROFILES DEFINED:
  1. beginner       — Fails often, blocks often (defensive)
  2. intermediate   — Balanced, middle-of-road difficulty
  3. advanced       — Strong all-around, one specific weakness
  4. phishing_spec  — Expert except vulnerable to phishing

ATTACK TYPE ORDER (must match cybermorph_env.py exactly):
  Index 0: phishing
  Index 1: rogue_software
  Index 2: password_attack
  Index 3: malvertising
  Index 4: malware
  Index 5: mitm
  Index 6: ddos
  Index 7: driveby

All values are normalized 0.0-1.0:
  0.0 = never happens
  0.5 = 50% chance
  1.0 = always happens

────────────────────────────────────────────────────────────────
"""

# ────────────────────────────────────────────────────────────────
# PLAYER PROFILES DICTIONARY
# ────────────────────────────────────────────────────────────────
# This is imported by cybermorph_env.py to initialize episodes.
# When env.reset() is called, it picks one of these profiles
# and adds small noise to create episode variety.
# ────────────────────────────────────────────────────────────────

PLAYER_PROFILES = {

    # ──────────────────────────────────────────────────────────
    # BEGINNER PROFILE
    # ──────────────────────────────────────────────────────────
    # Characteristic: Fails often, defensive
    #
    # This player:
    # - Is vulnerable to most attacks (high weakness)
    # - Over-blocks often (high false positive rate)
    # - Reacts slowly (defensive, cautious)
    # - Low starting confidence (just learning the game)
    #
    # Why train against this:
    # - AI learns easy exploitation patterns
    # - AI learns that some attacks are obviously effective
    # - Good baseline for learning phase
    
    "beginner": {
        "description": "Beginner player — vulnerable to most attacks",
        
        # Weakness vector: How often does player fail each attack?
        # High values = high vulnerability = AI can exploit easily
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness": [0.70, 0.65, 0.60, 0.70, 0.55, 0.50, 0.45, 0.50],
        
        # False positive vector: How often does player block SAFE events?
        # High values = player over-blocks = AI learns to be cautious
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "fp_rate": [0.35, 0.25, 0.30, 0.30, 0.20, 0.15, 0.10, 0.20],
        
        # Starting confidence: How confident is player at episode start?
        # 0.0 = not confident, 1.0 = very confident
        # Beginners start low (just learning)
        "confidence_start": 0.30,
    },

    # ──────────────────────────────────────────────────────────
    # INTERMEDIATE PROFILE
    # ──────────────────────────────────────────────────────────
    # Characteristic: Balanced, middle-of-road difficulty
    #
    # This is the MAIN profile for algorithm comparison.
    # According to CONTEXT.md:
    #   "All three algorithms trained under IDENTICAL conditions...
    #    Player profile: intermediate"
    #
    # This player:
    # - Moderate vulnerability (some attacks work, some don't)
    # - Moderate false positive rate (not paranoid, not oblivious)
    # - Normal reaction speed
    # - Medium starting confidence
    #
    # Why this profile:
    # - Stresses all algorithms enough to see differences
    # - Not so easy that all algorithms perform well
    # - Not so hard that training takes forever
    # - Represents typical engaged player
    
    "intermediate": {
        "description": "Intermediate player — balanced difficulty (MAIN PROFILE)",
        
        # Moderate weakness — some attacks work, others don't
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness": [0.40, 0.50, 0.55, 0.45, 0.50, 0.55, 0.40, 0.45],
        
        # Moderate false positives — player is reasonably cautious
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "fp_rate": [0.15, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.12],
        
        # Medium confidence — engaged but not overconfident
        "confidence_start": 0.50,
    },

    # ──────────────────────────────────────────────────────────
    # ADVANCED PROFILE
    # ──────────────────────────────────────────────────────────
    # Characteristic: Strong all-around, one critical weakness
    #
    # This player:
    # - Rarely falls for attacks (low weakness across board)
    # - Rarely false-positives (selective blocker, not paranoid)
    # - Fast reaction time (experienced)
    # - HIGH WEAKNESS on MITM specifically
    #
    # Why this profile:
    # - Tests if AI can identify and exploit specific weaknesses
    # - Tests if AI can learn when NOT to use certain attacks
    # - Simulates experienced player with known blind spot
    #
    # Real-world example:
    # - Expert cybersecurity professional who never uses VPNs
    # - Vulnerable to man-in-the-middle attacks despite expertise
    
    "advanced": {
        "description": "Advanced player — strong but weak to MITM",
        
        # Low weakness overall, but very high on MITM
        # Notice: mitm (index 5) is much higher than others
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness": [0.15, 0.20, 0.15, 0.15, 0.20, 0.85, 0.25, 0.20],
        
        # Low false positives — this player doesn't over-block
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "fp_rate": [0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05],
        
        # High confidence — experienced player
        "confidence_start": 0.80,
    },

    # ──────────────────────────────────────────────────────────
    # PHISHING SPECIALIST PROFILE
    # ──────────────────────────────────────────────────────────
    # Characteristic: Trained against phishing, vulnerable elsewhere
    #
    # This player:
    # - VERY resistant to phishing (specialized training)
    # - Vulnerable to other attacks (narrow expertise)
    # - Over-blocks sometimes (cautious about their weakness area)
    # - Medium confidence
    #
    # Why this profile:
    # - Tests if AI can adapt to player specialization
    # - Tests if AI learns to avoid ineffective attacks
    # - Simulates user who completed phishing training program
    #   but neglected other security awareness
    #
    # Real-world example:
    # - User completed security awareness training on phishing
    # - Never falls for phishing emails anymore
    # - But still vulnerable to malware, social engineering, etc.
    
    "phishing_specialist": {
        "description": "Specialist — trained on phishing, weak elsewhere",
        
        # Very low phishing weakness (resistant to phishing attacks)
        # But vulnerable to everything else
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness": [0.10, 0.60, 0.65, 0.55, 0.60, 0.65, 0.50, 0.55],
        
        # Medium false positives — some caution but not excessive
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "fp_rate": [0.20, 0.15, 0.18, 0.15, 0.12, 0.10, 0.08, 0.12],
        
        # Medium-high confidence — trained, but not overconfident
        "confidence_start": 0.60,
    },
}


# ────────────────────────────────────────────────────────────────
# USAGE IN CYBERMORPH_ENV
# ────────────────────────────────────────────────────────────────
#
# When training starts, cybermorph_env.py does:
#
#   from profiles.player_profiles import PLAYER_PROFILES
#
#   def reset(self):
#       profile = random.choice(list(PLAYER_PROFILES.values()))
#       weakness = np.array(profile["weakness"])
#       fp_rate = np.array(profile["fp_rate"])
#       confidence = profile["confidence_start"]
#
# This loads a random player each episode and adds small noise
# to create variety while keeping characteristic weaknesses.
#
# ────────────────────────────────────────────────────────────────
# ATTACK TYPE REFERENCE
# ────────────────────────────────────────────────────────────────
# Index 0: phishing
#   Real: Fake emails trying to steal credentials
#   Safe: Legitimate email from trusted sender
#   Weakness: How often player fails to identify phishing
#
# Index 1: rogue_software
#   Real: Malicious software disguised as legitimate app
#   Safe: Real software from trusted source
#   Weakness: How often player installs rogue apps
#
# Index 2: password_attack
#   Real: Brute force / credential stuffing attempts
#   Safe: Legitimate password change prompt
#   Weakness: How often player uses weak passwords
#
# Index 3: malvertising
#   Real: Malicious ads on websites
#   Safe: Legitimate ads from trusted sites
#   Weakness: How often player clicks malicious ads
#
# Index 4: malware
#   Real: Malicious code execution attempts
#   Safe: Legitimate software update
#   Weakness: How often player runs malware
#
# Index 5: mitm (Man-in-the-Middle)
#   Real: Network eavesdropping / intercept attempts
#   Safe: Legitimate encrypted connection
#   Weakness: How often player uses unsecured networks
#
# Index 6: ddos (Distributed Denial of Service)
#   Real: Overwhelming server with requests
#   Safe: Legitimate high-traffic event
#   Weakness: How often player falls for DDoS disguises
#
# Index 7: driveby
#   Real: Drive-by download attacks from visiting sites
#   Safe: Legitimate file download
#   Weakness: How often player gets compromised by driveby
#
# ────────────────────────────────────────────────────────────────
