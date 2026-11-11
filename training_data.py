"""
training_data.py
────────────────────────────────────────────────────────────────
TRAINING DUMMY DATA — FOR RL TRAINING ONLY

This file is used by cybermorph_env.py during AI training.
It does NOT connect to Godot or the game directly.

PURPOSE:
  Provides dummy player session records that simulate
  real gameplay data. The RL environment uses these to
  initialize player weakness profiles during training,
  making the AI learn from realistic starting conditions
  rather than purely random ones.

WHAT THIS FILE CONTAINS:
  1. DUMMY_SESSIONS     — fake completed game sessions
                          (mirrors game_sessions DB table)
  2. DUMMY_PLAYERS      — fake player profiles
                          (mirrors player_profiles DB table)
  3. DUMMY_ATTACK_LOGS  — fake per-attack interaction records
                          (mirrors threat_attack_log SQLite table)
  4. DERIVED_WEAKNESS   — the 5 hand-made weakness profiles the
                          environment trains against (placeholders
                          until real players' data replaces them)

ATTACK TYPE ORDER: see ATTACK_TYPES below. It is the only place the
order is written; it must equal Config.THREAT_CATEGORIES in the game.
────────────────────────────────────────────────────────────────
"""

# The 8 attack categories, in the game's order (Config.THREAT_CATEGORIES in
# the Godot game, which is DICT's Annex A order). Index i here is index i in
# every weakness list, in the observation, and attack_type = action // 2.
# Map k of the game offers the first 2k of them (Home 2, Office 4, Cafe 6,
# Park 8). Change this list and the game's list in the same sitting.
ATTACK_TYPES = [
    "Phishing",
    "Smishing",
    "Vishing",
    "Social Engineering",
    "Credential Theft",
    "Public Wi-Fi Attack",
    "Malware Infection",
    "Ransomware",
]

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — DUMMY PLAYER PROFILES
# Mirrors player_profiles table in PostgreSQL
# security_credits starts at 100 per data dictionary
# map_progress: 0=Home unlocked, 1=Office, 2=Internet Cafe, 3=Public Park
# ─────────────────────────────────────────────────────────────────

DUMMY_PLAYERS = [
    {
        "profile_id": "dummy-player-001",
        "username": "JuanDelaCruz",
        "map_progress": 0,
        "security_credits": 100,
        "total_sessions": 3,
        "notes": "Typical beginner. Fails Phishing and Smishing often."
    },
    {
        "profile_id": "dummy-player-002",
        "username": "MariaClara",
        "map_progress": 1,
        "security_credits": 340,
        "total_sessions": 8,
        "notes": "Intermediate. Good at Phishing, weak on Credential Theft."
    },
    {
        "profile_id": "dummy-player-003",
        "username": "KardingGamer",
        "map_progress": 2,
        "security_credits": 580,
        "total_sessions": 15,
        "notes": "Advanced gamer. Very weak on Public Wi-Fi — joins any open network."
    },
    {
        "profile_id": "dummy-player-004",
        "username": "TeacherRosa",
        "map_progress": 1,
        "security_credits": 290,
        "total_sessions": 6,
        "notes": "Educator playing to understand the game. "
                 "Suspicious of everything — high false positive rate."
    },
    {
        "profile_id": "dummy-player-005",
        "username": "BenjiPH",
        "map_progress": 3,
        "security_credits": 820,
        "total_sessions": 22,
        "notes": "Near-expert. Only real weakness is spotting Ransomware."
    },
]


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — DUMMY GAME SESSIONS
# Mirrors game_sessions table in PostgreSQL
# Each row = one completed map session
# result: "win" / "lose" / "timeout"
# ─────────────────────────────────────────────────────────────────

DUMMY_SESSIONS = [

    # ── JuanDelaCruz — Home Map sessions (beginner) ──────────────
    {
        "session_id": "sess-001",
        "profile_id": "dummy-player-001",
        "map_name": "Home",
        "duration_seconds": 420,
        "credits_earned": 80,
        "credits_lost": 120,
        "false_positives": 3,
        "result": "lose",
    },
    {
        "session_id": "sess-002",
        "profile_id": "dummy-player-001",
        "map_name": "Home",
        "duration_seconds": 390,
        "credits_earned": 110,
        "credits_lost": 90,
        "false_positives": 2,
        "result": "lose",
    },
    {
        "session_id": "sess-003",
        "profile_id": "dummy-player-001",
        "map_name": "Home",
        "duration_seconds": 350,
        "credits_earned": 150,
        "credits_lost": 60,
        "false_positives": 1,
        "result": "win",
    },

    # ── MariaClara — Home + Internet Cafe sessions ────────────────
    {
        "session_id": "sess-004",
        "profile_id": "dummy-player-002",
        "map_name": "Home",
        "duration_seconds": 310,
        "credits_earned": 200,
        "credits_lost": 40,
        "false_positives": 1,
        "result": "win",
    },
    {
        "session_id": "sess-005",
        "profile_id": "dummy-player-002",
        "map_name": "Internet Cafe",
        "duration_seconds": 480,
        "credits_earned": 160,
        "credits_lost": 100,
        "false_positives": 2,
        "result": "lose",
    },
    {
        "session_id": "sess-006",
        "profile_id": "dummy-player-002",
        "map_name": "Internet Cafe",
        "duration_seconds": 400,
        "credits_earned": 190,
        "credits_lost": 70,
        "false_positives": 1,
        "result": "win",
    },

    # ── KardingGamer — all maps, strong player ────────────────────
    {
        "session_id": "sess-007",
        "profile_id": "dummy-player-003",
        "map_name": "Home",
        "duration_seconds": 280,
        "credits_earned": 240,
        "credits_lost": 20,
        "false_positives": 0,
        "result": "win",
    },
    {
        "session_id": "sess-008",
        "profile_id": "dummy-player-003",
        "map_name": "Internet Cafe",
        "duration_seconds": 320,
        "credits_earned": 210,
        "credits_lost": 30,
        "false_positives": 0,
        "result": "win",
    },
    {
        "session_id": "sess-009",
        "profile_id": "dummy-player-003",
        "map_name": "Company Office",
        "duration_seconds": 550,
        "credits_earned": 140,
        "credits_lost": 130,
        "false_positives": 2,
        "result": "lose",
        "notes": "Failed Public Wi-Fi scenario — joined a fake hotspot"
    },

    # ── TeacherRosa — cautious over-blocker ───────────────────────
    {
        "session_id": "sess-010",
        "profile_id": "dummy-player-004",
        "map_name": "Home",
        "duration_seconds": 460,
        "credits_earned": 120,
        "credits_lost": 110,
        "false_positives": 5,
        "result": "lose",
        "notes": "Blocked legitimate GCash transaction — false positive"
    },
    {
        "session_id": "sess-011",
        "profile_id": "dummy-player-004",
        "map_name": "Internet Cafe",
        "duration_seconds": 510,
        "credits_earned": 150,
        "credits_lost": 95,
        "false_positives": 4,
        "result": "lose",
    },

    # ── BenjiPH — near-expert, fails only on Ransomware ─────────────────
    {
        "session_id": "sess-012",
        "profile_id": "dummy-player-005",
        "map_name": "Public Park",
        "duration_seconds": 290,
        "credits_earned": 260,
        "credits_lost": 25,
        "false_positives": 0,
        "result": "win",
    },
    {
        "session_id": "sess-013",
        "profile_id": "dummy-player-005",
        "map_name": "Public Park",
        "duration_seconds": 380,
        "credits_earned": 200,
        "credits_lost": 80,
        "false_positives": 1,
        "result": "win",
        "notes": "Fell for a fake ransom note"
    },
]


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — DUMMY ATTACK INTERACTION LOGS
# Mirrors threat_attack_log SQLite table (local-only, never synced)
# is_correct: 1 = player made right decision, 0 = wrong decision
# player_action: "block" or "accept"
# correct_action: what they SHOULD have done
# ─────────────────────────────────────────────────────────────────

DUMMY_ATTACK_LOGS = [

    # ── JuanDelaCruz interactions (beginner, fails Phishing a lot) ─
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "Phishing",        "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3200},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "Smishing",    "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 2800},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "Phishing",        "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4100},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "Smishing",    "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1900},
    # False positive — blocked a safe phishing-style email
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "Phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 5500},

    # ── MariaClara interactions (good at Phishing, weak on Credential Theft) ─
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "Phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1200},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "Credential Theft", "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3800},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "Social Engineering",  "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 2900},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "Phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1500},
    # False positive on a safe Social Engineering prompt
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "Social Engineering",  "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 2100},

    # ── KardingGamer interactions (strong except Public Wi-Fi) ─────────────
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "Phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 800},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "Malware Infection",         "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 950},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "Public Wi-Fi Attack",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4200},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "Public Wi-Fi Attack",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3800},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "Credential Theft", "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1100},

    # ── TeacherRosa interactions (over-blocker, high false positives)
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "Phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 2200},
    # False positives — blocks everything including safe events
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "Phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 6100},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "Smishing",    "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 5800},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "Social Engineering",  "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 4900},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "Phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 7200},

    # ── BenjiPH interactions (expert, only fails Ransomware) ─────────────
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "Phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 600},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "Public Wi-Fi Attack",            "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 750},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "Ransomware",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 5100},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "Malware Infection",         "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 820},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "Ransomware",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4800},
]


# ─────────────────────────────────────────────────────────────────
# SECTION 4 — DERIVED WEAKNESS VECTORS
# HAND-MADE PLACEHOLDERS, not computed from the logs above and not
# fitted to real players (phase 34 replaces them if real data exists).
# These are the actual numbers fed into the RL environment
# as starting weakness profiles for training.
#
# How to read:
#   weakness[i] = failure rate for attack type i
#   0.0 = never fails, 1.0 = always fails
#
# Index i follows ATTACK_TYPES (top of this file), the game's order:
#   [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
# The early categories (0-3) are the "early tier", the late ones the
# "late tier": beginners are weak early, experts only fail late.
#
# fp_rate[i] = false positive rate for attack type i
#   How often player blocks SAFE events of that type
#   Credential Theft (4) and Public Wi-Fi (5) have NO safe cards in the
#   game, so their fp_rate cannot be measured: it is the same fixed 0.05
#   for every profile, and phase 33 makes the safe action invalid there.
# ─────────────────────────────────────────────────────────────────

DERIVED_WEAKNESS = {

    "dummy-player-001": {
        "username": "JuanDelaCruz",
        "description": "Beginner — broad weakness, worst on the early types (Phishing, Smishing)",
        # [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
        "weakness":  [0.75, 0.65, 0.55, 0.60, 0.50, 0.45, 0.40, 0.35],
        "fp_rate":   [0.35, 0.25, 0.20, 0.20, 0.05, 0.05, 0.10, 0.05],
        "avg_response_time_ms": 3500,
    },

    "dummy-player-002": {
        "username": "MariaClara",
        "description": "Intermediate — trained on Phishing, weak on Credential Theft",
        # [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
        "weakness":  [0.20, 0.35, 0.40, 0.55, 0.65, 0.35, 0.30, 0.35],
        "fp_rate":   [0.10, 0.10, 0.10, 0.15, 0.05, 0.05, 0.10, 0.10],
        "avg_response_time_ms": 2400,
    },

    "dummy-player-003": {
        "username": "KardingGamer",
        "description": "Advanced gamer — strong overall, very weak on Public Wi-Fi",
        # [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
        "weakness":  [0.10, 0.12, 0.15, 0.10, 0.15, 0.80, 0.12, 0.15],
        "fp_rate":   [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00],
        "avg_response_time_ms": 950,
    },

    "dummy-player-004": {
        "username": "TeacherRosa",
        "description": "Over-blocker — low weakness but very high false positive rate",
        # [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
        "weakness":  [0.20, 0.25, 0.30, 0.25, 0.30, 0.35, 0.25, 0.30],
        "fp_rate":   [0.55, 0.45, 0.40, 0.50, 0.05, 0.05, 0.40, 0.35],
        "avg_response_time_ms": 5500,
    },

    "dummy-player-005": {
        "username": "BenjiPH",
        "description": "Near-expert — only real weakness is Ransomware",
        # [Phish, Smish, Vish, SocEng, CredTh, WiFi, Malw, Ransom]
        "weakness":  [0.05, 0.06, 0.08, 0.05, 0.08, 0.10, 0.06, 0.75],
        "fp_rate":   [0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.00, 0.05],
        "avg_response_time_ms": 720,
    },
}
