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
  4. DERIVED_WEAKNESS   — weakness vectors computed from
                          the dummy sessions above

ATTACK TYPE ORDER (must match cybermorph_env.py exactly):
  Index 0: phishing
  Index 1: rogue_software
  Index 2: password_attack
  Index 3: malvertising
  Index 4: malware
  Index 5: mitm
  Index 6: ddos
  Index 7: driveby
────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — DUMMY PLAYER PROFILES
# Mirrors player_profiles table in PostgreSQL
# security_credits starts at 100 per data dictionary
# map_progress: 0=Home unlocked, 1=Internet Cafe, etc.
# ─────────────────────────────────────────────────────────────────

DUMMY_PLAYERS = [
    {
        "profile_id": "dummy-player-001",
        "username": "JuanDelaCruz",
        "map_progress": 0,
        "security_credits": 100,
        "total_sessions": 3,
        "notes": "Typical beginner. Fails phishing and malvertising often."
    },
    {
        "profile_id": "dummy-player-002",
        "username": "MariaClara",
        "map_progress": 1,
        "security_credits": 340,
        "total_sessions": 8,
        "notes": "Intermediate. Good at phishing, weak on password attacks."
    },
    {
        "profile_id": "dummy-player-003",
        "username": "KardingGamer",
        "map_progress": 2,
        "security_credits": 580,
        "total_sessions": 15,
        "notes": "Advanced gamer. Very weak on MITM — never uses VPN."
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
        "notes": "Near-expert. Only real weakness is DDoS identification."
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
        "notes": "Failed MITM scenario — connected to fake WiFi"
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

    # ── BenjiPH — near-expert, fails only on DDoS ─────────────────
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
        "notes": "Failed to identify school portal DDoS alert"
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

    # ── JuanDelaCruz interactions (beginner, fails phishing a lot) ─
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "phishing",        "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3200},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "malvertising",    "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 2800},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "phishing",        "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4100},
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "malvertising",    "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1900},
    # False positive — blocked a safe phishing-style email
    {"profile_id": "dummy-player-001", "session_id": "sess-001",
     "attack_type": "phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 5500},

    # ── MariaClara interactions (good at phishing, weak passwords) ─
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1200},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "password_attack", "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3800},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "rogue_software",  "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 2900},
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1500},
    # False positive on safe rogue software prompt
    {"profile_id": "dummy-player-002", "session_id": "sess-005",
     "attack_type": "rogue_software",  "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 2100},

    # ── KardingGamer interactions (strong except MITM) ─────────────
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 800},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "malware",         "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 950},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "mitm",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4200},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "mitm",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 3800},
    {"profile_id": "dummy-player-003", "session_id": "sess-009",
     "attack_type": "password_attack", "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 1100},

    # ── TeacherRosa interactions (over-blocker, high false positives)
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 2200},
    # False positives — blocks everything including safe events
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 6100},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "malvertising",    "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 5800},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "rogue_software",  "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 4900},
    {"profile_id": "dummy-player-004", "session_id": "sess-010",
     "attack_type": "phishing",        "is_correct": 0,
     "player_action": "block",         "correct_action": "accept",
     "response_time_ms": 7200},

    # ── BenjiPH interactions (expert, only fails DDoS) ─────────────
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "phishing",        "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 600},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "mitm",            "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 750},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "ddos",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 5100},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "malware",         "is_correct": 1,
     "player_action": "block",         "correct_action": "block",
     "response_time_ms": 820},
    {"profile_id": "dummy-player-005", "session_id": "sess-013",
     "attack_type": "ddos",            "is_correct": 0,
     "player_action": "accept",        "correct_action": "block",
     "response_time_ms": 4800},
]


# ─────────────────────────────────────────────────────────────────
# SECTION 4 — DERIVED WEAKNESS VECTORS
# Computed from DUMMY_ATTACK_LOGS above.
# These are the actual numbers fed into the RL environment
# as starting weakness profiles for training.
#
# How to read:
#   weakness[i] = failure rate for attack type i
#   0.0 = never fails, 1.0 = always fails
#
# Attack type index order:
#   0=phishing, 1=rogue_software, 2=password_attack,
#   3=malvertising, 4=malware, 5=mitm, 6=ddos, 7=driveby
#
# fp_rate[i] = false positive rate for attack type i
#   How often player blocks SAFE events of that type
# ─────────────────────────────────────────────────────────────────

DERIVED_WEAKNESS = {

    "dummy-player-001": {
        "username": "JuanDelaCruz",
        "description": "Beginner — high phishing and malvertising weakness",
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness":  [0.75, 0.50, 0.55, 0.65, 0.50, 0.60, 0.40, 0.45],
        "fp_rate":   [0.35, 0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.10],
        "avg_response_time_ms": 3500,
    },

    "dummy-player-002": {
        "username": "MariaClara",
        "description": "Intermediate — trained on phishing, weak on rogue/password",
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness":  [0.20, 0.60, 0.65, 0.35, 0.40, 0.45, 0.30, 0.35],
        "fp_rate":   [0.10, 0.30, 0.15, 0.10, 0.10, 0.05, 0.05, 0.10],
        "avg_response_time_ms": 2400,
    },

    "dummy-player-003": {
        "username": "KardingGamer",
        "description": "Advanced gamer — strong overall, very weak on MITM",
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness":  [0.10, 0.15, 0.15, 0.10, 0.12, 0.80, 0.25, 0.15],
        "fp_rate":   [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00, 0.05],
        "avg_response_time_ms": 950,
    },

    "dummy-player-004": {
        "username": "TeacherRosa",
        "description": "Over-blocker — low weakness but very high false positive rate",
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness":  [0.20, 0.25, 0.30, 0.20, 0.25, 0.35, 0.30, 0.25],
        "fp_rate":   [0.55, 0.45, 0.50, 0.50, 0.40, 0.30, 0.20, 0.35],
        "avg_response_time_ms": 5500,
    },

    "dummy-player-005": {
        "username": "BenjiPH",
        "description": "Near-expert — only real weakness is DDoS identification",
        # [phish, rogue, pwd,  madv,  malw,  mitm,  ddos,  driveby]
        "weakness":  [0.05, 0.08, 0.08, 0.05, 0.06, 0.10, 0.75, 0.08],
        "fp_rate":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.00],
        "avg_response_time_ms": 720,
    },
}


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING EPISODE SEEDS
# Pre-defined episode configurations for reproducible training.
# The RL environment can pick from these to initialize episodes
# instead of purely random starts — gives more realistic variety.
# ─────────────────────────────────────────────────────────────────

TRAINING_EPISODE_SEEDS = [
    # Each entry = one episode starting configuration
    # profile_key: which DERIVED_WEAKNESS profile to start from
    # noise_scale: how much random noise to add to weakness vector
    # map_context: which map's actions are available
    # difficulty_label: human-readable difficulty reference

    {"profile_key": "dummy-player-001", "noise_scale": 0.05,
     "map_context": "home",           "difficulty_label": "beginner"},

    {"profile_key": "dummy-player-002", "noise_scale": 0.08,
     "map_context": "internet_cafe",  "difficulty_label": "intermediate"},

    {"profile_key": "dummy-player-003", "noise_scale": 0.10,
     "map_context": "company_office", "difficulty_label": "advanced"},

    {"profile_key": "dummy-player-004", "noise_scale": 0.05,
     "map_context": "home",           "difficulty_label": "over_blocker"},

    {"profile_key": "dummy-player-005", "noise_scale": 0.12,
     "map_context": "public_park",    "difficulty_label": "near_expert"},

    # Mixed difficulty seeds for more robust training
    {"profile_key": "dummy-player-001", "noise_scale": 0.15,
     "map_context": "internet_cafe",  "difficulty_label": "beginner_pushed"},

    {"profile_key": "dummy-player-002", "noise_scale": 0.10,
     "map_context": "company_office", "difficulty_label": "intermediate_hard"},

    {"profile_key": "dummy-player-003", "noise_scale": 0.08,
     "map_context": "public_park",    "difficulty_label": "advanced_full"},
]
