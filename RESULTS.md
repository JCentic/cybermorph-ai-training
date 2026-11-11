# RESULTS - the winner rule and (later) the results of the three algorithms

**Written:** 2026-11-11, **before any final result exists.** The rule below is fixed first so nobody can choose the result they like. If the rule must change, change it here with a date and a reason, BEFORE the five-seed run.

## Part A - the research question: WHY do we use RL for the game?
**Written 2026-11-11, before any final result.** The purpose of the training is to answer this question with evidence. Comparing PPO, DQN and A2C (Part B) only chooses the best of the three; Part A compares the winner with ways of choosing cards that need NO learning.

### A1. The rivals (baselines.py)
| Policy | What it does |
|---|---|
| random | a random valid card |
| standard pattern | the same fixed order for everyone (the game without the AI) |
| greedy weakest | always the card the player seems weakest at |
| greedy with variety | the same, but never the same category three times in a row (the strong hand-written rival) |
| RL (the winner of Part B) | the frozen model |

All policies only choose valid cards. All play the **same episodes** (same seeds) against the **same held-out players** (the real-data groups).

### A2. The learning model (an ASSUMPTION, not a measurement)
- A real player gets better at what they practise. In the comparison world (`learning=True`) the simulated player does too: after a card of a category, that weakness moves toward a floor by `PRACTISE_RATE` (0.06) of the gap; every step the player forgets `FORGET_RATE` (0.01) of the gap back toward the start. Floors: 0.05 for weakness and 0.02 for blocking safe cards.
- The numbers are **chosen, not measured**. So the comparison is repeated for several practise rates (**0.03, 0.06 and 0.10**), and a conclusion counts only if it holds at all three.
- The training itself does **not** use learning (the training world is unchanged).

### A3. What we measure (`python evaluate_policies.py`)
1. **Skill gain** (the main result): how much the player's average weakness fell during the run. Higher is better.
2. **Weak-spot lift:** how much more often than by chance the policy chose a card in the player's weakest categories (the weakest quarter of the map's categories, at least one). 1.0 is chance.
3. **Coverage:** the share of the map's categories the player met (a coach that shows one category forever is not good).
4. **Reward per card** and **invalid actions** (the training reward; invalid must stay below 0.5 per episode).
Every number comes with a 95% confidence interval (100 episodes per player and stage).

### A4. What counts as "RL is justified" (the claims, decided in advance)
- **Claim 1, it finds weak spots:** the weak-spot lift of RL is higher than random and than the standard pattern, with confidence intervals that do not overlap.
- **Claim 2, it helps players improve (in the simulation):** the skill gain of RL is higher than the standard pattern and than random, at all three practise rates.
- **Claim 3, against hand-written rules:** compare RL with "greedy with variety". If RL is higher with non-overlapping intervals, say "better". If the intervals overlap, say "comparable" and argue the other reasons for RL: it learns from noisy estimates of a few answers, it balances weak spots with variety without a hand-written rule, and it needs no new rule for a new map or category. If a rule is better, say so.
- **Not shown by this test:** that real players learn more. That needs a real test with testers (one group with the adaptive game, one with the standard game) once the AI is in the game.

### A5. Results of Part A
*Not run yet.* The first run (4 baselines only, 40 episodes) showed: greedy picks weak spots about 1.6 times more than chance, random and the standard pattern about 1.0; the skill gain of random (0.032) was lower than that of the standard pattern and the greedy rules (about 0.047 to 0.048), which are close to each other. The RL numbers wait for the models trained in the new environment (the models in `models/` now are old and wrong for it).

| Policy | Skill gain | Weak-spot lift | Coverage | Invalid |
|---|---|---|---|---|
| random | | | | |
| standard pattern | | | | |
| greedy weakest | | | | |
| greedy with variety | | | | |
| RL (winner) | | | | |

## Part B - which of the three algorithms (PPO, DQN, A2C) is best

### B1. What we compare
- PPO, DQN and A2C (Stable-Baselines3, `MlpPolicy`), each with the settings in its `train_<algo>.py`. No tuning for any one of them.
- Same environment, same 300,000 steps, same seeds. Phase A: seed 42 only (a sanity run, no winner). Phase B: seeds 42, 43, 44, 45, 46 (the real comparison).
- The environment (2026-11-11): episodes are as long as a real run (8, 15, 22 and 30 cards on the four maps), the agent sees estimates built from the last 50 answers, and an invalid action costs a step.

### B2. Who the agents are tested on (held-out players)
- The agents train on the five invented players (`DERIVED_WEAKNESS`).
- The evaluation uses players the agent **never trains on**: the three real groups built from the testers' play data (`python -m analysis.build_profiles --write` makes `data/profiles_real.json`: real-beginner, real-intermediate, real-expert).
- If `data/profiles_real.json` is missing, the training script prints a note and the evaluation uses the training players. **A comparison made that way is not held out and must be reported as such.**
- Evaluation: every 10,000 steps, 10 episodes on each of the four map stages, with the best action each time (`deterministic=True`).

### B3. The number that decides
**Reward per card** = the reward of an episode divided by its number of cards. It is the only fair number across stages, because the maps have different lengths.

For each algorithm:
1. Take the pooled reward per card (all four stages together, stage 0 in the CSV) at the **last five evaluations of a seed** (steps 260,000 to 300,000) and average them. That is the seed's score.
2. The algorithm's score is the **mean of its five seed scores**. Also write down the **standard deviation across the five seeds**.

### B4. Who wins
1. **Gate:** the average number of invalid actions per episode at the last five evaluations must be **below 0.5**. An agent that keeps choosing actions its map does not allow is not good enough, whatever its reward.
2. Among the algorithms that pass the gate, the **highest score** wins.
3. **Tie rule:** if the highest score is not bigger than the second highest by more than the standard deviation across seeds of the highest one, it is a tie. In a tie the algorithm with the **smaller standard deviation** wins (it is more reliable).
4. Also report each stage separately (map 1 to 4). The stage numbers do not change the winner, but a panelist may ask.

### B5. After the winner is chosen (the freeze)
- Pick the best seed of the winning algorithm by the same score. Copy its model to `models/frozen/` and write the table below.
- After that, no retraining. Any change to the categories, the stages, the observation, the episode lengths or the decision log means a new full run, a new export and a new parity file.

### B6. Results
*Not run yet.* Fill in after Phase B.

| Algorithm | Score (mean reward per card) | Std across 5 seeds | Invalid actions per episode | Passes the gate |
|---|---|---|---|---|
| PPO | | | | |
| DQN | | | | |
| A2C | | | | |

Winner: *(to fill)*. Seed chosen: *(to fill)*. Date: *(to fill)*. Library versions: see `requirements.txt`. Players for the evaluation: *(real groups from data/profiles_real.json, date and counts)*.

## Notes for the defense
- The old numbers (PPO 45.03, DQN 38.93, A2C 20.88) are from the environment before 2026-11-05 and must not be quoted.
- The evaluation players are built from about 10 testers, so they are a comparison, not a proof. More testers will make them firmer.
