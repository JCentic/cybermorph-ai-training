# RESULTS - the winner rule and (later) the results of the three algorithms

**Written:** 2026-11-11, **before any final result exists.** The rule below is fixed first so nobody can choose the result they like. If the rule must change, change it here with a date and a reason, BEFORE the five-seed run.

## 1. What we compare
- PPO, DQN and A2C (Stable-Baselines3, `MlpPolicy`), each with the settings in its `train_<algo>.py`. No tuning for any one of them.
- Same environment, same 300,000 steps, same seeds. Phase A: seed 42 only (a sanity run, no winner). Phase B: seeds 42, 43, 44, 45, 46 (the real comparison).
- The environment (2026-11-11): episodes are as long as a real run (8, 15, 22 and 30 cards on the four maps), the agent sees estimates built from the last 50 answers, and an invalid action costs a step.

## 2. Who the agents are tested on (held-out players)
- The agents train on the five invented players (`DERIVED_WEAKNESS`).
- The evaluation uses players the agent **never trains on**: the three real groups built from the testers' play data (`python -m analysis.build_profiles --write` makes `data/profiles_real.json`: real-beginner, real-intermediate, real-expert).
- If `data/profiles_real.json` is missing, the training script prints a note and the evaluation uses the training players. **A comparison made that way is not held out and must be reported as such.**
- Evaluation: every 10,000 steps, 10 episodes on each of the four map stages, with the best action each time (`deterministic=True`).

## 3. The number that decides
**Reward per card** = the reward of an episode divided by its number of cards. It is the only fair number across stages, because the maps have different lengths.

For each algorithm:
1. Take the pooled reward per card (all four stages together, stage 0 in the CSV) at the **last five evaluations of a seed** (steps 260,000 to 300,000) and average them. That is the seed's score.
2. The algorithm's score is the **mean of its five seed scores**. Also write down the **standard deviation across the five seeds**.

## 4. Who wins
1. **Gate:** the average number of invalid actions per episode at the last five evaluations must be **below 0.5**. An agent that keeps choosing actions its map does not allow is not good enough, whatever its reward.
2. Among the algorithms that pass the gate, the **highest score** wins.
3. **Tie rule:** if the highest score is not bigger than the second highest by more than the standard deviation across seeds of the highest one, it is a tie. In a tie the algorithm with the **smaller standard deviation** wins (it is more reliable).
4. Also report each stage separately (map 1 to 4). The stage numbers do not change the winner, but a panelist may ask.

## 5. After the winner is chosen (the freeze)
- Pick the best seed of the winning algorithm by the same score. Copy its model to `models/frozen/` and write the table below.
- After that, no retraining. Any change to the categories, the stages, the observation, the episode lengths or the decision log means a new full run, a new export and a new parity file.

## 6. Results
*Not run yet.* Fill in after Phase B.

| Algorithm | Score (mean reward per card) | Std across 5 seeds | Invalid actions per episode | Passes the gate |
|---|---|---|---|---|
| PPO | | | | |
| DQN | | | | |
| A2C | | | | |

Winner: *(to fill)*. Seed chosen: *(to fill)*. Date: *(to fill)*. Library versions: see `requirements.txt`. Players for the evaluation: *(real groups from data/profiles_real.json, date and counts)*.

## 7. Notes for the defense
- The old numbers (PPO 45.03, DQN 38.93, A2C 20.88) are from the environment before 2026-11-05 and must not be quoted.
- The evaluation players are built from about 10 testers, so they are a comparison, not a proof. More testers will make them firmer.
