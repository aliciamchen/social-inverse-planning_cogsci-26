# Data codebook

Each experiment folder contains `main_trials.csv`, `main_trials_long.csv`, and `exit_survey.csv`.

The three experiments included in this repo are:

- `food_forw_intimacy_desire/` — Experiment 1, forward planning
- `food_inv_intimacy_desire_alt/` — Experiment 2a, intimacy inference (alternatives shown)
- `food_inv_desire_intimacy_alt/` — Experiment 2b, desire inference (alternatives shown)

## Terminology note

Internal variable names in the inverse-planning experiments use "reward" (e.g. `p_high_reward`, `reward_condition`) or "motivation" rather than "desire" — the terminology was changed to "desire" in the paper after the experiments were run, for clarity.

## Anonymization

The raw jsPsych JSON output (collected via Prolific) is not included in this repo. Prolific Participant IDs in the raw data were deterministically mapped to UUID5 values (namespace `6ba7b810-9dad-11d1-80b4-00c04fd430c8`) before being written to the processed CSVs. The mapping was never persisted to disk. The CSVs in this repo only ever contain the anonymized UUIDs.

## Exclusion criteria

Participants are excluded from analysis if either is true:

- Failed attention check (`attention_passed != True`)
- Got 0 correct on the memory check (`memory_correct_count == 0`)

`main_trials_long.csv` reflects exclusions. `main_trials.csv` does not (it includes all collected participants, with the exclusion flags available via a join on `subject_id` to `exit_survey.csv`).

## Exit survey (all experiments)

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `gender` | Self-reported gender |
| `age` | Self-reported age |
| `understood` | Whether the participant understood the task ("yes"/"no") |
| `comments` | Free-text comments |
| `attention_passed` | Whether the participant passed the attention check (True/False) |
| `memory_correct_count` | Number of correct responses on the memory check (0–3) |

## Forward planning (`food_forw_intimacy_desire/`)

Each participant sees scenarios across 4 intimacy levels (0, 50, 75, 100) and 2 motivation conditions (low, high). For each (scenario, intimacy, motivation) cell, they distribute probability mass across 4 actions ranging from no sharing (action 0) to high-saliva-risk sharing (action 3).

**main_trials.csv** (wide format — one row per trial):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier (e.g. "hike", "wedding", "basketball") |
| `intimacy_condition` | Relationship closeness level (0, 50, 75, or 100) |
| `reward_condition` | Motivation condition ("low" or "high") |
| `action_0` | Probability allocated to action 0 (no sharing) |
| `action_1` | Probability allocated to action 1 (sharing with no saliva risk) |
| `action_2` | Probability allocated to action 2 (sharing with moderate saliva risk) |
| `action_3` | Probability allocated to action 3 (sharing with high saliva risk) |

**main_trials_long.csv** (long format — one row per action):

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `intimacy` | Relationship closeness level (0, 50, 75, or 100) |
| `motivation` | Motivation condition ("low" or "high") |
| `action` | Action index (0–3) |
| `p_action` | Probability allocated to this action |

## Intimacy inference, alternatives shown (`food_inv_intimacy_desire_alt/`)

Participants observe an actor's chosen action under a known motivation, then rate the intimacy of the relationship. Each (action, motivation) cell is rated twice — once at the prior stage (before observing the action) and once at the posterior stage (after).

**main_trials.csv** and **main_trials_long.csv**:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `action_condition` | Observed action ("action_0" through "action_3") |
| `reward_condition` / `motivation` | Motivation condition shown ("low" or "high"); the long format uses `motivation` |
| `stage` | Measurement timing ("prior" = before seeing the action, "posterior" = after) |
| `intimacy_rating` | Participant's intimacy estimate (0–100 scale) |

## Desire inference, alternatives shown (`food_inv_desire_intimacy_alt/`)

Participants observe an actor's chosen action under a known intimacy level, then rate the probability that the actors had high motivation. Same prior/posterior structure as Exp 2a.

**main_trials.csv** and **main_trials_long.csv**:

| Column | Description |
|--------|-------------|
| `subject_id` | Anonymized participant UUID |
| `scenario_label` | Scenario identifier |
| `action_condition` | Observed action ("action_0" through "action_3") |
| `intimacy_condition` / `intimacy` | Relationship closeness level shown (0, 50, 75, or 100); the long format uses `intimacy` |
| `stage` | Measurement timing ("prior" or "posterior") |
| `response` | Participant's high-motivation probability estimate (0–100 scale) |
