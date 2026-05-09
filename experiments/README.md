# Experiments

jsPsych code for the three experiments included in the CogSci 2026 paper. Each experiment subdirectory is a self-contained jsPsych 8.x task that collects data via the jsPsych-contrib/pipe plugin.

## Terminology note

In the inverse-planning experiments, internal variable names use "reward" (e.g., `p_high_reward`, `reward_condition`) rather than "desire" — we changed the terminology to "desire" after running the experiments, for clarity.

## Scenarios (`scenarios.csv`)

Spreadsheet of scenarios used to generate stimuli for the experiments. The CSV is a generated artifact — the source of truth is `scenarios.py`, which writes the CSV when run. To edit scenarios, modify `scenarios.py` and regenerate:

```bash
uv run python experiments/scenarios.py
```

| Column | Description |
|--------|-------------|
| `scenario_label` | Scenario identifier used in data files |
| `name_0`, `name_1` | Character names in the vignette |
| `vignette` | Base scenario description |
| `reward_low` | Text describing low motivation condition |
| `reward_high` | Text describing high motivation condition |
| `action_0` | Description of action 0 |
| `action_1` | Description of action 1 |
| `action_2` | Description of action 2 |
| `action_3` | Description of action 3 |

### Action scale

Actions are ordered by degree of saliva-sharing risk:
- **Action 0**: No sharing
- **Action 1**: Sharing with no saliva risk (e.g., cutting food in half, using separate utensils)
- **Action 2**: Sharing with moderate saliva risk (e.g., eating from opposite ends)
- **Action 3**: Sharing with high saliva risk (e.g., same utensil, same bite location)

## Stimuli generation

Each experiment directory has a `json/stimuli.json` that is generated from `scenarios.csv` by `csv_to_json.py`. After editing `scenarios.py`, regenerate the CSV and then propagate it into each experiment:

```bash
uv run python experiments/scenarios.py
uv run python experiments/csv_to_json.py
```

## Experiments

- [food_forw_intimacy_desire](food_forw_intimacy_desire/README.md) — Experiment 1, forward planning. Actors choose among four candidate actions given intimacy (4 levels) × desire (2 levels).
- [food_inv_intimacy_desire_alt](food_inv_intimacy_desire_alt/README.md) — Experiment 2a, intimacy inference. Observers infer intimacy from an observed action; all four candidate actions shown.
- [food_inv_desire_intimacy_alt](food_inv_desire_intimacy_alt/README.md) — Experiment 2b, desire inference. Observers infer desire from an observed action; all four candidate actions shown.
