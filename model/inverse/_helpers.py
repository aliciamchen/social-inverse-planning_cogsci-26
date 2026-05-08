"""
Shared infrastructure for inverse-planning fit + predict scripts.

Each experiment has its own thin fit/predict script that imports the helpers
it needs from this module. Shared concerns:

  - Loss functions (intimacy NLL, reward BCE NLL)
  - Observer fit loops (single-param alpha_observer)
  - Data loaders (per experiment)
  - Frozen-actor-param loaders (forward fits → inverse fits)
  - Fitted-α_observer loader (fit_results → predict scripts)
  - Variant registry + table-kwargs helper used by CV scripts

Forward-side helpers like `_fit_with_adam` are imported from
`forward/_shared.py` (available via the path setup at the top of any caller).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from tables import SCENARIO_TO_IDX
from utils import get_project_root
from _shared import _fit_with_adam  # forward/_shared.py


# ==============================================================================
# Loss functions
# ==============================================================================


@jax.jit
def compute_intimacy_mse(posterior, belief_update):
    """Per-trial squared error between model and human intimacy belief updates.

    posterior: shape (101,) over intimacy levels 0-100, summing to 1.
    belief_update: human's (posterior_rating - prior_rating) / 100, in [-1, 1].

    Model belief update = E[posterior intimacy] / 100 − 0.5, assuming the
    model's prior over intimacy is uniform on [0, 1] (so the prior mean is
    0.5). Summed across trials this becomes SSE; the fit minimizes that
    quantity, equivalent to least-squares regression of model on human
    belief updates.
    """
    intimacy_grid = jnp.arange(101) / 100.0  # 0, 0.01, ..., 1.0
    expected_posterior = jnp.sum(intimacy_grid * posterior)  # in [0, 1]
    model_belief_update = expected_posterior - 0.5
    return (model_belief_update - belief_update) ** 2


@jax.jit
def compute_reward_mse(p_high, belief_update):
    """Per-trial squared error between model and human desire-inference belief
    updates, on the rating scale.

    p_high: model's posterior P(reward=high), in (0, 1).
    belief_update: human's (posterior_rating - prior_rating) / 100, in [-1, 1].

    The model's prior P(high) is uniform = 0.5, so its rating-scale belief
    update is p_high - 0.5.
    """
    return (p_high - 0.5 - belief_update) ** 2


# ==============================================================================
# Frozen-param loaders
# ==============================================================================


def load_fitted_params(filepath: str = None) -> dict:
    """Load frozen actor parameters from forward planning fit results.

    Returns a dict: model_name -> dict of every `param_*` column present in that
    row (stripped of the `param_` prefix). Missing/NaN columns are omitted, so
    each model keeps only its own parameter set.

    Defaults to the canonical `food_forw_intimacy_desire/fit_results.csv`.
    """
    if filepath is None:
        filepath = (
            get_project_root()
            / "model"
            / "outputs"
            / "food_forw_intimacy_desire"
            / "fit_results.csv"
        )
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        model_name = row["model"]
        model_params = {}
        for col in df.columns:
            if col.startswith("param_") and pd.notna(row[col]):
                model_params[col.replace("param_", "")] = float(row[col])
        params[model_name] = model_params
    return params


def load_fitted_alpha_observer(filepath=None) -> dict:
    """Load fitted alpha_observer values from inverse planning fit_results.csv.

    `filepath` is a single path (single-experiment file). If None, reads from
    both inverse experiment dirs and merges (intimacy + reward).

    Returns dict with (model, experiment) -> alpha_observer. Defaults to 1.0 if NaN.
    """
    if filepath is not None:
        paths = [Path(filepath)]
    else:
        outputs_root = get_project_root() / "model" / "outputs"
        paths = [
            outputs_root / "food_inv_intimacy_desire_alt" / "fit_results.csv",
            outputs_root / "food_inv_desire_intimacy_alt" / "fit_results.csv",
        ]
    alpha_obs = {}
    for path in paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row["model"], row["experiment"])
            alpha_val = row["alpha_observer"]
            alpha_obs[key] = alpha_val if pd.notna(alpha_val) else 1.0
    return alpha_obs


# ==============================================================================
# Data loaders (one per experiment)
# ==============================================================================


def _attach_belief_update(data: pd.DataFrame, rating_col: str) -> pd.DataFrame:
    """Pivot to compute belief update = posterior_rating - prior_rating per
    (subject_id, scenario_label, action_condition), then return one row per
    posterior trial with `belief_update` (in [-1, 1]) attached.

    Mirrors the qmds' `calculate_belief_update()` helper but in pandas.
    """
    pivot = (
        data.pivot_table(
            index=["subject_id", "scenario_label", "action_condition"],
            columns="stage",
            values=rating_col,
            aggfunc="first",
        )
        .reset_index()
    )
    pivot["belief_update"] = (pivot["posterior"] - pivot["prior"]) / 100.0
    posterior_rows = data[data["stage"] == "posterior"].copy()
    return posterior_rows.merge(
        pivot[["subject_id", "scenario_label", "action_condition", "belief_update"]],
        on=["subject_id", "scenario_label", "action_condition"],
        how="left",
    ).dropna(subset=["belief_update"])


def load_intimacy_alt_data(filepath: str = None):
    """food_inv_intimacy_desire_alt — observer infers intimacy under known motivation.

    Returns per-trial belief updates (posterior − prior intimacy rating, scaled
    to [-1, 1]) rather than raw posteriors, so the fit loss compares model and
    human belief updates directly.
    """
    if filepath is None:
        filepath = get_project_root() / "data" / "food_inv_intimacy_desire_alt" / "main_trials_long.csv"
    print("Loading food_inv_intimacy_desire_alt data...")
    data = pd.read_csv(filepath)
    data = _attach_belief_update(data, "intimacy_rating")
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    motivation_map = {"low": 0, "high": 1}
    data["reward_condition"] = data["motivation"].map(motivation_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    reward_condition = jnp.array(data["reward_condition"].values)
    belief_update = jnp.array(data["belief_update"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior trials with belief updates")
    return data, action, reward_condition, belief_update, scenario_idx


def load_desire_alt_data(filepath: str = None):
    """food_inv_desire_intimacy_alt — observer infers desire under known intimacy.

    Returns per-trial belief updates on the rating scale:
    (posterior_rating - prior_rating) / 100, in [-1, 1]. The desire rating
    is P(high) on a 0-100 scale, so the rating-scale belief update is the
    raw probability shift.
    """
    if filepath is None:
        filepath = get_project_root() / "data" / "food_inv_desire_intimacy_alt" / "main_trials_long.csv"
    print("Loading food_inv_desire_intimacy_alt data...")
    data = pd.read_csv(filepath)
    data = _attach_belief_update(data, "response")
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["intimacy_idx"] = data["intimacy"].map(intimacy_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    intimacy_condition = jnp.array(data["intimacy_idx"].values)
    belief_update = jnp.array(data["belief_update"].values)
    scenario_idx = jnp.array(data["scenario_idx"].values)
    print(f"Loaded {len(data)} posterior trials with belief updates")
    return data, action, intimacy_condition, belief_update, scenario_idx


# ==============================================================================
# Single-α_observer fit loop
# ==============================================================================


def _fit_alpha_observer(
    observer_fn,
    actor_params: dict,
    actor_kwarg_names,
    action: jnp.ndarray,
    scenario_idx: jnp.ndarray,
    conditioning: jnp.ndarray,
    response: jnp.ndarray,
    mse_fn,
    posterior_slicer,
    table_kwargs: dict,
    lr: float = 0.1,
    max_steps: int = 1000,
    verbose: bool = True,
):
    """Fit alpha_observer by minimizing summed squared error (SSE) between
    model and human belief updates."""
    actor_kwargs = {k: actor_params[k] for k in actor_kwarg_names}

    def observer_table(alpha_observer):
        return observer_fn(
            **actor_kwargs, alpha_observer=alpha_observer,
            **table_kwargs,
        )

    def trial_se(alpha_observer, a, s, c, resp):
        table = observer_table(alpha_observer)
        slc = posterior_slicer(table, a, s, c)
        return mse_fn(slc, resp)

    vmap_trial_se = jax.vmap(
        lambda alpha_obs, a, s, c, resp: trial_se(alpha_obs, a, s, c, resp),
        in_axes=(None, 0, 0, 0, 0),
    )

    def loss_fn(params):
        return jnp.sum(
            vmap_trial_se(params[0], action, scenario_idx, conditioning, response)
        )

    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad_count = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)
        grad_mag = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or grad_mag < 1e-10:
            zero_grad_count += 1
            if zero_grad_count >= 5:
                if verbose:
                    print("  Gradient zero/NaN for 5 consecutive steps; alpha_observer=1.0")
                return 1.0, float(loss)
        else:
            zero_grad_count = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  Step {step}, SSE: {loss:.4f}, alpha_observer: {params[0]:.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  Loss increased at step {step}, stopping")
            break
        prev_loss = loss

    best_sse = float(loss_fn(params))
    final_alpha = float(params[0])
    if jnp.isnan(final_alpha):
        final_alpha = 1.0
    if verbose:
        print(f"  Final SSE: {best_sse:.4f}, alpha_observer: {final_alpha:.4f}")
    return final_alpha, best_sse


def fit_intimacy_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, conditioning, response, table_kwargs, **kwargs,
):
    """For observers whose table is (action, scenario, intimacy, conditioning_axis)."""
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=conditioning,
        response=response,
        mse_fn=compute_intimacy_mse,
        posterior_slicer=lambda tab, a, s, c: tab[a, s, :, c],
        table_kwargs=table_kwargs,
        **kwargs,
    )


def fit_reward_observer(
    observer_fn, actor_params, actor_kwarg_names,
    action, scenario_idx, intimacy_condition, response, table_kwargs, **kwargs,
):
    """For observers whose table is (action, scenario, relationship, reward_condition)."""
    return _fit_alpha_observer(
        observer_fn=observer_fn,
        actor_params=actor_params,
        actor_kwarg_names=actor_kwarg_names,
        action=action,
        scenario_idx=scenario_idx,
        conditioning=intimacy_condition,
        response=response,
        mse_fn=compute_reward_mse,
        posterior_slicer=lambda tab, a, s, i: tab[a, s, i, 1],
        table_kwargs=table_kwargs,
        **kwargs,
    )


# ==============================================================================
# Prediction grid helpers
# ==============================================================================


def compute_expected_intimacy(df: pd.DataFrame) -> pd.DataFrame:
    """Expected intimacy from posterior over the 0-100 grid."""
    df = df.copy()
    df["intimacy_scaled"] = df["intimacy"] * 100
    summary = df.groupby(
        ["scenario_label", "action", "reward_condition", "model"],
        dropna=False,
    ).apply(
        lambda g: pd.Series({"expected_intimacy": (g["intimacy_scaled"] * g["density"]).sum()})
    ).reset_index()
    return summary


def compute_p_high_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Extract P(high reward) for desire-inference experiments."""
    df_high = df[df["reward_condition"] == "high"].copy()
    df_high = df_high.rename(columns={"density": "p_high_reward"})
    df_high["p_high_reward"] = df_high["p_high_reward"] * 100
    df_high = df_high.drop(columns=["reward_condition"])
    return df_high


# ==============================================================================
# Variant registry (used by CV scripts that share observer functions)
# ==============================================================================

from observers import (  # noqa: E402
    observer_intimacy_base,
    observer_intimacy_discomfort_only,
    observer_intimacy_full,
    observer_reward_base,
    observer_reward_discomfort_only,
    observer_reward_full,
)
from tables import LLM_TABLES, load_lm_v  # noqa: E402

# Variant registry: name → (intimacy observer, reward observer, actor kwarg
# names, whether the variant uses V).
ACCESS_VARIANTS = {
    "full": (observer_intimacy_full, observer_reward_full,
             ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_intimacy_discomfort_only, observer_reward_discomfort_only,
                        ["alpha", "w_d", "gamma"], False),
    "base": (observer_intimacy_base, observer_reward_base,
             ["alpha", "w_v", "w_e"], True),
}


def alt_table_kwargs(uses_v):
    """Table kwargs for the 4-action observers."""
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw
