"""
One-off exploration: refit alpha_observer for the desire-inference inverse
(food_inv_desire_intimacy_alt) on the RATING scale instead of the log-odds
scale used in the paper. Compare fitted alpha_obs, MSE, and cell-mean
predictions against humans, to see whether the saturation visible in
Fig 2b's Full-model panel persists when the loss treats near-0/1
posteriors more leniently.

Does not touch model/outputs/. Prints results.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "model"))
sys.path.insert(0, str(_project_root / "model" / "forward"))
sys.path.insert(0, str(_project_root / "model" / "inverse"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pandas as pd  # noqa: E402

from _helpers import (  # noqa: E402
    _attach_belief_update,  # rating-scale belief update helper
    load_fitted_params,
)
from observers import (  # noqa: E402
    observer_reward_base,
    observer_reward_discomfort_only,
    observer_reward_full,
)
from tables import LLM_TABLES, SCENARIO_TO_IDX, load_lm_v  # noqa: E402
from utils import get_project_root  # noqa: E402


EXPERIMENT_SLUG = "food_inv_desire_intimacy_alt"

VARIANTS = {
    "full": (observer_reward_full, ["alpha", "w_v", "w_d", "w_e", "gamma"], True),
    "discomfort_only": (observer_reward_discomfort_only, ["alpha", "w_d", "gamma"], False),
    "base": (observer_reward_base, ["alpha", "w_v", "w_e"], True),
}


def _table_kwargs(uses_v):
    kw = {"access_table": LLM_TABLES["access"], "effort_table": LLM_TABLES["effort"]}
    if uses_v:
        kw["v_table"] = load_lm_v("food")
    return kw


def load_desire_data_rating_scale():
    """Load food_inv_desire_intimacy_alt with belief updates on the rating scale.

    `response` rating is P(high) on a 0-100 scale, so belief update on the
    rating scale is (posterior - prior) / 100 in [-1, 1]. Mirrors what
    Exp 2a uses for intimacy inference.
    """
    fp = get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
    data = pd.read_csv(fp)
    data = _attach_belief_update(data, "response")
    data["action"] = data["action_condition"].str.replace("action_", "").astype(int)
    intimacy_map = {0: 0, 50: 1, 75: 2, 100: 3}
    data["intimacy_idx"] = data["intimacy"].map(intimacy_map)
    data["scenario_idx"] = data["scenario_label"].map(SCENARIO_TO_IDX)

    action = jnp.array(data["action"].values)
    intimacy_condition = jnp.array(data["intimacy_idx"].values)
    belief_update = jnp.array(data["belief_update"].values)  # (post - prior)/100
    scenario_idx = jnp.array(data["scenario_idx"].values)
    return data, action, intimacy_condition, belief_update, scenario_idx


@jax.jit
def compute_reward_rating_mse(p_high, belief_update_rating):
    """Squared error between model and human belief updates on the rating scale.

    Model p_high in (0, 1); model belief update on the rating scale is
    p_high - 0.5 (uniform model prior = 0.5).
    Human belief_update_rating = (posterior - prior) / 100, in [-1, 1].
    """
    return (p_high - 0.5 - belief_update_rating) ** 2


def fit_alpha_obs_rating(
    obs_fn, actor_params, kw_names, action, scenario_idx,
    intimacy_condition, belief_update, table_kwargs,
    lr=0.1, max_steps=1000, verbose=True,
):
    actor_kwargs = {k: actor_params[k] for k in kw_names}

    def observer_table(alpha_observer):
        return obs_fn(**actor_kwargs, alpha_observer=alpha_observer, **table_kwargs)

    def trial_se(alpha_observer, a, s, i, resp):
        table = observer_table(alpha_observer)
        p_high = table[a, s, i, 1]  # P(reward = HIGH)
        return compute_reward_rating_mse(p_high, resp)

    vmap_se = jax.vmap(
        lambda ao, a, s, i, r: trial_se(ao, a, s, i, r),
        in_axes=(None, 0, 0, 0, 0),
    )

    def loss_fn(params):
        return jnp.sum(
            vmap_se(params[0], action, scenario_idx, intimacy_condition, belief_update)
        )

    params = jnp.array([1.0])
    grad_fn = jax.value_and_grad(loss_fn)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    prev_loss = None
    zero_grad = 0
    for step in range(max_steps):
        loss, grad = grad_fn(params)
        gmag = float(jnp.abs(grad[0]))
        if jnp.isnan(grad[0]) or gmag < 1e-10:
            zero_grad += 1
            if zero_grad >= 5:
                if verbose:
                    print("  zero/NaN grad x5; stopping with alpha_observer=1.0")
                return 1.0, float(loss)
        else:
            zero_grad = 0

        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 0.01, 10.0)

        if verbose and step % 200 == 0:
            print(f"  step {step}: SSE={float(loss):.4f}, alpha_obs={float(params[0]):.4f}")

        if prev_loss is not None and loss > prev_loss + 1e-4:
            if verbose:
                print(f"  loss increased at step {step}, stopping")
            break
        prev_loss = loss

    final_alpha = float(params[0])
    if jnp.isnan(final_alpha):
        final_alpha = 1.0
    return final_alpha, float(loss_fn(params))


def predictions_at_alpha(obs_fn, actor_params, kw_names, alpha_obs, table_kwargs):
    """Return the full observer table at the fitted alpha_obs (shape: A, S, I, R)."""
    actor_kwargs = {k: actor_params[k] for k in kw_names}
    return np.asarray(
        obs_fn(**actor_kwargs, alpha_observer=alpha_obs, **table_kwargs)
    )


def cell_means(data, model_p_high_table, action_arr, scenario_arr, intim_arr):
    """Aggregate human and model belief updates by (action, intimacy)."""
    # Model belief update per trial on the rating scale.
    n = len(data)
    model_bu = np.empty(n)
    for i in range(n):
        p_high = float(model_p_high_table[
            int(action_arr[i]), int(scenario_arr[i]), int(intim_arr[i]), 1
        ])
        model_bu[i] = p_high - 0.5
    out = data.copy()
    out["model_bu_rating"] = model_bu
    summary = (
        out.groupby(["action", "intimacy"])
        .agg(
            human_bu=("belief_update", "mean"),
            model_bu=("model_bu_rating", "mean"),
            n=("belief_update", "size"),
        )
        .reset_index()
    )
    return summary


def main():
    print("=" * 60)
    print(f"Refit alpha_obs on RATING scale for {EXPERIMENT_SLUG}")
    print("(comparison vs paper's log-odds-scale fit)")
    print("=" * 60)

    actor_params = load_fitted_params()
    data, action, intim_cond, bu_rating, scenario_idx = load_desire_data_rating_scale()
    n = len(bu_rating)
    print(f"\nLoaded {n} posterior trials")
    print(f"Human belief update (rating scale) range: "
          f"{float(bu_rating.min()):.3f}, {float(bu_rating.max()):.3f}")
    print(f"Human belief update mean: {float(bu_rating.mean()):.3f}")

    # Reference: paper's log-odds-fit numbers from outputs/fit_results.csv
    paper_path = (
        get_project_root() / "model" / "outputs" / EXPERIMENT_SLUG / "fit_results.csv"
    )
    paper = pd.read_csv(paper_path).set_index("model")
    print("\nPaper (log-odds-scale) fit:")
    print(paper[["alpha_observer", "mse"]].round(4).to_string())

    rows = []
    summaries = {}
    for name, (obs_fn, kw_names, uses_v) in VARIANTS.items():
        if name not in actor_params:
            continue
        print(f"\n{'-'*40}\nFitting {name} (rating-scale MSE)\n{'-'*40}")
        tk = _table_kwargs(uses_v)
        a_obs, sse = fit_alpha_obs_rating(
            obs_fn, actor_params[name], kw_names,
            action, scenario_idx, intim_cond, bu_rating, tk,
        )
        rating_mse = sse / n

        # For apples-to-apples comparison, also recompute the log-odds MSE
        # at the rating-scale-fitted alpha_obs.
        table = predictions_at_alpha(obs_fn, actor_params[name], kw_names, a_obs, tk)
        log_odds_mse = float(_log_odds_mse_at_table(
            table, np.asarray(action), np.asarray(scenario_idx),
            np.asarray(intim_cond), data
        ))

        rows.append({
            "model": name,
            "alpha_obs_rating_fit": a_obs,
            "rating_mse": rating_mse,
            "log_odds_mse_at_rating_alpha": log_odds_mse,
            "alpha_obs_logodds_fit (paper)": float(paper.loc[name, "alpha_observer"]),
            "log_odds_mse (paper)": float(paper.loc[name, "mse"]),
        })

        # Per-cell summary for the full model (the focus of the question).
        summaries[name] = cell_means(
            data, table, np.asarray(action),
            np.asarray(scenario_idx), np.asarray(intim_cond),
        )

    print("\n" + "=" * 60)
    print("FIT COMPARISON (per-trial MSE; lower is better)")
    print("=" * 60)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n" + "=" * 60)
    print("FULL MODEL — cell-mean belief updates by (action, intimacy)")
    print("rating scale, posterior - prior in [-1, 1]")
    print("=" * 60)
    full = summaries["full"].copy()
    full["model_bu"] = full["model_bu"].round(3)
    full["human_bu"] = full["human_bu"].round(3)
    full["diff"] = (full["model_bu"] - full["human_bu"]).round(3)
    print(full.to_string(index=False))


def _log_odds_mse_at_table(table, action_arr, scenario_arr, intim_arr, data):
    """Compute log-odds-scale MSE at a given observer table.

    Reads the human's logit_belief_update column (computed the same way the
    paper's fit does). If not present in `data`, recompute via _attach_logit_belief_update.
    """
    # We have rating-scale belief_update on `data`; compute log-odds version
    # the same way the paper does.
    eps = 1e-3
    pivot = (
        pd.read_csv(
            get_project_root() / "data" / EXPERIMENT_SLUG / "main_trials_long.csv"
        )
        .pivot_table(
            index=["subject_id", "scenario_label", "action_condition"],
            columns="stage", values="response", aggfunc="first",
        )
        .reset_index()
    )
    p_prior = (pivot["prior"] / 100.0).clip(eps, 1 - eps)
    p_post = (pivot["posterior"] / 100.0).clip(eps, 1 - eps)
    pivot["logit_bu"] = (
        np.log(p_post / (1 - p_post)) - np.log(p_prior / (1 - p_prior))
    )
    merged = data.merge(
        pivot[["subject_id", "scenario_label", "action_condition", "logit_bu"]],
        on=["subject_id", "scenario_label", "action_condition"], how="left",
    )

    eps2 = 1e-6
    sse = 0.0
    n = len(merged)
    for i in range(n):
        p = float(table[
            int(action_arr[i]), int(scenario_arr[i]), int(intim_arr[i]), 1
        ])
        p = min(max(p, eps2), 1 - eps2)
        model_logit = np.log(p / (1 - p))
        sse += (model_logit - float(merged["logit_bu"].iloc[i])) ** 2
    return sse / n


if __name__ == "__main__":
    main()
