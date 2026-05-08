"""
Regenerate cogsci-2026/values.tex from the latest fit results and
demographics. Run after `make fits predicts cv` (or as part of `make all`)
to keep the paper's participant counts, fitted parameters, AIC values, and
r-values in sync with the model outputs.

Usage:
    uv run python scripts/generate_values_tex.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # cogsci-cr/
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "model" / "outputs"
OUTPUT_FILE = PROJECT_ROOT / "cogsci-2026" / "values.tex"

# Per-experiment slugs
FORWARD_SLUG = "food_forw_intimacy_desire"
INTIMACY_INFERENCE_SLUG = "food_inv_intimacy_desire_alt"
DESIRE_INFERENCE_SLUG = "food_inv_desire_intimacy_alt"

# Paper-level constants (not derivable from data)
PAY_EXP_ONE = "6.25"
TIME_EXP_ONE = "25"
PAY_EXP_TWO = "5"
TIME_EXP_TWO = "20"
N_SCENARIOS = 16

# Display decimals
PARAM_DECIMALS = 2
AIC_DECIMALS = 2
MSE_DECIMALS = 3
R_DECIMALS = 2
ALPHA_DECIMALS = 2
AGE_DECIMALS = 1


def fmt(value, decimals: int) -> str:
    """Format a float with fixed decimals; integer 0 stays as '0'."""
    if value == 0:
        return "0"
    return f"{value:.{decimals}f}"


def load_demographics(slug: str) -> dict:
    """Read exit_survey.csv and compute counts + age stats.

    Recruited = all rows in exit_survey.csv (it includes everyone collected).
    Excluded = those who failed attention check OR scored 0 on memory check.
    """
    df = pd.read_csv(DATA_DIR / slug / "exit_survey.csv")
    n_recruited = len(df)
    passed = df[(df["attention_passed"] == True) & (df["memory_correct_count"] > 0)]
    n_excluded = n_recruited - len(passed)

    counts = df["gender"].value_counts().to_dict()
    n_female = int(counts.get("female", 0))
    n_male = int(counts.get("male", 0))
    # Survey uses "nonconforming"; accept synonyms for robustness against
    # future surveys that may use a different label.
    n_nonbinary = sum(
        int(counts.get(k, 0))
        for k in ("nonconforming", "non-binary", "nonbinary")
    )
    n_abstain = int(counts.get("abstain", 0))

    return {
        "n_recruited": n_recruited,
        "n_excluded": n_excluded,
        "n_female": n_female,
        "n_male": n_male,
        "n_nonbinary": n_nonbinary,
        "n_abstain": n_abstain,
        "age_min": int(df["age"].min()),
        "age_max": int(df["age"].max()),
        "age_mean": round(float(df["age"].mean()), AGE_DECIMALS),
        "age_sd": round(float(df["age"].std()), AGE_DECIMALS),
    }


def load_fits(slug: str) -> pd.DataFrame:
    """Load fit_results.csv for an experiment."""
    return pd.read_csv(OUTPUTS_DIR / slug / "fit_results.csv")


def cv_pooled_r(slug: str, group_cols, n_boot: int = 1000, seed: int = 42) -> dict:
    """Compute pooled out-of-sample Pearson r and bootstrap 95% CI per variant
    from cv_preds.csv. Pools cell means across all CV folds.

    Also writes the per-variant r/CI to `<slug>/cv_correlation.csv` so the R
    figure code can read the same values, avoiding cross-language bootstrap
    drift between the table and the plot labels.

    Returns a dict: variant -> {"r": float, "ci_lower": float, "ci_upper": float}.
    """
    df = pd.read_csv(OUTPUTS_DIR / slug / "cv_preds.csv")
    rng = np.random.default_rng(seed)
    out = {}
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        cells = (
            sub.groupby(group_cols)
            .agg(human=("p_action", "mean"), model=("p_action_pred", "mean"))
            .reset_index()
        )
        if len(cells) < 3 or cells["model"].std() == 0:
            out[variant] = {"r": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
            continue
        r, _ = stats.pearsonr(cells["model"], cells["human"])
        boot_rs = []
        n = len(cells)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_r, _ = stats.pearsonr(
                cells["model"].iloc[idx].values,
                cells["human"].iloc[idx].values,
            )
            boot_rs.append(boot_r)
        out[variant] = {
            "r": float(r),
            "ci_lower": float(np.percentile(boot_rs, 2.5)),
            "ci_upper": float(np.percentile(boot_rs, 97.5)),
        }
    # Write the same numbers the LaTeX table will use to a CSV the R plot can read.
    pd.DataFrame([
        {"variant": v, "r": out[v]["r"],
         "ci_lower": out[v]["ci_lower"], "ci_upper": out[v]["ci_upper"]}
        for v in out
    ]).to_csv(OUTPUTS_DIR / slug / "cv_correlation.csv", index=False)
    return out


def with_delta_aic(fits: pd.DataFrame, aic_col: str = "aic") -> pd.DataFrame:
    """Add `delta_aic` column = aic - aic[full], so the Full model is the
    reference (delta_aic = 0) and alternatives have positive ΔAIC (worse)."""
    out = fits.copy()
    full_aic = float(out.loc[out["model"] == "full", aic_col].iloc[0])
    out["delta_aic"] = out[aic_col] - full_aic
    return out


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def section(header: str) -> list:
    return ["", f"%% --- {header} ---"]


def build_demographics_block(
    label: str, suffix: str, demo: dict, include_abstain: bool
) -> list:
    """Emit demographics macros for one experiment.

    `suffix` is e.g. "ExpOne", "ExpTwoA", "ExpTwoB" — appended to each macro name.
    `include_abstain` controls whether the NAbstain macro is emitted (Exp 2a only).
    """
    lines = section(f"Participants: {label}")
    lines += [
        macro(f"NRecruited{suffix}", demo["n_recruited"]),
        macro(f"NFemale{suffix}", demo["n_female"]),
        macro(f"NMale{suffix}", demo["n_male"]),
        macro(f"NNonbinary{suffix}", demo["n_nonbinary"]),
    ]
    if include_abstain:
        lines.append(macro(f"NAbstain{suffix}", demo["n_abstain"]))
    lines += [
        macro(f"AgeMin{suffix}", demo["age_min"]),
        macro(f"AgeMax{suffix}", demo["age_max"]),
        macro(f"AgeMean{suffix}", fmt(demo["age_mean"], AGE_DECIMALS)),
        macro(f"AgeSD{suffix}", fmt(demo["age_sd"], AGE_DECIMALS)),
        macro(f"NExcluded{suffix}", demo["n_excluded"]),
    ]
    return lines


def build_forward_block(suffix: str, fits: pd.DataFrame, cv_r: dict) -> list:
    """Emit forward-fit macros: full-model params, AIC/Delta AIC, r/CI per ablation.

    `suffix` is "One" (Experiment 1). `cv_r` is the pooled CV r dict from
    `cv_pooled_r()`, keyed by variant.
    """
    by_model = {row["model"]: row for _, row in fits.iterrows()}
    full = by_model["full"]

    lines = section("Forward (Exp 1) full-model fitted parameters")
    lines += [
        macro("FullWv", fmt(float(full["param_w_v"]), PARAM_DECIMALS)),
        macro("FullWd", fmt(float(full["param_w_d"]), PARAM_DECIMALS)),
        macro("FullWe", fmt(float(full["param_w_e"]), PARAM_DECIMALS)),
        macro("FullGamma", fmt(float(full["param_gamma"]), PARAM_DECIMALS)),
    ]

    lines += section(f"Forward (Exp {suffix}) AIC and Delta AIC (in-sample)")
    for model_key, prefix in [
        ("base", "Base"),
        ("discomfort_only", "Discomfort"),
        ("full", "Full"),
    ]:
        row = by_model[model_key]
        lines.append(
            macro(f"{prefix}AIC{suffix}", fmt(float(row["aic"]), AIC_DECIMALS))
        )
        lines.append(
            macro(
                f"{prefix}DeltaAIC{suffix}", fmt(float(row["delta_aic"]), AIC_DECIMALS)
            )
        )

    lines += section(
        f"Forward (Exp {suffix}) pooled out-of-sample Pearson r and 95% bootstrap CI (LOSO CV)"
    )
    for model_key, prefix in [
        ("base", "Base"),
        ("discomfort_only", "Discomfort"),
        ("full", "Full"),
    ]:
        r = cv_r[model_key]
        lines.append(macro(f"{prefix}R{suffix}", fmt(r["r"], R_DECIMALS)))
        lines.append(macro(f"{prefix}RCILo{suffix}", fmt(r["ci_lower"], R_DECIMALS)))
        lines.append(macro(f"{prefix}RCIHi{suffix}", fmt(r["ci_upper"], R_DECIMALS)))

    return lines


def cv_pooled_inverse_mse(slug: str) -> dict:
    """Compute pooled out-of-sample MSE from cv_folds.csv per variant.

    Pools test SSE across the 16 LOSO folds: pooled_mse = sum(test_sse) / sum(n_test).
    Also returns the mean per-fold alpha_observer (the values that actually
    generate the held-out predictions in cv_preds_summary.csv / the figures).

    Returns dict: variant -> {"mse": float, "alpha_obs_mean": float}.
    """
    df = pd.read_csv(OUTPUTS_DIR / slug / "cv_folds.csv")
    out = {}
    for variant, g in df.groupby("variant"):
        out[variant] = {
            "mse": float(g["test_sse"].sum() / g["n_test"].sum()),
            "alpha_obs_mean": float(g["alpha_observer"].mean()),
        }
    return out


def build_inverse_block(label: str, suffix: str, cv_inverse: dict) -> list:
    """Emit inverse-fit macros: alpha_obs (mean across LOSO folds), pooled
    out-of-sample MSE, and ΔMSE vs. Base.

    `suffix` is "TwoA" or "TwoB". `cv_inverse` is the dict from `cv_pooled_inverse_mse`.
    ΔMSE is computed relative to the Full model (Full = 0; alternatives positive).
    """
    full_mse = cv_inverse["full"]["mse"]
    lines = section(
        f"Inverse (Exp {suffix}, {label}): mean LOSO alpha_obs, pooled out-of-sample MSE, and ΔMSE vs. Full"
    )
    for model_key, prefix in [
        ("base", "Base"),
        ("discomfort_only", "Discomfort"),
        ("full", "Full"),
    ]:
        entry = cv_inverse[model_key]
        mse = entry["mse"]
        lines.append(
            macro(
                f"{prefix}AlphaObs{suffix}",
                fmt(entry["alpha_obs_mean"], ALPHA_DECIMALS),
            )
        )
        lines.append(macro(f"{prefix}MSE{suffix}", fmt(mse, MSE_DECIMALS)))
        lines.append(macro(f"{prefix}DeltaMSE{suffix}", fmt(mse - full_mse, MSE_DECIMALS)))
    return lines


def main():
    # Demographics
    demo_one = load_demographics(FORWARD_SLUG)
    demo_two_a = load_demographics(INTIMACY_INFERENCE_SLUG)
    demo_two_b = load_demographics(DESIRE_INFERENCE_SLUG)

    # Forward fits + delta AIC; CV r from cv_preds.csv
    fwd_fits = with_delta_aic(load_fits(FORWARD_SLUG))
    fwd_cv_r = cv_pooled_r(
        FORWARD_SLUG, group_cols=["intimacy", "motivation", "action"]
    )

    # Inverse fits: pooled out-of-sample MSE from cv_folds.csv (LOSO over the
    # 16 scenarios, refitting alpha_obs per fold; actor params frozen at the
    # all-data forward fit). Loss is per-trial squared error on rating-scale
    # belief updates (posterior - prior) for both inference targets.
    inv_a_cv = cv_pooled_inverse_mse(INTIMACY_INFERENCE_SLUG)
    inv_b_cv = cv_pooled_inverse_mse(DESIRE_INFERENCE_SLUG)

    lines = [
        "% =============================================================================",
        "% Auto-generated by scripts/generate_values_tex.py — do not edit by hand.",
        "% Re-run after `make fits cv` to refresh from the latest model outputs.",
        "% =============================================================================",
    ]

    lines += build_demographics_block(
        "Experiment 1 (forward planning)", "ExpOne", demo_one, include_abstain=False
    )
    lines += [
        macro("PayExpOne", PAY_EXP_ONE),
        macro("TimeExpOne", TIME_EXP_ONE),
    ]

    lines += build_demographics_block(
        "Experiment 2a (intimacy inference)",
        "ExpTwoA",
        demo_two_a,
        include_abstain=True,
    )
    lines += build_demographics_block(
        "Experiment 2b (desire inference)", "ExpTwoB", demo_two_b, include_abstain=False
    )

    lines += section("Pay/time: Experiment 2 (both 2a and 2b)")
    lines += [
        macro("PayExpTwo", PAY_EXP_TWO),
        macro("TimeExpTwo", TIME_EXP_TWO),
    ]

    lines += build_forward_block("One", fwd_fits, fwd_cv_r)
    lines += build_inverse_block("intimacy inference", "TwoA", inv_a_cv)
    lines += build_inverse_block("desire inference", "TwoB", inv_b_cv)

    lines += section("Model display names (rename in one place if desired)")
    lines += [
        macro("BaseModelName", "Base"),
        macro("DiscomfortModelName", "Discomfort-only"),
        macro("FullModelName", "Full"),
    ]

    lines += section("Design constants")
    lines.append(macro("NScenarios", N_SCENARIOS))

    OUTPUT_FILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
