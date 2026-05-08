"""
Test that the cogsci-cr model implementation matches the canonical
access-utility spec used in the camera-ready paper.

Run standalone: uv run python model/test_model_compliance.py
"""

import jax.numpy as jnp
import numpy as np

from actors import (
    actor_forw_base,
    actor_forw_discomfort_only,
    actor_forw_full,
)
from observers import observer_intimacy_full
from tables import LLM_TABLES, RewardConditions, load_lm_v
from utility import (
    get_utility_base,
    get_utility_discomfort_only,
    get_utility_full,
)


_V_TABLE_FOOD = load_lm_v("food")


def _tables():
    return (LLM_TABLES["access"], LLM_TABLES["effort"])


def _v_table():
    return _V_TABLE_FOOD


# ==============================================================================
# Stipulated tables: shape and range
# ==============================================================================


def test_stipulated_tables_shape_and_range():
    access = LLM_TABLES["access"]
    effort = LLM_TABLES["effort"]
    v = _v_table()

    assert access.shape == (16, 4), f"access shape {access.shape} != (16, 4)"
    assert effort.shape == (16, 4), f"effort shape {effort.shape} != (16, 4)"
    assert v.shape == (16, 4, 2), f"v shape {v.shape} != (16, 4, 2)"

    assert 0.0 <= float(access.min()) and float(access.max()) <= 2.01, \
        f"access range [{float(access.min()):.3f}, {float(access.max()):.3f}] ⊄ [0, 2]"
    assert 0.0 <= float(effort.min()) and float(effort.max()) <= 1.01, \
        f"effort range [{float(effort.min()):.3f}, {float(effort.max()):.3f}] ⊄ [0, 1]"
    assert -1.01 <= float(v.min()) and float(v.max()) <= 1.01, \
        f"v range [{float(v.min()):.3f}, {float(v.max()):.3f}] ⊄ [-1, 1]"

    print("✓ Stipulated tables shapes and ranges match spec")


# ==============================================================================
# Utility ablation algebra
# ==============================================================================


def test_full_collapses_to_restricted_variants():
    """full utility reduces to base (w_d=0) and to discomfort_only (w_v=w_e=0)."""
    action = jnp.array(2)
    scenario_idx = jnp.array(0)
    intimacy = 0.4
    rc = RewardConditions.HIGH
    alpha, w_v, w_d, w_e, gamma = 1.0, 1.0, 0.7, 0.5, 1.0
    a_tab, e_tab = _tables()
    v_tab = _v_table()

    u_full_base = float(get_utility_full(
        action, scenario_idx, intimacy, rc,
        alpha, w_v, 0.0, w_e, gamma,
        a_tab, e_tab, v_tab,
    ))
    u_none = float(get_utility_base(
        action, scenario_idx, intimacy, rc,
        alpha, w_v, w_e,
        a_tab, e_tab, v_tab,
    ))
    assert abs(u_full_base - u_none) < 1e-6, \
        f"full(w_d=0) should match base: {u_full_base} vs {u_none}"

    u_full_discomfort_only = float(get_utility_full(
        action, scenario_idx, intimacy, rc,
        alpha, 0.0, w_d, 0.0, gamma,
        a_tab, e_tab, v_tab,
    ))
    u_only = float(get_utility_discomfort_only(
        action, scenario_idx, intimacy, rc,
        alpha, w_d, gamma,
        a_tab, e_tab,
    ))
    assert abs(u_full_discomfort_only - u_only) < 1e-6, \
        f"full(w_v=w_e=0) should match discomfort_only: {u_full_discomfort_only} vs {u_only}"

    print("✓ full collapses to base (w_d=0) and discomfort_only (w_v=w_e=0)")


def test_gamma_unity_recovers_linear_intimacy():
    """With γ=1, get_utility_full reproduces w_v·V - w_d·access·(1-I) - w_e·effort."""
    action = jnp.array(1)
    scenario_idx = jnp.array(3)
    intimacy = 0.6
    rc = RewardConditions.HIGH
    alpha, w_v, w_d, w_e, gamma = 1.0, 1.2, 0.9, 0.4, 1.0
    a_tab, e_tab = _tables()
    v_tab = _v_table()

    u_gamma = float(get_utility_full(
        action, scenario_idx, intimacy, rc,
        alpha, w_v, w_d, w_e, gamma,
        a_tab, e_tab, v_tab,
    ))

    access = float(a_tab[scenario_idx, action])
    effort = float(e_tab[scenario_idx, action])
    V = float(v_tab[scenario_idx, action, int(rc)])
    u_linear = alpha * (w_v * V - w_d * access * (1.0 - intimacy) - w_e * effort)

    assert abs(u_gamma - u_linear) < 1e-5, \
        f"γ=1 should reproduce linear (1-I): {u_gamma} vs {u_linear}"
    print("✓ γ=1 reproduces canonical linear-intimacy utility")


def test_gamma_not_unity_changes_predictions():
    """γ ≠ 1 should produce measurably different actor probabilities."""
    a_tab, e_tab = _tables()
    probs_g1 = actor_forw_full(1.0, 1.0, 1.0, 1.0, 1.0, a_tab, e_tab, _v_table())
    probs_g2 = actor_forw_full(1.0, 1.0, 1.0, 1.0, 2.0, a_tab, e_tab, _v_table())
    diff = float(jnp.abs(probs_g1 - probs_g2).max())
    assert diff > 0.01, f"γ=2 should differ from γ=1, max abs diff was {diff}"
    print(f"✓ γ=2 vs γ=1 produces different predictions (max abs diff {diff:.3f})")


# ==============================================================================
# Probability distribution validity
# ==============================================================================


def test_actor_probabilities_sum_to_one():
    a_tab, e_tab = _tables()
    probs = actor_forw_full(1.0, 1.0, 1.0, 1.0, 1.0, a_tab, e_tab, _v_table())
    for s in [0, 8, 15]:
        for i_idx in [0, 50, 100]:
            for r in [0, 1]:
                prob_sum = float(probs[:, s, i_idx, r].sum())
                assert np.isclose(prob_sum, 1.0, atol=1e-5), \
                    f"actor_forw_full probs don't sum to 1: {prob_sum} (s={s}, i={i_idx}, r={r})"
    print("✓ actor_forw_full probabilities sum to 1")


def test_observer_posterior_sums_to_one():
    a_tab, e_tab = _tables()
    result = observer_intimacy_full(
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, a_tab, e_tab, _v_table(),
    )
    for s in [0, 8, 15]:
        for a in range(4):
            for r in [0, 1]:
                posterior_sum = float(result[a, s, :, r].sum())
                assert np.isclose(posterior_sum, 1.0, atol=1e-4), \
                    f"observer posterior doesn't sum to 1: {posterior_sum} (a={a}, s={s}, r={r})"
    print("✓ observer_intimacy_full posteriors sum to 1")


# ==============================================================================
# Qualitative predictions
# ==============================================================================


def test_motivation_increases_sharing():
    """HIGH motivation should raise P(action 1-3) above LOW motivation under full."""
    a_tab, e_tab = _tables()
    probs = actor_forw_full(1.0, 1.0, 1.0, 1.0, 1.0, a_tab, e_tab, _v_table())
    for s in [0, 8, 15]:
        for i_idx in [50, 75, 100]:
            p_share_low = float(probs[1:, s, i_idx, 0].sum())
            p_share_high = float(probs[1:, s, i_idx, 1].sum())
            assert p_share_high > p_share_low, (
                f"HIGH motivation should raise sharing (s={s}, i={i_idx}): "
                f"low={p_share_low:.3f}, high={p_share_high:.3f}"
            )
    print("✓ HIGH motivation increases sharing under full")


def test_intimacy_increases_risky_actions():
    """Higher intimacy should raise P(action 2-3) under full, high motivation."""
    a_tab, e_tab = _tables()
    probs = actor_forw_full(1.0, 1.0, 1.0, 1.0, 1.0, a_tab, e_tab, _v_table())
    for s in [0, 8, 15]:
        p_risky_low = float(probs[2:, s, 0, 1].sum())
        p_risky_high = float(probs[2:, s, 100, 1].sum())
        assert p_risky_high > p_risky_low, (
            f"Higher intimacy should raise risky actions (s={s}): "
            f"low={p_risky_low:.3f}, high={p_risky_high:.3f}"
        )
    print("✓ Higher intimacy raises P(action 2-3) under full")


def test_base_actor_invariant_to_intimacy():
    """base utility omits access terms, so the actor policy cannot depend on I."""
    a_tab, e_tab = _tables()
    probs = actor_forw_base(1.0, 1.0, 1.0, a_tab, e_tab, _v_table())
    for s in [0, 8, 15]:
        low_int = np.asarray(probs[:, s, 0, 1])
        mid_int = np.asarray(probs[:, s, 50, 1])
        high_int = np.asarray(probs[:, s, 100, 1])
        np.testing.assert_allclose(low_int, mid_int, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(low_int, high_int, rtol=1e-6, atol=1e-6)
    print("✓ base policy is invariant to intimacy across scenarios")


def test_discomfort_only_actor_invariant_to_motivation():
    """discomfort_only utility omits V, so the actor policy cannot depend on motivation."""
    a_tab, e_tab = _tables()
    probs = actor_forw_discomfort_only(1.0, 1.0, 1.0, a_tab, e_tab)
    for s in [0, 8, 15]:
        for i_idx in [0, 50, 100]:
            low_mot = np.asarray(probs[:, s, i_idx, 0])
            high_mot = np.asarray(probs[:, s, i_idx, 1])
            np.testing.assert_allclose(low_mot, high_mot, rtol=1e-6, atol=1e-6)
    print("✓ discomfort_only policy is invariant to motivation")


# ==============================================================================
# Main
# ==============================================================================


def run_all_tests():
    print("=" * 60)
    print("cogsci-cr model compliance tests")
    print("=" * 60)

    print("\n--- Stipulated tables ---")
    test_stipulated_tables_shape_and_range()

    print("\n--- Utility ablation algebra ---")
    test_full_collapses_to_restricted_variants()
    test_gamma_unity_recovers_linear_intimacy()
    test_gamma_not_unity_changes_predictions()

    print("\n--- Probability validity ---")
    test_actor_probabilities_sum_to_one()
    test_observer_posterior_sums_to_one()

    print("\n--- Qualitative predictions ---")
    test_motivation_increases_sharing()
    test_intimacy_increases_risky_actions()
    test_base_actor_invariant_to_intimacy()
    test_discomfort_only_actor_invariant_to_motivation()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
