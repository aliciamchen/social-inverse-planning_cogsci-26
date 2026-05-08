"""
Pure utility functions (jax-jit-compiled) used by all actor and observer memo
models in the cogsci-cr camera-ready subset.

Three model variants — Full, Discomfort-only, Base — each have:
  - a canonical form: `get_utility_<variant>(action, scenario_idx, intimacy, ...)`
  - a discrete-relationship form (`_disc`): wraps the canonical form, mapping
    a discrete relationship_condition to a continuous intimacy via get_intimacy

Canonical utility:
  U(a|s, I, m) =  w_v * V(a|s, m)
                - w_d * access[s, a] * (1 - I)^gamma
                - w_e * effort[s, a]

V is signed in [-1, +1]: positive when an action serves the active state,
negative when it actively works against it. With the cogsci-cr stipulated
tables, V/access/effort are action-only (broadcast across scenarios).

Dependency layer 1: imports nothing from sibling modules. `actors.py` and
`observers.py` import from here.
"""

import jax
import jax.numpy as jnp


# ==============================================================================
# Basic helpers
# ==============================================================================


@jax.jit
def get_intimacy(relationship_condition):
    """Map a discrete relationship condition to a continuous intimacy in [0, 1]."""
    return jnp.array([0, 0.5, 0.75, 1])[relationship_condition]


@jax.jit
def get_lm_v(action, scenario_idx, reward_condition, v_table):
    """Stipulated signed valence: v_table[scenario_idx, action, reward_condition].

    v_table has shape (16, 4, 2). Values in [-1, +1]. Positive = action serves
    the active state; negative = action actively counterproductive.
    """
    return v_table[scenario_idx, action, reward_condition]


# ==============================================================================
# Canonical (4-action) utility — Full / Discomfort-only / Base
# ==============================================================================


@jax.jit
def get_utility_full(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    access = access_table[scenario_idx, action]
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (
        w_v * V
        - w_d * access * jnp.power(one_minus_I, gamma)
        - w_e * effort
    )


@jax.jit
def get_utility_full_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_d, w_e, gamma,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_full(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_d, w_e, gamma,
        access_table, effort_table, v_table,
    )


@jax.jit
def get_utility_discomfort_only(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    access = access_table[scenario_idx, action]
    one_minus_I = jnp.maximum(1.0 - intimacy, 1e-8)
    return alpha * (-w_d * access * jnp.power(one_minus_I, gamma))


@jax.jit
def get_utility_discomfort_only_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_d, gamma,
    access_table, effort_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_discomfort_only(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_d, gamma,
        access_table, effort_table,
    )


@jax.jit
def get_utility_base(
    action, scenario_idx, intimacy, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    effort = effort_table[scenario_idx, action]
    V = get_lm_v(action, scenario_idx, reward_condition, v_table)
    return alpha * (w_v * V - w_e * effort)


@jax.jit
def get_utility_base_disc(
    action, scenario_idx, relationship_condition, reward_condition,
    alpha, w_v, w_e,
    access_table, effort_table, v_table,
):
    intimacy = get_intimacy(relationship_condition)
    return get_utility_base(
        action, scenario_idx, intimacy, reward_condition,
        alpha, w_v, w_e,
        access_table, effort_table, v_table,
    )
