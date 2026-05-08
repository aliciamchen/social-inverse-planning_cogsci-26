"""
Actor memo models — both forward (`actor_forw_*`) and inverse
(`actor_discrete_*`, `actor_continuous_*`) variants used inside observer
`thinks[...]` blocks.

Three model variants per shape: `_full`, `_discomfort_only`, `_base`.

Dependency layer 2: imports from `tables.py` (enums, axes) and `utility.py`
(get_utility_*). `observers.py` imports from here.
"""

from memo import memo

from tables import (
    IntimacyLevels,
    RelationshipConditions,
    RewardConditions,
    Scenarios,
    actions,
)
from utility import (
    get_utility_base,
    get_utility_base_disc,
    get_utility_discomfort_only,
    get_utility_discomfort_only_disc,
    get_utility_full,
    get_utility_full_disc,
)


# ==============================================================================
# Forward-planning actors (canonical 4-action, continuous intimacy)
# ==============================================================================


@memo
def actor_forw_full[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_forw_base[
    action: actions,
    scenario_idx: Scenarios,
    intimacy: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(intimacy)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action, scenario_idx, intimacy, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actors — discrete relationship (observer infers reward)
# ==============================================================================


@memo
def actor_discrete_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_discrete_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship_condition: RelationshipConditions,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship_condition)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base_disc(
                action, scenario_idx, relationship_condition, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


# ==============================================================================
# Inverse-planning actors — continuous intimacy (observer infers intimacy)
# ==============================================================================


@memo
def actor_continuous_full[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_d, w_e, gamma, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_full(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_d, w_e, gamma,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_discomfort_only[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_d, gamma, access_table: ..., effort_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_discomfort_only(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_d, gamma,
                access_table, effort_table,
            )
        ),
    )
    return Pr[actor.action == action]


@memo
def actor_continuous_base[
    action: actions,
    scenario_idx: Scenarios,
    relationship: IntimacyLevels,
    reward_condition: RewardConditions,
](alpha, w_v, w_e, access_table: ..., effort_table: ..., v_table: ...):
    cast: [actor]
    actor: knows(scenario_idx)
    actor: knows(relationship)
    actor: knows(reward_condition)
    actor: chooses(
        action in actions,
        wpp=exp(
            get_utility_base(
                action, scenario_idx, relationship, reward_condition,
                alpha, w_v, w_e,
                access_table, effort_table, v_table,
            )
        ),
    )
    return Pr[actor.action == action]
