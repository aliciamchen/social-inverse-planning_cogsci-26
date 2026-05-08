"""
Constants, enums, scenario maps, and stipulated table accessors shared across
all model code.

V, access, and effort come from stipulated_tables.py (hand-set action-only
values broadcast across all 16 scenarios).

Dependency layer 0: imports nothing from sibling modules. `utility.py`,
`actors.py`, and `observers.py` all import from here.
"""

from enum import IntEnum

import jax.numpy as jnp

from stipulated_tables import ACCESS_TABLE, EFFORT_TABLE, V_TABLE

# ==============================================================================
# Action and intimacy axes
# ==============================================================================

actions = jnp.array([0, 1, 2, 3])
IntimacyLevels = jnp.arange(0, 1.01, 0.01)


class RewardConditions(IntEnum):
    LOW = 0
    HIGH = 1


class RelationshipConditions(IntEnum):
    ZERO = 0
    FIFTY = 1
    SEVENTY_FIVE = 2
    ONE_HUNDRED = 3


# ==============================================================================
# Scenario labels
# ==============================================================================
SCENARIO_LABELS = [
    "apples",
    "basketball",
    "birthday",
    "brunch",
    "cooking",
    "dip",
    "drinks",
    "driving",
    "fair",
    "gala",
    "hike",
    "oysters",
    "social",
    "soup",
    "takeout",
    "wedding",
]
SCENARIO_TO_IDX = {label: idx for idx, label in enumerate(SCENARIO_LABELS)}


class Scenarios(IntEnum):
    """Memo-friendly enum of the 16 scenarios (alphabetical order)."""
    APPLES = 0
    BASKETBALL = 1
    BIRTHDAY = 2
    BRUNCH = 3
    COOKING = 4
    DIP = 5
    DRINKS = 6
    DRIVING = 7
    FAIR = 8
    GALA = 9
    HIKE = 10
    OYSTERS = 11
    SOCIAL = 12
    SOUP = 13
    TAKEOUT = 14
    WEDDING = 15


# ==============================================================================
# Stipulated table accessors
# ==============================================================================

LLM_TABLES = {
    "access": ACCESS_TABLE,
    "effort": EFFORT_TABLE,
}


def load_lm_v(domain: str = "food"):
    """Return the stipulated signed-valence table.

    Shape is (16, 4, 2) — `V_TABLE[scenario_idx, action, reward_condition]`,
    values in [-1, +1].
    """
    if domain != "food":
        raise ValueError(f"only the food domain is supported; got {domain!r}")
    return V_TABLE


def load_domain_assets(domain: str = "food"):
    """Return (scenario_labels, scenario_to_idx, llm_tables)."""
    if domain != "food":
        raise ValueError(f"only the food domain is supported; got {domain!r}")
    return SCENARIO_LABELS, SCENARIO_TO_IDX, LLM_TABLES
