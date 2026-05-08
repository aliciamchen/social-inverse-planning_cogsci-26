"""
Hand-stipulated, action-only V (desire), access (risk), and effort (cost)
tables for the model. Values are action-only and broadcast across all 16
scenarios into the (16, 4) and (16, 4, 2) shapes the actor and observer
code expects.

  V_low  = [0, -1, -1, -1]   # under low desire, sharing actions are
                             # actively counterproductive
  V_high = [0,  1,  1,  1]   # under high desire, sharing actions serve the
                             # active goal
  access = [0,  0,  1,  2]   # saliva-transfer risk increases with action idx
  effort = [0,  1,  1,  1]   # action 0 (no share) is free; the 3 share
                             # actions all cost equal effort
"""

import jax.numpy as jnp


N_SCENARIOS = 16
N_ACTIONS = 4

V_BY_ACTION_LOW = jnp.array([0.0, -1.0, -1.0, -1.0])
V_BY_ACTION_HIGH = jnp.array([0.0, 1.0, 1.0, 1.0])
ACCESS_BY_ACTION = jnp.array([0.0, 0.0, 1.0, 2.0])
EFFORT_BY_ACTION = jnp.array([0.0, 1.0, 1.0, 1.0])

ACCESS_TABLE = jnp.broadcast_to(ACCESS_BY_ACTION, (N_SCENARIOS, N_ACTIONS))
EFFORT_TABLE = jnp.broadcast_to(EFFORT_BY_ACTION, (N_SCENARIOS, N_ACTIONS))

# motivation index 0 = LOW, 1 = HIGH (matches RewardConditions in tables.py)
V_TABLE = jnp.stack(
    [
        jnp.broadcast_to(V_BY_ACTION_LOW, (N_SCENARIOS, N_ACTIONS)),
        jnp.broadcast_to(V_BY_ACTION_HIGH, (N_SCENARIOS, N_ACTIONS)),
    ],
    axis=-1,
)
