"""Environment-specific termination functions for model-based rollouts.

Ported verbatim from Unifloral (algorithms/termination_fns.py), adapted to use
Minari dataset IDs for dispatch rather than D4RL task strings.
"""

from collections.abc import Callable

import jax.numpy as jnp


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    not_done = jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
    done = ~not_done
    return done


def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    height = next_obs[0]
    angle = next_obs[1]
    not_done = (
        jnp.isfinite(next_obs).all()
        * (jnp.abs(next_obs[1:]) < 100).all()  # fix: Unifloral has wrong precedence here
        * (height > 0.7)
        * (jnp.abs(angle) < 0.2)
    )
    done = ~not_done
    return done


def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1

    height = next_obs[0]
    angle = next_obs[1]
    not_done = (
        jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
        * (height > 0.8)
        * (height < 2.0)
        * (angle > -1.0)
        * (angle < 1.0)
    )
    done = ~not_done
    return done


def get_termination_fn(dataset_id: str) -> Callable:
    """Return the termination function for the given Minari dataset ID."""
    if "halfcheetah" in dataset_id:
        return termination_fn_halfcheetah
    elif "hopper" in dataset_id:
        return termination_fn_hopper
    elif "walker2d" in dataset_id:
        return termination_fn_walker2d
    else:
        raise ValueError(f"No termination function registered for dataset: {dataset_id}")
