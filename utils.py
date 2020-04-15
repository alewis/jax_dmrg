"""
Miscellanious utilities for finite DMRG.
"""
import jax
import jax.numpy as jnp


def paulis():
    sX = jnp.array([[0, 1], [1, 0]])
    sY = jnp.array([[0, -1j], [1j, 0]])
    sZ = jnp.array([[1, 0], [0, -1]])
    return (sX, sY, sZ)
