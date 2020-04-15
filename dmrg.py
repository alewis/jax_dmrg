"""
Jax implementation of single-site DMRG.

Adam GM Lewis
"""

import numpy as np
import jax.numpy as jnp

import jax_dmrg.errors as errors
import jax_dmrg.operations as op
import jax_dmrg.lz as lz


def left_to_right(mps_chain, H_block, mpo_chain, lz_params,
                  initialization=False):
    N = len(mpo_chain)
    Es = np.zeros(N)
    for n in range(N-1):
        mps = mps_chain[n]
        mpo = mpo_chain[n]
        L = H_block[n]
        if not initialization:
            R = H_block[n+1]
            E, mps = lz.dmrg_solve(mps, L, R, mpo, lz_params)
            Es[n] = E
        mps_L, C = op.qrpos(mps)
        mps_chain[n+1] = op.leftcontract(C, mps_chain[n+1])
        H_block[n] = op.XopL(L, mpo, mps_L)
    mps_L, C = op.qrpos(mps_chain[-1])
    mps_chain[-1] = mps_L
    H_block[-1] = op.joinR(C, H_block[-1])
    return (Es, mps_chain, H_block)


def right_to_left(mps_chain, H_block, mpo_chain, lz_params):
    N = len(mpo_chain)
    Es = np.zeros(N)
    for n in range(N-1, -1, -1):
        mps = mps_chain[n]
        L = H_block[n]
        R = H_block[n+1]
        mpo = mpo_chain[n]
        E, mps = lz.dmrg_solve(mps, L, R, mpo, lz_params)
        Es[n] = E
        C, mps_R = op.rqpos(mps)
        mps_chain[n-1] = op.rightcontract(mps_chain[n-1], C)
        H_block[n] = op.XopR(R, mpo, mps_R)
    C, mps_R = op.rqpos(mps_chain[0])
    mps_chain[0] = mps_R
    H_block[0] = op.joinL(H_block[0], C)
    return (Es, mps_chain, H_block)


def dmrg_single(mpo_chain, N_sweeps: int, maxchi: int,
                lz_params: dict = None,
                L=None, R=None, mps_chain=None):
    """
    Main loop for single-site finite-chain DMRG.
    """

    errflag, errstr = errors.check_natural(N_sweeps, "N_sweeps")
    if errflag:
        raise ValueError(errstr)

    errflag, errstr = errors.check_natural(maxchi, "maxchi")
    if errflag:
        raise ValueError(errstr)

    N = len(mpo_chain)

    if mps_chain is None:
        mps_chain = op.random_finite_mps(N)

    if L is None:
        L = op.left_boundary_eye()
    if R is None:
        R = op.right_boundary_eye()
    H_block = [0 for _ in N]
    H_block[0] = L
    H_block[-1] = R

    Es = np.array(2*N_sweeps, N)

    print("Initializing chain...")
    _, mps_chain, H_block = left_to_right(mps_chain, H_block, mpo_chain,
                                          lz_params, initialization=True)
    print("Initialization complete. And so it begins...")
    for sweep in range(N_sweeps):
        EsR, mps_chain, H_block = right_to_left(mps_chain, H_block, mpo_chain,
                                                lz_params)
        EsL, mps_chain, H_block = left_to_right(mps_chain, H_block, mpo_chain,
                                                lz_params)
        Es[2*sweep, :] = EsL
        Es[2*sweep + 1, :] = EsR
        E = 0.5*(jnp.mean(EsL)[0] + jnp.mean(EsR)[0])
        print("Sweep: ", sweep, "<E>: ", E)
    return (Es, mps_chain, H_block)
