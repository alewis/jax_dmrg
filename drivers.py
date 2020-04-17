import jax_dmrg.operations as ops
import jax_dmrg.lanczos as lz
import jax_dmrg.dmrg as dmrg


def xx_ground_state(N, maxchi, N_sweeps, ncv=20, lz_tol=1E-6, lz_maxiter=10):
    """
    Find the ground state of the quantum XX model with single-site DMRG.
    """

    mpo_chain = [ops.xx_mpo() for _ in range(N)]
    lz_params = lz.lz_params(ncv=ncv, lz_tol=lz_tol, lz_maxiter=lz_maxiter)
    Es, mps_chain, H_block, timings = dmrg.dmrg_single(mpo_chain, maxchi,
                                                       N_sweeps,
                                                       lz_params)
    return (Es, mps_chain, H_block, timings)
