import numpy as np


import jax_dmrg.operations as ops
import jax_dmrg.lanczos as lz
import jax_dmrg.dmrg as dmrg


def time_xx(chis=None, N=100, N_sweeps=1, fname="./timings.txt",
        ncv=10, lz_tol=1E-12, lz_maxiter=4):
    timings = []
    if chis is None:
        chis = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        chis = np.array(chis, dtype=np.int)
    chis = np.array(chis)

    for chi in chis:
        print("***************************************************************")
        print("Timing chi= ", chi)
        _ = xx_ground_state(N, chi, 1)
        _, _, _, timing = xx_ground_state(N, chi, N_sweeps, ncv=ncv,
                                          lz_tol=lz_tol, lz_maxiter=lz_maxiter)
        for key in timing:
            print(key, ": ", timing[key])
        print("One sweep: ", (timing["total"]-timing["initialization"])/N_sweeps)
        print("***************************************************************")
        timings.append(timing)
    ts = np.zeros((chis.size, 6))

    t_tot = [timing["total"] for timing in timings]
    ts[:, 0] = t_tot
    t_lz = [timing["lz"] for timing in timings]
    ts[:, 1] = t_lz
    t_qr = [timing["qr"] for timing in timings]
    ts[:, 2] = t_qr
    t_up = [timing["update"] for timing in timings]
    ts[:, 3] = t_up
    t_init = [timing["initialization"] for timing in timings]
    ts[:, 4] = t_init
    t_swp = (np.array(t_tot) - np.array(t_init)) / N_sweeps
    ts[:, 5] = t_swp
    np.savetxt(fname, ts, header="total, lz, qr, update")
    return timings


def xx_ground_state(N, maxchi, N_sweeps, ncv=4, lz_tol=1E-12, lz_maxiter=2):
    """
    Find the ground state of the quantum XX model with single-site DMRG.
    """

    mpo_chain = [ops.xx_mpo() for _ in range(N)]
    lz_params = lz.lz_params(ncv=ncv, lz_tol=lz_tol, lz_maxiter=lz_maxiter)
    Es, mps_chain, H_block, timings = dmrg.dmrg_single(mpo_chain, maxchi,
                                                       N_sweeps,
                                                       lz_params)
    return (Es, mps_chain, H_block, timings)
