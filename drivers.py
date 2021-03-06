import numpy as np

import jax.numpy as jnp

import jax_dmrg.operations as ops
import jax_dmrg.utils as utils
import jax_dmrg.lanczos as lz
import jax_dmrg.dmrg as dmrg
import jax_dmrg.benchmark as benchmark
import jax_dmrg.map


def time_dmrg_eigensolve(chis, n_krylov, d=2, chiM=4, n_restart=1,
                         n_tries=10,
                         fname="./dmrg_eigen_timings.txt"):
    """
    Times the lanczos minimum eigenproblem on a single DMRG site using
    random float32 input.

    PARAMETERS
    ---------
    chis     : A list or array of integer-valued MPS bond dimensions. It will
               be iterated over and a timing will be produced for each.
    n_krylov : Size of the Krylov subspace.
    d        : Physical dimension.
    chiM     : MPO bond dimension.
    n_restart: The Lanczos iteration is repeated this many times - at each
               restart a new Krylov space is built and a new set of Ritz
               eigenvectors are computed. Therefore, the matvec operation
               will be called n_krylov * n_restart times (the Lanczos solve
               does not terminate early at convergence). The behaviour of
               an unrestarted Lanczos algorithm is roughly recovered by
               setting n_restart=1, n_krylov=ncv.
    n_tries  : Each chi is timed this many times. The reported time is the
               minimum one observed. A single additional, untimed, warmup run
               performed for each chi.
    fname    : The timings are saved to disk at this location.

    RETURNS
    ------
    timings (array, (2, len(chis))) : timings[0, :] = chis
                                      timings[1, :] is the timings.
    This array is also saved to disk, and can be recovered by
    timings = np.loadtxt(fname).

    """
    ts = np.zeros((2, len(chis)))
    ts[0, :] = chis

    for chidx, chi in enumerate(chis):
        print("**************************************************************")
        print("Timing chi= ", chi)
        mps = np.random.rand(chi, d, chi).astype(np.float32)
        L = np.random.rand(chiM, chi, chi).astype(np.float32)
        R = np.random.rand(chiM, chi, chi).astype(np.float32)
        mpo = np.random.rand(chiM, chiM, d, d).astype(np.float32)

        t_chi, _, _ = time_eigensolve_fixed(mps, L, R, mpo, n_krylov,
                                            n_restart, n_tries)
        ts[1, chidx] = t_chi
    return ts


def time_eigensolve_fixed(mps, L, R, mpo, n_krylov, n_restart, n_tries):
    """
    Times the Lanczos minimum eigenproblem on a single DMRG site
    with given specific tensors. mps, L, R, mpo are these tensors;
    the other parameters are the same as those described in
    time_dmrg_eigensolve.

    RETURNS
    ------
    dt (float) : The measured time.
    E  (float) : The computed eigenvalue.
    eV (array) : The computed eigenvector.
    """
    mps = jnp.array(mps)
    L = jnp.array(L)
    R = jnp.array(R)
    mpo = jnp.array(mpo)
    jax_map = jax_dmrg.map.SingleMPOHeffMap(mpo, L, R)

    dts = np.zeros(n_tries)
    for i in range(n_tries):
        t0 = benchmark.tick()
        E, eV, _ = lz.minimum_eigenpair(jax_map, n_krylov,
                                        maxiter=n_restart)
        dts[i] = benchmark.tock(t0, eV)
    dt = np.amin(dts)
    print("t = ", dt, "E = ", E)
    return (dt, E, eV)


def time_tridiagonalize(chis, n_krylov, fname="./tridiag_timings.txt"):
    d = 2
    chiM = 4
    ntries = 10
    ts = np.zeros((2, len(chis)))
    np_mv = jax_dmrg.map.np_matvec
    funcnames = ["Jax", "NumPy"]

    for chidx, chi in enumerate(chis):
        print("**************************************************************")
        print("Timing chi= ", chi)
        
        npmps = np.random.rand(chi, d, chi).astype(np.float32)
        npmps /= np.linalg.norm(npmps)
        npL = np.random.rand(chiM, chi, chi).astype(np.float32)
        npL /= np.linalg.norm(npL)
        npR = np.random.rand(chiM, chi, chi).astype(np.float32)
        npR /= np.linalg.norm(npR)
        npmpo = np.random.rand(chiM, chiM, d, d).astype(np.float32)
        npmpo /= np.linalg.norm(npmpo)
        np_data = (npmpo, npL, npR)

        mps = jnp.array(npmps)
        L = jnp.array(npL)
        R = jnp.array(npR)
        mpo = jnp.array(npmpo)

        jax_map = jax_dmrg.map.SingleMPOHeffMap(mpo, L, R)
        jax_mv = jax_map.matvec

        jax_data = jax_map.data

        def jax_trid():
            return lz.tridiagonalize(jax_mv, jax_data, jnp.ravel(mps),
                                     n_krylov)


        def np_trid():
            return lz.numpy_tridiagonalize(np_mv, np_data, np.ravel(npmps),
                                           n_krylov)

        outKs = []
        outTs = []
        for idx, f in enumerate([jax_trid, np_trid]):
            print("Function:", funcnames[idx])
            dts = np.zeros(ntries)
            for i in range(ntries):
                t0 = benchmark.tick()
                out = f()
                if funcnames[idx] == "Jax":
                    dts[i] = benchmark.tock(t0, out[0])
                else:
                    dts[i] = benchmark.tock(t0)
            outKs.append(out[0])
            outTs.append(out[1])
            ts[idx, chidx] = np.amin(dts)
            print("t=", ts[idx, chidx])

        errK = jnp.linalg.norm(jnp.abs(outKs[0] - outKs[1]))/outKs[0].size
        jnpdiag = jnp.diag(outTs[0])
        npdiag = jnp.diag(outTs[1])
        errT = jnp.linalg.norm(jnp.abs(outTs[0] - outTs[1]))/outTs[0].size
        #print(jnp.abs(outTs[0]-outTs[1]))
        #  print(outTs[0])
        #  print(outTs[1])
        print("ErrK = ", errK)
        print("ErrT = ", errT)
    return ts


def time_contract(contract_fs, funcnames, chis,
                  fname="./contract_timings.txt"):
    d = 2
    chiM = 4
    ts = np.zeros((len(contract_fs), len(chis)))

    for chidx, chi in enumerate(chis):
        print("**************************************************************")
        print("Timing chi= ", chi)
        mps, L, R, mpo = utils.random_tensors([(chi, d, chi),
                                              (chiM, chi, chi),
                                              (chiM, chi, chi),
                                              (chiM, chiM, d, d)])
        outs = []
        for idx, f in enumerate(contract_fs):
            print("Function:", funcnames[idx])
            dts = np.zeros(20)
            for i in range(20):
                t0 = benchmark.tick()
                out = f(mpo, L, R, mps)
                dts[i] = benchmark.tock(t0, out)
            outs.append(out)
            ts[idx, chidx] = np.amin(dts)
            print("t=", ts[idx, chidx])
        A = outs[0]
        for outidx, out in enumerate(outs[1:]):
            err = jnp.linalg.norm(jnp.abs(A) - jnp.abs(out))/A.size
            print("Err", outidx+1, "= ", err)
    return ts


def time_xx(chis=None, N=100, N_sweeps=1, fname="./xxtimings.txt",
        ncv=10, lz_tol=1E-12, lz_maxiter=4):
    timings = []
    if chis is None:
        chis = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        chis = np.array(chis, dtype=np.int)
    chis = np.array(chis)

    for chi in chis:
        print("**************************************************************")
        print("Timing chi= ", chi)
        _ = xx_ground_state(N, chi, 1, ncv=2, lz_maxiter=1)
        #  for mps in mps_chain:
        #      mps = mps.block_until_ready()
        _, _, _, timing = xx_ground_state(N, chi, N_sweeps, ncv=ncv,
                                          lz_tol=lz_tol, lz_maxiter=lz_maxiter)
        for key in timing:
            print(key, ": ", timing[key])
        print("One sweep: ", (timing["total"]-timing["initialization"])/N_sweeps)
        print("**************************************************************")
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


def xx_ground_state(N, maxchi, N_sweeps, ncv=20, lz_tol=1E-5, lz_maxiter=50):
    """
    Find the ground state of the quantum XX model with single-site DMRG.
    """
    

    mpo_chain = [ops.xx_mpo() for _ in range(N)]
    lz_params = lz.lz_params(ncv=ncv, lz_tol=lz_tol, lz_maxiter=lz_maxiter)
    Es, mps_chain, H_block, timings = dmrg.dmrg_single(mpo_chain, maxchi,
                                                       N_sweeps,
                                                       lz_params)
    return (Es, mps_chain, H_block, timings)
