from functools import partial

import jax
import jax.numpy as jnp

import jax_dmrg.utils as utils
import jax_dmrg.map


def lz_params(
        ncv=4,
        lz_tol=1E-12,
        lz_maxiter=2
        ):
    params = {"ncv": ncv,
              "lz_tol": lz_tol,
              "lz_maxiter": lz_maxiter}
    return params


@jax.jit
def softnorm(v):
    return jnp.amax([jnp.linalg.norm(v), 1E-8])


def dmrg_solve(A, L, R, mpo, lz_params=None):
    """
    The local ground state step of single-site DMRG.
    """
    if lz_params is None:
        lz_params = lz_params()
    keys = ["ncv", "lz_tol", "lz_maxiter"]
    n_krylov, tol, maxiter = [lz_params[key] for key in keys]

    mpo_map = jax_dmrg.map.SingleMPOHeffMap(mpo, L, R)
    A_vec = jnp.ravel(A)
    E, eV, err = minimum_eigenpair(mpo_map, n_krylov, v0=A_vec,
                                   tol=tol, maxiter=maxiter, verbose=False)
    newA = eV.reshape(A.shape)
    return (E, newA, err)


def matrix_optimize(A, n_krylov=32, tol=1E-6, rtol=1E-8, maxiter=10, v0=None,
                    verbose=False):
    """
    The minimum eigenpair of a dense Hermitian matrix A.
    """
    A_map = jax_dmrg.map.MatrixMap(A, hermitian=True)
    E, eV, err = minimum_eigenpair(A_map, n_krylov, tol=tol, rtol=rtol, v0=v0,
                                   maxiter=maxiter, verbose=verbose)

    return (E, eV, err)


def minimum_eigenpair(A_op, n_krylov, tol=1E-6, rtol=1E-9, maxiter=10,
                      v0=None, verbose=False):
    """
    Find the algebraically minimum eigenpair of the Hermitian operator A_op
    using explicitly restarted Lanczos iteration.

    PARAMETERS
    ----------
    A_op: Hermitian operator.
    n_krylov: Size of Krylov subspace.
    tol, rtol: Absolute and relative error tolerance at which convergence is
               declared.
    maxiter: The program ends after this many iterations even if unconverged.
    v0: An optional initial vector.


    """
    m, n = A_op.shape
    if m != n:
        raise ValueError("Lanczos requires a Hermitian matrix; your shape was",
                         A_op.shape)
    if v0 is None:
        v, = utils.random_tensors([(n,)], dtype=A_op.dtype)
    else:
        v = v0
    olderr = 0.
    A_mv = A_op.matvec
    A_data = A_op.data
    for it in range(maxiter):
        E, v, err = eigenpair_iteration(A_mv, A_data, v, n_krylov)
        if verbose:
            print("LZ Iteration: ", it)
            print("\t E=", E, "err= ", err)
        if jnp.abs(err - olderr) < rtol or err < tol:
            return (E, v, err)
        if not it % 3:
            olderr = err
    if verbose:
        print("Warning: Lanczos solve exited without converging.")
    return (E, v, err)


@partial(jax.jit, static_argnums=(3,))
def eigenpair_iteration(A_matvec, A_data, v, n_krylov):
    """
    Performs one iteration of the explicitly restarted Lanczos method.
    """
    K, T = tridiagonalize(A_matvec, A_data, v, n_krylov)
    Es, eVsT = jnp.linalg.eigh(T)
    E = Es[0]
    min_eVT = eVsT[:, 0]
    psi = K @ min_eVT
    psi = psi / softnorm(psi)
    Apsi = A_matvec(A_data, psi)
    err = jnp.linalg.norm(jnp.abs(E*psi - Apsi))
    return (E, psi, err)


@partial(jax.jit, static_argnums=(3,))
def tridiagonalize(A_matvec, A_data, v0, n_krylov):
    """
    Lanczos tridiagonalization. A_matvec and A_data collectively represent
    a Hermitian
    linear map of a length-n vector with initial value v0 onto its vector space
    v = A_matvec(A_data, v0). Returns an n x n_krylov vector V that
    orthonormally spans the Krylov space of A, and an symmetric, real,
    and tridiagonal matrix T = V^dag A V of size n_krylov x n_krylov.

    PARAMETERS
    ----------
    A_matvec, A_data: represent the linear operator.
    v0              : a length-n vector.
    n_krylov        : size of the krylov space.

    RETURNS
    -------
    K (n, n_krylov) : basis of the Krylov space.
    T (n_krylov, n_krylov) : Tridiagonal projection of A onto V.
    """

    vs = []  # Krylov vectors.
    alphas = []  # Diagonal entries of T.
    betas = []  # Off-diagonal entries of T.
    betas.append(0.)
    vs.append(0)
    v = v0/softnorm(v0) + 0.j
    vs.append(v)
    for k in range(1, n_krylov + 1):
        v = A_matvec(A_data, vs[k])
        alpha = (vs[k] @ v).real
        alphas.append(alpha)
        v = v - betas[k - 1] * vs[k - 1] - alpha * vs[k]
        beta = softnorm(v).real
        betas.append(beta)
        vs.append(v / beta)

    K = jnp.array(vs[1:-1]).T
    alpha_arr = jnp.array(alphas)
    beta_arr = jnp.array(betas[1:-1])
    T = jnp.diag(alpha_arr) + jnp.diag(beta_arr, 1) + jnp.diag(beta_arr, -1)
    return (K, T)
