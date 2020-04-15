"""
Low-level tensor network manipulations for single-site finite chain DMRG.

Adam GM Lewis
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax_dmrg.errors as errors
###############################################################################
# INITIALIZERS
###############################################################################
def finite_chis(N):
    chis = []
    raise NotImplementedError()
    return chis


def random_finite_mps(d, N):
    """
    """
    mps_chain = [0 for _ in N]
    raise NotImplementedError()
    return mps_chain


def left_boundary_eye(chiM, dtype=jnp.float32):
    """
    'Identity' left boundary Hamiltonian for a finite chain. chiM
    is the mpo bond dimensions.
    """
    errflag, errstr = errors.check_natural(chiM, "chiM")
    if errflag:
        raise ValueError(errstr)

    L = np.zeros(chiM)
    L[0] = 1.
    L = jnp.array(L, dtype=dtype)
    return L


def right_boundary_eye(chiM, dtype=jnp.float32):
    """
    'Identity' right boundary Hamiltonian for a finite chain. chiM
    is the mpo bond dimensions.
    """
    errflag, errstr = errors.check_natural(chiM, "chiM")
    if errflag:
        raise ValueError(errstr)

    R = np.zeros(chiM)
    R[-1] = 1.
    R = jnp.array(R, dtype=dtype)
    return R

def xx_mpo():
    """
    The XX Hamiltonian in MPO form.
    """
    d = 2
    sP = np.sqrt(2)*np.array([[0, 0], [1, 0]])
    sM = np.sqrt(2)*np.array([[0, 1], [0, 0]])
    sI = np.array([[1, 0], [0, 1]])
    M = np.zeros([4, 4, d, d])
    M[0, 0, :, :] = sI
    M[3, 3, :, :] = sI
    M[0, 1, :, :] = sM
    M[1, 3, :, :] = sP
    M[0, 2, :, :] = sP
    M[2, 3, :, :] = sM
    M = jnp.array(M)
    return M


def xx_chain(N):
    return [xx_mpo() for _ in range(N)]


###############################################################################
# CONTRACTIONS
###############################################################################
def leftcontract(C, mps):
    """
    --0C1----0mps2-- -> --0mps2--
               1            1
               |            |
    """
    mps = jnp.einsum("ab, bcd", C, mps)
    return mps


def rightcontract(mps, C):
    """
    --0mps2----0C1-- -> --0mps2--
        1                   1
        |                   |
    """
    mps = jnp.dot(mps, C)
    return mps


def XopL(L, mpo, mps):
    """
    ----0mps2--      ---
    |     1          |
    2     3          2
    L0--0mpo1--  ->  L0-
    1     2          1
    |     |          |
    -----mps*--      ---
    """
    mps_d = jnp.conj(mps)
    L = jnp.einsum("hed, hagf, dfc, egb", L, mpo, mps, mps_d)
    return L


def XopR(R, mpo, mps):
    """
    ---0mps2--       --
         1    |       |
         3    2       2
    ---0mpo1-0R  ->  0R
         2    1       1
         |    |       |
    ----mps*--|      --
    """
    mps_d = jnp.conj(mps)
    R = jnp.einsum("hed, ahgf, cfd, bge", R, mpo, mps, mps_d)
    return R


@jax.jit
def joinL(L, C):
    """
    --0C1-     ---
    |          |
    2          2
    L0--   ->  L0-
    1          1
    |          |
    ----       ---
    """
    return jnp.dot(L, C)


@jax.jit
def joinR(C, R):
    """
      --          --
       |           |
       2           2
     -0R   ->    -0R
       1           1
       |           |
    -0C1          --
    """
    return jnp.dot(C, R)

###############################################################################
# QR
###############################################################################
def qrpos(mps):
    """
    Reshapes the (chiL, d, chiR) MPS tensor into a (chiL*d, chiR) matrix,
    and computes its QR decomposition, with the phase of R fixed so as to
    have a non-negative main diagonal. A new left-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    R.

    PARAMETERS
    ----------
    mps (array-like): The (chiL, d, chiR) MPS tensor.

    RETURNS
    -------
    mps_L, R: A left-orthogonal (chiL, d, chiR) MPS tensor, and an upper
              triangular (chiR x chiR) matrix with a non-negative main
              diagonal such that mps = mps_L @ R.
    """
    chiL, d, chiR = mps.shape
    mps_mat = jnp.reshape(mps, (chiL*d, chiR))
    Q, R = jnp.linalg.qr(mps_mat)
    phases = jnp.sign(jnp.diag(R))
    Q = Q*phases
    R = jnp.conj(phases)[:, None] * R
    mps_L = Q.reshape(mps.shape)
    return (mps_L, R)


def lqpos(mps):
    """
    Reshapes the (chiL, d, chiR) MPS tensor into a (chiL, d*chiR) matrix,
    and computes its LQ decomposition, with the phase of L fixed so as to
    have a non-negative main diagonal. A new right-orthogonal
    (chiL, d, chiR) MPS tensor (reshaped from Q) is returned along with
    L.

    PARAMETERS
    ----------
    mps (array-like): The (chiL, d, chiR) MPS tensor.

    RETURNS
    -------
    L, mps_R:  A lower-triangular (chiL x chiL) matrix with a non-negative 
               main-diagonal, and a right-orthogonal (chiL, d, chiR) MPS 
               tensor such that mps = L @ mps_R.
    """
    chiL, d, chiR = mps.shape
    mps_mat = jnp.reshape(mps, (chiL, chiR*d))
    mps_mat = jnp.conj(mps_mat.T)
    Qdag, Ldag = jnp.linalg.qr(mps_mat)
    Q = jnp.conj(Qdag.T)
    L = jnp.conj(Ldag.T)
    phases = jnp.sign(jnp.diag(L))
    L = L*phases
    Q = jnp.conj(phases)[:, None] * Q
    mps_R = Q.reshape(mps.shape)
    return (L, mps_R)





