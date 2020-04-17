import jax
import jax.numpy as jnp


class AbstractMap(jax.ShapedArray):
    """
    ABC for a Jit-friendly linear transformation v -> Av, where v is a
    length-N vector and A an M x N matrix. The transform is performed by either
    (SubClass).matvec(v), (SubClass).matvec({data specifying A}, v),
    depending on the subclass.

    AbstractMap.matvec is an instance of jax.tree_util.Partial, which is a
    Jax type, and can thus be fed into Jitted code without needing to be
    treated via static_args. For certain of its subclasses,
    recompilation will trigger only on new AbstractMap *shapes* instead of
    instances.

    Subclasses of AbstractMap are designed to implement similar functionality
    as
    scipy.sparse.LinearMap, in a way that gets along with Jax and especially
    Jit.

    An AbstractMap is a ShapedArray, whose shape and dtype are that of
    A. AbstractMap also accepts a flag 'hermitian', which has no effect
    internally, but may be used by other code (e.g. sparse solvers) to make
    algorithmic choices.

    The generic subclass is:
        -LinearMap, which accepts a generic matvec function at construction,
                    but triggers a recompilation once per new *instance*
                    if fed into Jitted code. That is,
                        #  map_1 = AbstractMap(*args)
                        #  map_2 = AbstractMap(*args)
                        #  jitted_function(map_1.matvec)
                        #  jitted_function(map_2.matvec)
                    should trigger two recompilations, even though *args is
                    unchanged, but
                        #  map_1 = AbstractMap(*args)
                        #  jitted_function(map_1.matvec)
                        #  jitted_function(map_1.matvec)
                    should trigger only one. This is the same behaviour that
                    would be obtained if jitted_function treated A.matvec
                    as using static_args.

                    When using LinearMap, map_1(v) implements v->Av.

    The specialized subclasses each implement a specific
    matvec operation. matvec in these cases is a static method, and thus
                        #  map_1 = (specialized_subclass)(*args)
                        #  map_2 = (specialized_subclass)(*args)
                        #  jitted_function(map_1.matvec)
                        #  jitted_function(map_2.matvec)
    recompiles only *once*, because map_1.matvec and map_2.matvec are the same
    object. For the same reason, map_1.matvec needs to be explicitly
    fed in the data specifying A.

    That is, map_1.matvec((data specifying A), v)
    implements v->Av. jitted_function will therefore typically also need the
    data specifying A - usually one or more Jax arrays - as additional
    arguments compared to the LinearMap case. These arguments will trigger
    recompilations upon changes of *shape*.

    The available specialized subclasses are:
          -MatrixMap: does A@v where A is an explicit dense matrix.
                      To do the map:
                        # map = MatrixMap(array)
                        # new_v = map.matvec(map.A, v)

          -MPSTransferRMap
    """

    def __init__(self, shape, dtype=jnp.float32,
                 hermitian: bool = False, weak_type: bool = False):
        """
        shape: shape of the matrix A.
        dtype: Expected data type of A and v.
        hermitian: self.hermitian is a flag signalling to external code that
                   A is Hermitian.
        weak_type: This is required by ShapedArray but I'm not yet sure what it
                   does.

        Consistency of the above with the actual matvec is not checked.
        """
        self.hermitian = hermitian
        super().__init__(shape, dtype, weak_type=weak_type)

    def matvec(*args, v: jnp.array):
        raise NotImplementedError("Can't call matvec from the ABC!")


class LinearMap(AbstractMap):
    """
    The generic AbstractMap, taking an arbitrary function matvec at
    construction. matvec is essentially treated as a static argument as
    far as Jax is concerned.
    """
    def __init__(self, matvec, shape,
                 dtype=jnp.float32,
                 hermitian: bool = False, weak_type: bool = False):
        """
        matvec: Function object representing new_v = A @ v. It must be
                Jittable.
        """
        super().__init__(shape, dtype=dtype, hermitian=hermitian,
                         weak_type=weak_type)

        self.data = None
        self.matvec = jax.tree_util.Partial(jax.jit(matvec))

    @jax.tree_util.Partial
    def matvec(self, v):
        return self.matvec(v)


@jax.tree_util.Partial
@jax.jit
def matrix_matvec(A, v):
    return A@v


class MatrixMap(AbstractMap):
    """
    An AbstractMap representing A@v where A is an explicit dense matrix.
    To do the map:
    # map = MatrixMap(array)
    # new_v = map.matvec(map.A, v) -> new_v = array @ v.
    MatrixLinearMap.matvec can be passed into Jitted code, without new
    instances triggering recompilations.
    """
    def __init__(self, A, hermitian: bool = False):
        if len(A.shape) != 2:
            raise ValueError("Must construct from a dim 2 array.")

        super().__init__(A.shape, dtype=A.dtype, hermitian=hermitian)
        self.data = [A]

    @jax.tree_util.Partial
    def matvec(data, v):
        """
        Does the matrix-vector multiplication. This is effectively a static
        method, so the calling sequence is a bit weird:
        # op = MatrixLinearMap(A)
        # newv = op.matvec(op, v)
        """
        A, = data
        return matrix_matvec(A, v)


class MPSTransferMap(AbstractMap):
    """
    Represents a contraction with the MPS transfer matrix. The subclasses
    MPSTransferMap_L and MPSTransferMap_R contract on the left and right
    respectively.

    The contraction is handled "sparsely" by contracting with A instead of
    forming the dense transfer matrix. To perform the map:

    # A = jnp.ones((chiL, d, chiR))
    # mapA = MPSTransferMapL(A) # or MPSTransferMapR(A)
    # r = jnp.ones((chi, chi)).reshape((chi**2))
    # newr = mapA.matvec(A, v) -> shaped as a chi**2 vector
    """
    def __init__(self, A):
        """
        A: the (chiL, d, chiR) MPS tensor. self.shape is (chiL^2, chiR^2).
        """
        if len(A.shape) != 3:
            raise ValueError(
                "Must construct from a dim 3 MPS tensor (chiL, d, chiR)")
        chiL, d, chiR = A.shape
        tm_shape = (chiL**2, chiR**2)
        super().__init__(tm_shape, dtype=A.dtype, hermitian=False)
        self.data = A


@jax.tree_util.Partial
@jax.jit
def apply_transfer_right(A, v):
    Av = jnp.dot(A, v)
    Bt = A.conj().transpose((0, 2, 1))
    AvBt = jnp.dot(Av, Bt)
    new_r = jnp.trace(AvBt, axis1=1, axis2=2)
    return new_r


class MPSTransferMapR(MPSTransferMap):
    """
    Contracts an MPS transfer matrix with a matrix r on its right:
     \     ---A---\
    newr =    |   r
     /     ---A*--/

     A is (chiL, d, chiR) and v should be (chiR*chiR)

    """
    def __init__(self, A):
        """
        A: the (chiL, d, chiR) MPS tensor. self.shape is (chiL^2, chiR^2).
        """
        super().__init__(A)

    @jax.tree_util.Partial
    @jax.jit
    def matvec(A, v):
        chiR = A.shape[2]
        v = v.reshape((chiR, chiR))
        r_mat = apply_transfer_right(A, v)
        r_vec = r_mat.reshape((chiR**2))
        return r_vec


@jax.tree_util.Partial
@jax.jit
def apply_transfer_left(A, v):
    LA = jnp.einsum('ab, bcd', v, A.conj())
    ALA = jnp.einsum('abc, ade', A, LA)
    new_l = jnp.trace(ALA, axis1=0, axis2=2)
    return new_l


class MPSTransferMapL(MPSTransferMap):
    """
    Contracts an MPS transfer matrix with a matrix l on its left:
     /     /---A---
    newl = l   |
     \     \---A*--

    The contraction is handled "sparsely" by contracting with A instead of
    forming the dense transfer matrix. To perform the map:

    # A = jnp.ones((chiL, d, chiR))
    # mapA = MPSTransferRMap(A)
    # l = jnp.ones((chiL, chiL))
    # newl = mapA.matvec(A, v)
    """
    def __init__(self, A):
        """
        A: the (chiL, d, chiL) MPS tensor. self.shape is (chiL^2, chiR^2).
        """
        super().__init__(A)

    @jax.tree_util.Partial
    @jax.jit
    def matvec(A, v):
        chiL = A.shape[0]
        v = v.reshape((chiL, chiL))
        l_mat = apply_transfer_left(A, v)
        l_vec = l_mat.reshape((chiL**2))
        return l_vec


@jax.tree_util.Partial
@jax.jit
def apply_mpo_transfer_left(A, mpo, v):
    AL = jnp.einsum('abe, ecd', v, A)
    LAM = jnp.einsum('aefd, ecfb', LA, mpo)
    new_l = jnp.einsum('deba, dec', LAM, A.conj())
    return new_l


@jax.tree_util.Partial
@jax.jit
def apply_mpo_transfer_right(A, mpo, v):
    AR = jnp.einsum('abe, ecd', A, v)
    ARM = jnp.einsum('aefd, bfec', AR, mpo)
    new_r = jnp.einsum('abde, cde', ARM, A.conj())
    return new_r


class MPOEnvironmentMap(AbstractMap):
    """
    """
    def __init__(self, A, mpo):
        if len(mpo.shape) != 4:
            raise ValueError("mpo must be a dim-4 array. Your shape: ",
                             mpo.shape)
        ML, MR, d1, d2 = mpo.shape
        if d1 != d2:
            raise ValueError("mpo must have last two dims equal. Your shape: ",
                             mpo.shape)

        if len(A.shape) != 3:
            print("A must be a dim-3 array. Your shape: ", A.shape)
        chiL, d, chiR = A.shape
        if d != d1:
            raise ValueError("last two dims of mpo (your shape: ", mpo.shape,
                             ") must equal second dim of A (your shape: ",
                             A.shape, ")")
        shape = (ML*chiL*chiL, MR*chiR*chiR)
        super().__init__(shape, dtype=A.dtype, hermitian=False)
        self.data = [A, mpo]


class MPOEnvironmentMapL(MPOEnvironmentMap):
    def __init__(self, A, mpo):
        super().__init__(A, mpo)

    @jax.tree_util.Partial
    @jax.jit
    def matvec(data, v):
        A, mpo = data
        chiL = A.shape[0]
        ML = mpo.shape[0]
        v = v.reshape((chiL, chiL, ML))
        l_ten = apply_mpo_transfer_left(A, mpo, v)
        l_vec = l_ten.reshape((chiL*chiL*ML))
        return l_vec


class MPOEnvironmentMapR(MPOEnvironmentMap):
    def __init__(self, A, mpo):
        super().__init__(A, mpo)

    @jax.tree_util.Partial
    @jax.jit
    def matvec(data, v):
        A, mpo = data
        chiR = A.shape[2]
        MR = mpo.shape[3]
        v = v.reshape((chiR, chiR, MR))
        r_ten = apply_mpo_transfer_right(A, mpo, v)
        r_vec = r_ten.reshape((chiR*chiR*MR))
        return r_vec


@jax.tree_util.Partial
@jax.jit
def single_mpo_heff(mpo, L, R, A):
    newA = jnp.einsum('fad, dhe, fgbh, gce', L, A, mpo, R)
    return newA


class SingleMPOHeffMap(AbstractMap):
    """
    ---A---
    |  |   |
    L--M---R
    |  |   |
    """

    def __init__(self, mpo, L, R):
        if len(mpo.shape) != 4:
            raise ValueError("mpo must be a dim-4 array. Your shape: ",
                             mpo.shape)
        ML, MR, d1, d2 = mpo.shape
        if d1 != d2:
            raise ValueError("mpo must have last two dims equal. Your shape: ",
                             mpo.shape)

        if len(L.shape) != 3:
            print("L must be a dim-3 array. Your shape: ", L.shape)

        if len(R.shape) != 3:
            print("R must be a dim-3 array. Your shape: ", R.shape)

        ML_L, chiL, chiL2 = L.shape
        if ML_L != ML:
            raise ValueError("First dim of L (your shape was ", L.shape,
                             ") must equal first dim of mpo (your shape was ",
                             mpo.shape, ")")
        if chiL != chiL2:
            raise ValueError("First two dims of L must be equal (your shape: ",
                             L.shape, ")")
        MR_R, chiR, chiR2 = R.shape
        if MR_R != MR:
            raise ValueError("First dim of R (your shape was ", R.shape,
                             ") must equal second dim of mpo (your shape: ",
                             mpo.shape, ")")
        if chiR != chiR2:
            raise ValueError("Last two dims of R must be equal (your shape: ",
                             R.shape, ")")
        shape = (ML*chiL*chiR, MR*chiL*chiR)
        super().__init__(shape, dtype=mpo.dtype, hermitian=True)
        self.data = [mpo, L, R]

    @jax.tree_util.Partial
    @jax.jit
    def matvec(data, v):
        mpo, L, R = data
        ML, chiL, _ = L.shape
        MR, chiR, _ = R.shape
        d = mpo.shape[3]
        A = v.reshape((chiL, d, chiR))
        newA = single_mpo_heff(mpo, L, R, A)
        newv = newA.reshape((chiL*d*chiR))
        return newv
