import time
import numpy as np
import jax_dmrg.drivers as drivers


def tick():
    return time.perf_counter()


def tock(t0=0., dat=None):
    if dat is not None:
        dat.block_until_ready()
    return time.perf_counter() - t0


def time_xx(chis=None, N=30, N_sweeps=2):
    if chis is None:
        chis = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        chis = np.array(chis, dtype=np.int)
        #chis = np.logspace(2, 8, base=2, num=10, dtype=np.int)
    for chi in chis:
        print("Timing chi= ", chi)
        _ = drivers.xx_ground_state(N, chi, 1)
        _, _, _, timings = drivers.xx_ground_state(N, chi, N_sweeps)
        print(timings)
    return timings
