import numpy as np


def convert_planar_bsf_to_ising_sign(bsf, size):
    """
    Convert a phase flip planar code Pauli bsf to sign of coefficients in
    random bond Ising.

    params:
        bsf: planar code Pauli bsf
        size: size of planar code lattice in format (rows, columns), e.g. (5, 5).
    """
    bsf_part = bsf[bsf.size//2:]
    bsf_part = (-1) ** bsf_part
    v_sign = bsf_part[:size[0]*size[1]].reshape(size)
    h_sign = bsf_part[size[0] * size[1]:].reshape((size[0] - 1, size[1] - 1))

    single_sign = np.zeros((size[0]-1, size[1]))
    single_sign[0,:] = v_sign[0,:]
    single_sign[-1, :] = v_sign[-1, :]

    v_sign = v_sign[1:-1, :]

    return v_sign, h_sign, single_sign
