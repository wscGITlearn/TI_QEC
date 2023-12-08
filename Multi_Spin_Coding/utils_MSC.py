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

def encode_spins(spins_2d):
    """
        Encode a 2D array of spins (+1 or -1) into an integer.
        +1 -> 0, -1 -> 1
    """
    spins_flat = spins_2d.flatten()
    bits = (spins_flat == -1).astype(int)
    encoded = np.packbits(np.flip(bits))  # 翻转位顺序以匹配最低有效位
    return int.from_bytes(encoded.tobytes(), 'big')  # 使用大端字节顺序


def decode_spins(encoded_state, shape):
    """
    Decode an integer back to a 2D array of spins (+1 or -1).
    0 -> +1, 1 -> -1
    """
    d_1, d_2 = shape
    total_bits = d_1 * d_2
    jump = (1 + (total_bits // 8)) * 8 - total_bits
    if jump % 8 == 0:
        jump = 0  # 如果 jump 是8的整数倍，则不需要跳跃
    
    spins_flat = []
    for i in range(total_bits):
        position = i  # 直接使用 i 来索引位置
        spin = ((encoded_state >> (position + jump)) & 1) * (-2) + 1
        spins_flat.append(spin)

    return np.array(spins_flat).reshape(shape)


def encode_spin_samples(spins_3d):
    """
    Encode a 3D array of spins (num_samples, d - 1, d) into an array of integers.
    Each (d - 1, d) spin matrix is encoded into an integer.
    """
    num_samples = spins_3d.shape[0]
    encoded_array = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        encoded_array[i] = encode_spins(spins_3d[i])

    return encoded_array


