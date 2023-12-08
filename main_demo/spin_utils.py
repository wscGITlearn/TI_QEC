import numpy as np

def encode_spins(spins_2d):
        """
        Encode a 2D array of spins (+1 or -1) into an integer.
        +1 -> 0, -1 -> 1
        """
        spins_flat = spins_2d.flatten()
        bits = (spins_flat == -1).astype(int)
        encoded = np.packbits(bits)
        return int.from_bytes(encoded.tobytes(), 'little')

def decode_spins(encoded_state, shape):
    """
    Decode an integer back to a 2D array of spins (+1 or -1).
    0 -> +1, 1 -> -1
    """
    total_bits = shape[0] * shape[1]
    bits = np.array([(encoded_state & (1 << i)) >> i for i in range(total_bits)])    
    spins_flat = np.where(bits == 0, 1, -1)
    return spins_flat.reshape(shape)


