import numpy as np
from numpy.random import rand


from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel, BitFlipErrorModel, PhaseFlipErrorModel
from qecsim.models.planar import PlanarCode, PlanarMWPMDecoder, PlanarMPSDecoder

import pickle

# We try to find syndromes such that min-weight matching (MWPM) decoder fails
# and MPS decoder succeeds.


def gen_hard_synd(d=17, error_probability=0.1, num_samples=1, seed=1):
    # initialise models

    my_code = PlanarCode(d, d)
    my_error_model = PhaseFlipErrorModel()
    mwpm_decoder = PlanarMWPMDecoder()
    mps_decoder = PlanarMPSDecoder(chi=8)


    # initialize syndrome and error list
    syndrome_list = []
    error_list = []
    gen_number = 0
    rng = np.random.default_rng(seed)
    while gen_number < num_samples:
        # error: random error based on error probability
        error = my_error_model.generate(my_code, error_probability, rng=rng)

        # syndrome: stabilizers that do not commute with the error
        syndrome = pt.bsp(error, my_code.stabilizers.T)

        recovery = mwpm_decoder.decode(my_code, syndrome)

        if np.sum(pt.bsp(recovery ^ error,
                         my_code.logicals.T)) > 0:  # check if mwqm fails
            recovery1 = mps_decoder.decode(my_code, syndrome)
            if np.sum(pt.bsp(recovery1 ^ error, my_code.logicals.T)) == 0:
                syndrome_list.append(syndrome)
                error_list.append(error)
                gen_number += 1

    return syndrome_list, error_list
