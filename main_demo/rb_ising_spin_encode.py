import numpy as np
from numpy.random import rand
from utility import convert_planar_bsf_to_ising_sign

from spin_utils import encode_spins, decode_spins


class PhaseFlipIsingMonteCarlo_spin_encoded:

    def __init__(self, d, p, error_bsf):
        self.d = d
        self.p = p
        self.beta = np.log((1 - p) / p) / 2
        
        # Initialize the state as an encoded integer
        self.state = encode_spins(np.ones((d - 1, d)))

        self.logical_op_row = d // 2
        self.init_v, self.init_h, self.init_single = convert_planar_bsf_to_ising_sign(error_bsf, (d, d))
        self.v_coup, self.h_coup, self.single = self.init_v, self.init_h, self.init_single

        self.final_v = np.copy(self.init_v)
        self.final_v[self.logical_op_row, :] = -1 * self.final_v[self.logical_op_row, :]
        self.derivative_v = self.final_v - self.init_v
        self.derivative_v = self.derivative_v[self.logical_op_row,]

    def mcmove(self, num_steps=100):
        '''Monte Carlo move using Metropolis algorithm '''
        d = self.d

        for i in range(num_steps):
            a = np.random.randint(0, d - 1)
            b = np.random.randint(0, d)

            # Extract spin at position (a, b)
            position = a * d + b
            spin = ((self.state >> position) & 1) * (-2) + 1

            # Calculate cost
            cost = self.single[a, b]
            if a > 0:
                cost += self.v_coup[a - 1, b] * (((self.state >> ((a - 1) * d + b)) & 1) * (-2) + 1)
            if a < d - 2:
                cost += self.v_coup[a, b] * (((self.state >> ((a + 1) * d + b)) & 1) * (-2) + 1)
            if b > 0:
                cost += self.h_coup[a, b - 1] * (((self.state >> (a * d + b - 1)) & 1) * (-2) + 1)
            if b < d - 1:
                cost += self.h_coup[a, b] * (((self.state >> (a * d + b + 1)) & 1) * (-2) + 1)

            cost *= 2*spin

            # Metropolis update
            if cost < 0 or rand() < np.exp(-cost * self.beta):
                self.state ^= (1 << position)

    def estimate_energy_derivative(self, l, mcmove_steps=100, num_samples=1000):
        '''Estimate the derivative of the energy with respect to lambda'''
        self.v_coup = l * self.final_v + (1 - l) * self.init_v
        self.mcmove(1000)

        s = 0
        for i in range(num_samples):
            spins_row = np.array([((self.state >> (i + self.logical_op_row * self.d)) & 1) * (-2) + 1 for i in range(self.d)])
            spins_row_next = np.array([((self.state >> (i + (self.logical_op_row + 1) * self.d)) & 1) * (-2) + 1 for i in range(self.d)])

            # Calculate the sum of the element-wise product
            s += np.sum(self.derivative_v * spins_row * spins_row_next)
            self.mcmove(mcmove_steps)

        return s / num_samples
    

    def estimate_energy_derivative_decode(self, l, mcmove_steps=100, num_samples=1000):
        '''Estimate the derivative of the energy with respect to lambda'''
        self.v_coup = l * self.final_v + (1 - l) * self.init_v
        self.mcmove(1000)
        
        s = 0
        for i in range(num_samples):
            # Decoding the current state to a 2D array
            state_2d = decode_spins(self.state, (self.d - 1, self.d))
            
            # Calculate the sum of the element-wise product directly
            s += np.sum(self.derivative_v * state_2d[self.logical_op_row] * state_2d[self.logical_op_row + 1])
            self.mcmove(mcmove_steps)

        return s / num_samples

