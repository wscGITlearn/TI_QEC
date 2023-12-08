import numpy as np
from numpy.random import rand

from utils_MSC import encode_spins, decode_spins, encode_spin_samples, convert_planar_bsf_to_ising_sign


class PhaseFlipIsingMC_spin_encoded:

    def __init__(self, d, p, error_bsf):
        self.d = d
        self.p = p
        self.beta = np.log((1 - p) / p) / 2
        total_bits = d*(d-1)
        self.decode_jump = (1 + (total_bits // 8)) * 8 - total_bits
        if self.decode_jump % 8 == 0:
            self.decode_jump = 0  # 如果 jump 是8的整数倍，则不需要跳跃
        
        # Initialize the state as an encoded integer
        state_init = np.random.choice([-1, 1], size=(d - 1, d))
        self.state = encode_spins(state_init)

        self.logical_op_row = d // 2
        self.init_v, self.init_h, self.init_single = convert_planar_bsf_to_ising_sign(error_bsf, (d, d))
        self.v_coup, self.h_coup, self.single = self.init_v, self.init_h, self.init_single

        self.final_v = np.copy(self.init_v)
        self.final_v[self.logical_op_row, :] = -1 * self.final_v[self.logical_op_row, :]
        self.derivative_v = self.final_v - self.init_v
        self.derivative_v = self.derivative_v[self.logical_op_row,]

        state_2d = decode_spins(self.state, (self.d - 1, self.d))
        self.energy_derivative_temp = np.sum(self.derivative_v * state_2d[self.logical_op_row] * state_2d[self.logical_op_row + 1])

        #test for encode
        energy_derivative = np.sum(self.derivative_v * state_init[self.logical_op_row] * state_init[self.logical_op_row + 1])
        print(self.energy_derivative_temp-energy_derivative)
        print(state_2d-state_init)
        print((self.derivative_v * state_2d[self.logical_op_row] * state_2d[self.logical_op_row + 1]))
        print((self.derivative_v * state_init[self.logical_op_row] * state_init[self.logical_op_row + 1]))



    def mcmove(self, num_steps=100):
        '''Monte Carlo move using Metropolis algorithm '''
        d = self.d
        self.energy_derivative_record=[]

        for i in range(num_steps):
            a = np.random.randint(0, d - 1)
            b = np.random.randint(0, d)

            # Extract spin at position (a, b)
            position = a * d + b + self.decode_jump
            spin = ((self.state >> position) & 1) * (-2) + 1
            
            #Calculate energy variation
            position_coupling = np.where(a == self.logical_op_row, (self.logical_op_row + 1) * d + b + self.decode_jump, self.logical_op_row * d + b + self.decode_jump)
            spin_coupling = ((self.state >> position_coupling) & 1) * (-2) + 1
            energy_variation = np.where((a == self.logical_op_row) | (a == self.logical_op_row + 1), -2*self.derivative_v[b]*spin*spin_coupling, 0)
            
            # Calculate cost
            cost = self.single[a, b]
            if a > 0:
                cost += self.v_coup[a - 1, b] * (((self.state >> ((a - 1) * d + b + self.decode_jump)) & 1) * (-2) + 1)
            if a < d - 2:
                cost += self.v_coup[a, b] * (((self.state >> ((a + 1) * d + b + self.decode_jump)) & 1) * (-2) + 1)
            if b > 0:
                cost += self.h_coup[a, b - 1] * (((self.state >> (a * d + b - 1 + self.decode_jump)) & 1) * (-2) + 1)
            if b < d - 1:
                cost += self.h_coup[a, b] * (((self.state >> (a * d + b + 1 + self.decode_jump)) & 1) * (-2) + 1)

            cost *= 2*spin

            # Metropolis update
            if cost < 0 or rand() < np.exp(-cost * self.beta):
                self.state ^= (1 << position)
                self.energy_derivative_temp += energy_variation
                
            self.energy_derivative_record.append(self.energy_derivative_temp)
    
    def estimate_energy_derivative_decode(self, l, mcmove_steps=100, num_samples=1000):
        '''Estimate the derivative of the energy with respect to lambda'''
        self.v_coup = l * self.final_v + (1 - l) * self.init_v
        self.mcmove(1000+num_samples*mcmove_steps)
        sum_selected_elements = sum(self.energy_derivative_record[1000::mcmove_steps])

        return sum_selected_elements / num_samples

class PhaseFlipIsingMC_spin_encoded_parallel:

    def __init__(self, d, p, error_bsf, num_samples=32):
        # System parameters
        self.d = d
        self.p = p
        self.beta = np.log((1 - p) / p) / 2
        self.num_samples = num_samples

        #编码读取跨越填充位
        total_bits = d*(d-1)
        self.decode_jump = (1 + (total_bits // 8)) * 8 - total_bits
        if self.decode_jump % 8 == 0:
            self.decode_jump = 0  # 如果 jump 是8的整数倍，则不需要跳跃

        # State array: num_samples x (d-1) x d
        self.state = np.random.choice([-1, 1], size=(num_samples, d - 1, d))

        self.encoded_states = encode_spin_samples(self.state)

        #acception ratio 
        self.flip_acceptance = 0.5

        # Initialize couplings from error_bsf
        self.logical_op_row = d // 2
        self.init_v, self.init_h, self.init_single = convert_planar_bsf_to_ising_sign(error_bsf, (d, d))
        self.v_coup, self.h_coup, self.single = self.init_v, self.init_h, self.init_single

        # Compute final and derivative couplings
        self.final_v = np.copy(self.init_v)
        self.final_v[self.logical_op_row, :] = -1 * self.final_v[self.logical_op_row, :]
        self.derivative_v = self.final_v - self.init_v
        self.derivative_v = self.derivative_v[self.logical_op_row,]

        #compute energy_derivatives_temp
        self.energy_derivatives_temp = np.sum(self.derivative_v * self.state[:, self.logical_op_row] * self.state[:, self.logical_op_row + 1], axis=1)

        
    def mcmove_parallel_decode(self, num_steps=100):
        '''Monte Carlo move using Metropolis algorithm for multiple systems in parallel'''
        num_samples = self.num_samples
        d = self.d
        
        #moniter acceptance_rate
        acceptance_rates = []

        for _ in range(num_steps):
            # Randomly select positions for each sample
            a = np.random.randint(0, d - 1, size=num_samples)
            b = np.random.randint(0, d, size=num_samples)
            positions = a * d + b + self.decode_jump

            # Extract spins at selected positions
            spins = ((self.encoded_states >> positions) & 1) * (-2) + 1

            # Calculate energy variation
            
            position_coupling = np.where(a == self.logical_op_row, (self.logical_op_row + 1) * d + b + self.decode_jump, self.logical_op_row * d + b + self.decode_jump)
            spins_coupling = ((self.encoded_states >> position_coupling) & 1) * (-2) + 1
            energy_variation = np.where((a == self.logical_op_row) | (a == self.logical_op_row + 1), -2*self.derivative_v[b]*spins*spins_coupling, 0)           

            # Calculate cost using masks
            cost = np.zeros(num_samples)+ self.single[a, b]

            mask_a_gt_0 = a > 0
            cost[mask_a_gt_0] += self.v_coup[a[mask_a_gt_0] - 1, b[mask_a_gt_0]] * (((self.encoded_states[mask_a_gt_0] >> ((a[mask_a_gt_0] - 1) * d + b[mask_a_gt_0] + self.decode_jump)) & 1) * (-2) + 1)

            mask_a_lt_d_minus_2 = a < d - 2
            cost[mask_a_lt_d_minus_2] += self.v_coup[a[mask_a_lt_d_minus_2], b[mask_a_lt_d_minus_2]] * (((self.encoded_states[mask_a_lt_d_minus_2] >> ((a[mask_a_lt_d_minus_2] + 1) * d + b[mask_a_lt_d_minus_2] + self.decode_jump)) & 1) * (-2) + 1)

            mask_b_gt_0 = b > 0
            cost[mask_b_gt_0] += self.h_coup[a[mask_b_gt_0], b[mask_b_gt_0] - 1] * (((self.encoded_states[mask_b_gt_0] >> (a[mask_b_gt_0] * d + b[mask_b_gt_0] + self.decode_jump - 1)) & 1) * (-2) + 1)

            mask_b_lt_d_minus_1 = b < d - 1
            cost[mask_b_lt_d_minus_1] += self.h_coup[a[mask_b_lt_d_minus_1], b[mask_b_lt_d_minus_1]] * (((self.encoded_states[mask_b_lt_d_minus_1] >> (a[mask_b_lt_d_minus_1] * d + b[mask_b_lt_d_minus_1] + self.decode_jump + 1)) & 1) * (-2) + 1)

            cost *= 2 * spins

            # Metropolis update
            mask_flip = (cost < 0) | (np.random.rand(num_samples) < np.exp(-cost * self.beta))
            self.encoded_states[mask_flip] ^= (1 << positions[mask_flip])
            self.energy_derivatives_temp[mask_flip] += energy_variation[mask_flip]
            #monitor the acception ratio
            acceptance_rate = np.sum(mask_flip) / self.num_samples
            acceptance_rates.append(acceptance_rate)

        self.flip_acceptance = np.mean(acceptance_rates)

    def estimate_energy_derivative_parallel_decode(self, l, mcmove_steps=1000):
        # Set the coupling based on the current lambda value
        self.v_coup = l * self.final_v + (1 - l) * self.init_v

        # Update the state using mcmove
        self.mcmove_parallel_decode(mcmove_steps)

        """
        for i in range(self.num_samples):
            self.state[i] = decode_spins(self.encoded_states[i], (self.d - 1, self.d))

        energy_derivatives = np.sum(self.derivative_v * self.state[:, self.logical_op_row] * self.state[:, self.logical_op_row + 1], axis=1)
        """


        #return self.energy_derivatives_temp
        return self.energy_derivatives_temp
