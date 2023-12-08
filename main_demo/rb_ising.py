import numpy as np
from numpy.random import rand
from utility import convert_planar_bsf_to_ising_sign


class PhaseFlipIsingMonteCarlo:

    def __init__(self, d, p, error_bsf):
        self.d = d
        self.p = p

        self.beta = np.log((1 - p) / p) / 2
        self.state = np.ones((d - 1, d))

        self.logical_op_row = d // 2

        self.init_v, self.init_h, self.init_single = convert_planar_bsf_to_ising_sign(
            error_bsf, (d, d))

        self.v_coup, self.h_coup, self.single = self.init_v, self.init_h, self.init_single

        self.final_v = np.copy(self.init_v)
        # flip the sign of the final vertical couplings to create logical operator
        self.final_v[
            self.logical_op_row, :] = -1 * self.final_v[self.logical_op_row, :]

        self.derivative_v = self.final_v - self.init_v
        self.derivative_v = self.derivative_v[self.logical_op_row,]

        # print(self.init_v)
        # print(self.final_v)
        # print(self.derivative_v)

    def mcmove(self, num_steps=100):
        '''Monte Carlo move using Metropolis algorithm '''
        d = self.d
        state = self.state
        for i in range(num_steps):

            a = np.random.randint(0, d - 1)
            b = np.random.randint(0, d)

            cost = 0
            if a > 0:
                cost += self.v_coup[a - 1, b] * state[a - 1, b]
            if a < d - 2:
                cost += self.v_coup[a, b] * state[a + 1, b]
            if b > 0:
                cost += self.h_coup[a, b - 1] * state[a, b - 1]
            if b < d - 1:
                cost += self.h_coup[a, b] * state[a, b + 1]

            cost += self.single[a, b]
            cost = 2 * state[a, b] * cost

            if cost < 0:
                state[a, b] *= -1
            elif rand() < np.exp(-cost * self.beta):
                state[a, b] *= -1

        self.state = state

    def estimate_energy_derivative(self, l, mcmove_steps=100,num_samples=1000):
        '''Estimate the derivative of the energy with respect to lambda'''

        self.v_coup = l * self.final_v + (1 - l) * self.init_v
        self.mcmove(1000)

        s = 0
        for i in range(num_samples):
            s += np.sum(self.derivative_v * self.state[self.logical_op_row] *
                        self.state[self.logical_op_row + 1])

            self.mcmove(mcmove_steps)

        return s / num_samples

class ParallelPhaseFlipIsingMonteCarlo:

    def __init__(self, d, p, error_bsf, num_samples=32):
        # System parameters
        self.d = d
        self.p = p
        self.beta = np.log((1 - p) / p) / 2
        self.num_samples = num_samples

        # State array: num_samples x (d-1) x d
        self.state = np.random.choice([-1, 1], size=(num_samples, d - 1, d))

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

    def mcmove(self, num_steps=100):
        #同时检测接受率
        acceptance_rates = []
        for _ in range(num_steps):
            # Randomly select a spin position for each simulation
            a = np.random.randint(0, self.d - 1, size=self.num_samples)
            b = np.random.randint(0, self.d, size=self.num_samples)

            # Compute interaction energy for each simulation
            cost = np.zeros(self.num_samples)+ self.single[a, b]
            mask_a_gt_0 = a > 0
            cost[mask_a_gt_0] += self.v_coup[a[mask_a_gt_0] - 1, b[mask_a_gt_0]] * self.state[mask_a_gt_0, a[mask_a_gt_0] - 1, b[mask_a_gt_0]]
            mask_a_lt_d_minus_2 = a < self.d - 2
            cost[mask_a_lt_d_minus_2] += self.v_coup[a[mask_a_lt_d_minus_2], b[mask_a_lt_d_minus_2]] * self.state[mask_a_lt_d_minus_2, a[mask_a_lt_d_minus_2] + 1, b[mask_a_lt_d_minus_2]]
            
            mask_b_gt_0 = b > 0
            cost[mask_b_gt_0] += self.h_coup[a[mask_b_gt_0], b[mask_b_gt_0] - 1] * self.state[mask_b_gt_0, a[mask_b_gt_0], b[mask_b_gt_0] - 1]
            mask_b_lt_d_minus_1 = b < self.d - 1
            cost[mask_b_lt_d_minus_1] += self.h_coup[a[mask_b_lt_d_minus_1], b[mask_b_lt_d_minus_1]] * self.state[mask_b_lt_d_minus_1, a[mask_b_lt_d_minus_1], b[mask_b_lt_d_minus_1] + 1]

            cost *= 2 * self.state[np.arange(self.num_samples), a, b]

            # Update spins based on Metropolis criterion
            mask_flip = (cost < 0) | (np.random.rand(self.num_samples) < np.exp(-cost * self.beta))
            #print(mask_flip)
            self.state[mask_flip, a[mask_flip], b[mask_flip]] *= -1

            #monitor the acception ratio
            acceptance_rate = np.sum(mask_flip) / self.num_samples
            acceptance_rates.append(acceptance_rate)

        self.flip_acceptance = np.mean(acceptance_rates)


    def estimate_energy_derivative(self, l, mcmove_steps=1000):
        # Set the coupling based on the current lambda value
        self.v_coup = l * self.final_v + (1 - l) * self.init_v

        # Update the state using mcmove
        self.mcmove(mcmove_steps)

        # Compute the energy derivative for each simulation
        energy_derivatives = np.sum(self.derivative_v * self.state[:, self.logical_op_row] *
                                self.state[:, self.logical_op_row + 1], axis=1)

        # Return the average energy derivative
        #return np.mean(energy_derivatives), energy_derivatives
        return energy_derivatives
    
class LambdaParallelPhaseFlipIsingMonteCarlo:

    def __init__(self, d, p, error_bsf, num_int_steps):
        self.d = d
        self.p = p
        self.beta = np.log((1 - p) / p) / 2
        self.lambda_values = np.linspace(0., 1, num_int_steps)
        self.num_lambdas = num_int_steps

        # State array for each lambda: num_lambdas x (d-1) x d
        self.state = np.ones((self.num_lambdas, d - 1, d), dtype=int)

        self.logical_op_row = d // 2

        init_v, init_h, self.single = convert_planar_bsf_to_ising_sign(error_bsf, (d, d))
        final_v = np.copy(init_v)
        final_v[self.logical_op_row, :] = -1 * final_v[self.logical_op_row, :]

        self.derivative_v = final_v - init_v
        self.derivative_v = self.derivative_v[self.logical_op_row,]

        # Prepare couplings for each lambda
        self.v_couplings = np.array([l * final_v + (1 - l) * init_v for l in self.lambda_values])
        self.h_coup = init_h

    def mcmove(self, num_steps=100):
        for _ in range(num_steps):
            a = np.random.randint(0, self.d - 1, size=self.num_lambdas)
            b = np.random.randint(0, self.d, size=self.num_lambdas)

            # Compute interaction energy for each lambda
            cost = np.zeros(self.num_lambdas)+ self.single[a, b]
            mask_a_gt_0 = a > 0
            cost[mask_a_gt_0] += self.v_couplings[mask_a_gt_0, a[mask_a_gt_0] - 1, b[mask_a_gt_0]] * self.state[mask_a_gt_0, a[mask_a_gt_0] - 1, b[mask_a_gt_0]]
            mask_a_lt_d_minus_2 = a < self.d - 2
            cost[mask_a_lt_d_minus_2] += self.v_couplings[mask_a_lt_d_minus_2, a[mask_a_lt_d_minus_2], b[mask_a_lt_d_minus_2]] * self.state[mask_a_lt_d_minus_2, a[mask_a_lt_d_minus_2] + 1, b[mask_a_lt_d_minus_2]]

            mask_b_gt_0 = b > 0
            cost[mask_b_gt_0] += self.h_coup[a[mask_b_gt_0], b[mask_b_gt_0] - 1] * self.state[mask_b_gt_0, a[mask_b_gt_0], b[mask_b_gt_0] - 1]
            mask_b_lt_d_minus_1 = b < self.d - 1
            cost[mask_b_lt_d_minus_1] += self.h_coup[a[mask_b_lt_d_minus_1], b[mask_b_lt_d_minus_1]] * self.state[mask_b_lt_d_minus_1, a[mask_b_lt_d_minus_1], b[mask_b_lt_d_minus_1] + 1]

            cost *= 2 * self.state[np.arange(self.num_lambdas), a, b]

            # Update spins based on Metropolis criterion
            mask_flip = (cost < 0) | (np.random.rand(self.num_lambdas) < np.exp(-cost * self.beta))
            self.state[mask_flip, a[mask_flip], b[mask_flip]] *= -1

    def estimate_energy_derivative(self, mcmove_steps=100, num_samples=1000):
        accumulated_sums = np.zeros(self.num_lambdas)

        for i in range(num_samples):
            accumulated_sums += np.sum(self.derivative_v * self.state[:, self.logical_op_row] *
                                       self.state[:, self.logical_op_row + 1], axis=1)
            self.mcmove(mcmove_steps)

        return accumulated_sums / num_samples
