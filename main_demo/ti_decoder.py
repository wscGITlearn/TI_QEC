import numpy as np
from rb_ising import PhaseFlipIsingMonteCarlo, ParallelPhaseFlipIsingMonteCarlo, LambdaParallelPhaseFlipIsingMonteCarlo
from rb_ising_spin_encode import PhaseFlipIsingMonteCarlo_spin_encoded

import matplotlib.pyplot as plt




def ti_calc(recovery,
               error_probability=0.1,
               d=7,
               plot_samples=False,
               num_int_steps=51,
               mcmove_steps_for_derivative=100,
               num_samples=1000):

    #mc = PhaseFlipIsingMonteCarlo(d, error_probability, recovery)
    mc = ParallelPhaseFlipIsingMonteCarlo(d, error_probability, recovery,num_samples)#测试并行
    #mc = LambdaParallelPhaseFlipIsingMonteCarlo(d, error_probability, recovery,num_int_steps)

    '''生成耦合常数用于理解图像
    print(f"init_v: \n{mc.init_v}\n")#测试用
    print(f"init_h: \n{mc.init_h}\n")#测试用
    print(f"init_single: \n{mc.init_single}\n")#测试用
    print(f"final_v: \n{mc.final_v}\n")#测试用
    print(f"derivative_v: \n{mc.derivative_v}\n")#测试用
    '''

    mc.mcmove(500000) # 初始化至平衡态步骤，步数固定

    """"""
    #lambda串行积分适用于PhaseFlipIsingMonteCarlo, ParallelIsingMonteCarlo,
    int_sum = 0

    for l in np.linspace(0., 1, num_int_steps):

        # estimate energy derivative
        energy_derivative = mc.estimate_energy_derivative(l, mcmove_steps=mcmove_steps_for_derivative)
        print('l = {:.3f}, energy derivative = {:.4f}'.format(l, energy_derivative))
        if plot_samples:
            plt.imshow(-mc.state, cmap='Greys')
            plt.show()
            print(mc.derivative_v * mc.state[mc.logical_op_row] *
                        mc.state[mc.logical_op_row + 1])

        int_sum += energy_derivative

    return int_sum/num_int_steps * mc.beta
    

    """
    #lambda并行积分：LambdaParallelIsingMonteCarlo
    # Step 1: Estimate energy derivatives for all lambda values
    energy_derivatives = mc.estimate_energy_derivative(mcmove_steps=mcmove_steps_for_derivative, num_samples=num_samples)
    print(energy_derivatives)
    integral_result = np.trapz(energy_derivatives, dx=1/mc.num_lambdas)

    return integral_result * mc.beta
    """

def ti_calc_serial(recovery,
               error_probability=0.1,
               d=7,
               plot_samples=False,
               num_int_steps=51,
               mcmove_steps_for_derivative=100,
               num_samples=1000):

    mc = PhaseFlipIsingMonteCarlo(d, error_probability, recovery)

    mc.mcmove(500000) # 初始化至平衡态步骤，步数固定
    print("initial balance achieved")

    user_input = input("Do you want to continue? (yes/y): ")
    if user_input.lower() not in ['yes', 'y']:
        return "Operation aborted by the user."

    """"""
    #lambda串行积分适用于PhaseFlipIsingMonteCarlo, ParallelIsingMonteCarlo,
    int_sum = 0

    for l in np.linspace(0., 1, num_int_steps):

        # estimate energy derivative
        energy_derivative = mc.estimate_energy_derivative(l, mcmove_steps=mcmove_steps_for_derivative,num_samples=num_samples)
        print('l = {:.3f}, energy derivative = {:.4f}'.format(l, energy_derivative))
        if plot_samples:
            plt.imshow(-mc.state, cmap='Greys')
            plt.show()
            print(mc.derivative_v * mc.state[mc.logical_op_row] *
                        mc.state[mc.logical_op_row + 1])

        int_sum += energy_derivative

    return int_sum/num_int_steps * mc.beta

def ti_calc_Parallel(recovery,
               error_probability=0.1,
               d=7,
               num_int_steps=51,
               mcmove_steps_init = 500000, 
               mcmove_steps_for_derivative=1000,
               num_samples=1000):

    mc = ParallelPhaseFlipIsingMonteCarlo(d, error_probability, recovery,num_samples)#测试并行

    mc.mcmove(mcmove_steps_init)#初始哈密顿量弛豫

    # Store energy derivatives for all samples across all λ values
    energy_derivatives_samples = np.zeros((num_int_steps, num_samples))

    lambda_values = np.linspace(0., 1, num_int_steps)
    for idx, l in enumerate(lambda_values):
        # Estimate energy derivatives for all samples at this λ;monitor acception ratio
        energy_derivatives_samples[idx] = mc.estimate_energy_derivative(l, mcmove_steps=mcmove_steps_for_derivative)
        print('l = {:.3f}, energy derivative = {:.4f}, spin flip ratio = {:.3f}'.format(l, np.mean(energy_derivatives_samples[idx]), mc.flip_acceptance))

    # Integrate for each sample
    integrals_for_each_sample = np.trapz(energy_derivatives_samples, x=lambda_values, axis=0) * mc.beta

    # Compute mean and error bar
    mean_integral = np.mean(integrals_for_each_sample)
    error_bar = np.std(integrals_for_each_sample) / np.sqrt(num_samples)

    return mean_integral, error_bar


def ti_calc_LambdaParallel(recovery,
               error_probability=0.1,
               d=7,
               plot_samples=False,
               num_int_steps=51,
               mcmove_steps_for_derivative=100,
               num_samples=1000):

    mc = LambdaParallelPhaseFlipIsingMonteCarlo(d, error_probability, recovery,num_int_steps)

    mc.mcmove(400000) # 初始化至平衡态步骤，步数固定  

    #lambda并行积分：LambdaParallelIsingMonteCarlo
    # Step 1: Estimate energy derivatives for all lambda values
    energy_derivatives = mc.estimate_energy_derivative(mcmove_steps=mcmove_steps_for_derivative, num_samples=num_samples)
    print(energy_derivatives)
    integral_result = np.trapz(energy_derivatives, dx=1/mc.num_lambdas)

    return integral_result * mc.beta

def ti_calc_serial_spin_encoded(recovery,
               error_probability=0.1,
               d=7,
               num_int_steps=51,
               mcmove_steps_for_derivative=100,
               num_samples=1000):

    mc = PhaseFlipIsingMonteCarlo_spin_encoded(d, error_probability, recovery)
    mc.mcmove(500000) # 初始化至平衡态步骤，步数固定
    print("initial balance achieved")

    user_input = input("Do you want to continue? (yes/y): ")
    if user_input.lower() not in ['yes', 'y']:
        return "Operation aborted by the user."

    int_sum = 0

    for l in np.linspace(0., 1, num_int_steps):

        # estimate energy derivative
        #energy_derivative = mc.estimate_energy_derivative(l, 
        #                                                   mcmove_steps=mcmove_steps_for_derivative,
        #                                                   num_samples=num_samples)
        energy_derivative = mc.estimate_energy_derivative_decode(l, 
                                                                 mcmove_steps=mcmove_steps_for_derivative,
                                                                 num_samples=num_samples)
        print('l = {:.3f}, energy derivative = {:.4f}'.format(l, energy_derivative))
        
        int_sum += energy_derivative

    return int_sum/num_int_steps * mc.beta