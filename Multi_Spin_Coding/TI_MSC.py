import numpy as np
from rb_ising_MSC import PhaseFlipIsingMC_spin_encoded, PhaseFlipIsingMC_spin_encoded_parallel

def ti_calc_serial_spin_encoded(recovery,
               error_probability=0.1,
               d=7,
               num_int_steps=101,
               mcmove_steps_for_derivative=100,
               num_samples=1000):

    mc = PhaseFlipIsingMC_spin_encoded(d, error_probability, recovery)

    print('v_coup:\n',mc.v_coup)
    print('h_coup:\n',mc.h_coup)
    print('single:\n',mc.single)
    print('v_derivative:\n',mc.derivative_v)

    mc.mcmove(500000) # 初始化至平衡态步骤，步数固定
    print("initial balance achieved")
  
    user_input = input("Do you want to continue? (yes/y): ")
    if user_input.lower() not in ['yes', 'y']:
        return "Operation aborted by the user."

    int_sum = 0

    for l in np.linspace(0., 1, num_int_steps):
        energy_derivative = mc.estimate_energy_derivative_decode(l, 
                                                                 mcmove_steps=mcmove_steps_for_derivative,
                                                                 num_samples=num_samples)
        print('l = {:.3f}, energy derivative = {:.4f}'.format(l, energy_derivative))
        
        int_sum += energy_derivative

    return int_sum/num_int_steps * mc.beta


def ti_calc_Parallel_spin_encoded(recovery,
               error_probability=0.1,
               d=7,
               num_int_steps=51,
               mcmove_steps_init = 500000, 
               mcmove_steps_for_derivative=1000,
               num_samples=1000):

    mc = PhaseFlipIsingMC_spin_encoded_parallel(d, error_probability, recovery,num_samples)#

    mc.mcmove_parallel_decode(mcmove_steps_init)#初始哈密顿量弛豫
        
    print("initial balance achieved")
  
    user_input = input("Do you want to continue? (yes/y): ")
    if user_input.lower() not in ['yes', 'y']:
        return "Operation aborted by the user."

    # Store energy derivatives for all samples across all λ values
    energy_derivatives_samples = np.zeros((num_int_steps, num_samples))

    lambda_values = np.linspace(0., 1, num_int_steps)
    for idx, l in enumerate(lambda_values):
        # Estimate energy derivatives for all samples at this λ;monitor acception ratio
        energy_derivatives_samples[idx] = mc.estimate_energy_derivative_parallel_decode(l, mcmove_steps=mcmove_steps_for_derivative)
        print('l = {:.3f}, energy derivative = {:.4f}, spin flip ratio = {:.3f}'.format(l, np.mean(energy_derivatives_samples[idx]), mc.flip_acceptance))

    # Integrate for each sample
    integrals_for_each_sample = np.trapz(energy_derivatives_samples, x=lambda_values, axis=0) * mc.beta

    # Compute mean and error bar
    mean_integral = np.mean(integrals_for_each_sample)
    error_bar = np.std(integrals_for_each_sample) / np.sqrt(num_samples)

    return mean_integral, error_bar, mc.v_coup, mc.h_coup, mc.single