import jax
import jax.numpy as jnp
from phase_flip_jax import initialize_jax_mc, mcmove_jax, estimate_energy_derivative
from functools import partial
from jax import lax

def ti_calc_jax(recovery, 
                error_probability=0.1, 
                d=7, 
                num_int_steps=51,
                mcmove_steps_init=500000, 
                mcmove_steps_for_derivative=1000,
                num_samples=1000, 
                rng_key=jax.random.PRNGKey(42), 
                animation_step=1):

    # 使用 initialize_jax_mc 初始化
    state, beta, rng_key, v_coup, h_coup, single, final_v, logical_op_row, derivative_v = initialize_jax_mc(recovery, d, error_probability, num_samples, rng_key)

    #输出横纵边框的数值
    h_borders = jnp.vstack([single[0,:], v_coup, single[-1,:]])
    v_borders = h_coup

    # 初始蒙特卡洛步骤
    state, rng_key, _, states_record = mcmove_jax(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, mcmove_steps_init)

    lambda_values = jnp.linspace(0., 1., num_int_steps)
    energy_derivatives_samples = jnp.zeros((num_int_steps, num_samples))

    for idx, l in enumerate(lambda_values):
        energy_derivatives, acceptance_rate, state, rng_key, states_temp, v_coup_temp = estimate_energy_derivative(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, final_v, derivative_v, logical_op_row, l, mcmove_steps_for_derivative)

        energy_derivatives_samples = energy_derivatives_samples.at[idx].set(energy_derivatives)

        print(f'l = {l:.3f}, energy derivative = {energy_derivatives.mean():.4f}, spin flip ratio = {acceptance_rate:.3f}')

        if idx == (animation_step-1):
            states_record=states_temp
            h_borders = jnp.vstack([single[0,:], v_coup_temp, single[-1,:]])


    integrals_for_each_sample = jnp.trapz(energy_derivatives_samples, x=lambda_values, axis=0) * beta

    mean_integral = integrals_for_each_sample.mean()
    error_bar = integrals_for_each_sample.std() / jnp.sqrt(num_samples)

    return mean_integral, error_bar, states_record, h_borders, v_borders

def ti_calc_jax_op(recovery, 
                error_probability=0.1, 
                d=7, 
                num_int_steps=51,
                mcmove_steps_init=500000, 
                mcmove_steps_for_derivative=1000,
                num_samples=1000, 
                rng_key=jax.random.PRNGKey(42)):

    # 初始化
    state, beta, rng_key, v_coup, h_coup, single, final_v, logical_op_row, derivative_v = initialize_jax_mc(recovery, d, error_probability, num_samples, rng_key)
    state, rng_key, _, _ = mcmove_jax(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, mcmove_steps_init)

    lambda_values = jnp.linspace(0., 1., num_int_steps)

    mean_integral, error_bar=energy_samples_integral(state, rng_key, beta, num_samples, d, v_coup, h_coup, single, final_v, derivative_v, logical_op_row, lambda_values, mcmove_steps_for_derivative, num_int_steps)

    return mean_integral, error_bar

@partial(jax.jit, static_argnums=(2,3,4,10,12,13))
def energy_samples_integral(state, rng_key, beta, num_samples, d, v_coup, h_coup, single, final_v, derivative_v, logical_op_row, lambda_values, mcmove_steps_for_derivative, num_int_steps):
    def scan_fn(carry, l):
        state, rng_key, acc_energy_derivatives = carry
        energy_derivatives, acceptance_rate, state, rng_key, _, _ = estimate_energy_derivative(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, final_v, derivative_v, logical_op_row, lambda_values[l], mcmove_steps_for_derivative)
        acc_energy_derivatives = acc_energy_derivatives.at[l].set(energy_derivatives)
        return (state, rng_key, acc_energy_derivatives), acceptance_rate

    initial_carry = (state, rng_key, jnp.zeros((num_int_steps, num_samples)))
    final_carry, _ = lax.scan(scan_fn, initial_carry, jnp.arange(num_int_steps))

    # 使用梯形规则计算积分
    integrals_for_each_sample = jnp.trapz(final_carry[2], x=lambda_values, axis=0) * beta

    # 计算平均值和误差
    mean_integral = integrals_for_each_sample.mean()
    error_bar = integrals_for_each_sample.std() / jnp.sqrt(num_samples)

    return mean_integral, error_bar

