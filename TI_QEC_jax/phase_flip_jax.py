import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from jax import lax


def convert_planar_bsf_to_ising_sign_jax(bsf, size):
    """
    Convert a phase flip planar code Pauli bsf to sign of coefficients in
    random bond Ising using JAX.

    params:
        bsf: planar code Pauli bsf
        size: size of planar code lattice in format (rows, columns), e.g. (5, 5).
    """
    bsf_jax = jnp.array(bsf)
    bsf_part = bsf_jax[bsf_jax.size // 2:]
    bsf_part = (-1) ** bsf_part
    v_sign = bsf_part[:size[0]*size[1]].reshape(size)
    h_sign = bsf_part[size[0] * size[1]:].reshape((size[0] - 1, size[1] - 1))

    single_sign = jnp.zeros((size[0]-1, size[1]))
    single_sign = single_sign.at[0,:].set(v_sign[0,:])
    single_sign = single_sign.at[-1, :].set(v_sign[-1, :])

    v_sign = v_sign[1:-1, :]

    return v_sign, h_sign, single_sign

def initialize_jax_mc(error_bsf, d=7, p=0.1, num_samples=32, rng_key=None):
    beta = float(jnp.log((1 - p) / p) / 2)

    rng_key = rng_key if rng_key is not None else random.PRNGKey(0)
    state = random.choice(rng_key, jnp.array([-1, 1]), shape=(num_samples, d - 1, d))

    init_v, init_h, init_single = convert_planar_bsf_to_ising_sign_jax(error_bsf, (d, d))

    final_v = jnp.copy(init_v)
    logical_op_row = d // 2
    final_v = final_v.at[logical_op_row, :].set(-final_v[logical_op_row, :])
    derivative_v = final_v - init_v
    derivative_v = derivative_v[logical_op_row,]

    return state, beta, rng_key, init_v, init_h, init_single, final_v, logical_op_row, derivative_v


@partial(jax.jit, static_argnums=(4,5,6,))
def step(state, interaction_key_a, interaction_key_b, flip_decision_key, beta, num_samples, d, v_coup, h_coup, single):
    a = random.randint(interaction_key_a, (num_samples,), 0, d - 1)
    b = random.randint(interaction_key_b, (num_samples,), 0, d)

    # Compute interaction energy for each simulation
    cost = single[a, b]
    cost += jnp.where(a > 0, v_coup[(a - 1)%d, b] * state[jnp.arange(num_samples), (a - 1)%d, b], 0)
    cost += jnp.where(a < d - 2, v_coup[a%(d - 2), b] * state[jnp.arange(num_samples), (a + 1)%(d-1), b], 0)
    cost += jnp.where(b > 0, h_coup[a, (b - 1)%d] * state[jnp.arange(num_samples), a, (b - 1)%d], 0)
    cost += jnp.where(b < d - 1, h_coup[a, b%(d - 1)] * state[jnp.arange(num_samples), a, (b + 1)%d], 0)
    cost *= 2 * state[jnp.arange(num_samples), a, b]

    # Update spins based on Metropolis criterion
    uniform_random = random.uniform(flip_decision_key, (num_samples,))
    flip = (cost < 0) | (uniform_random < jnp.exp(-beta * cost))
    state = state.at[jnp.arange(num_samples), a, b].set(jnp.where(flip, -state[jnp.arange(num_samples), a, b], state[jnp.arange(num_samples), a, b]))

    # Calculate acceptance rate
    acceptance_rate = jnp.mean(flip.astype(jnp.float32))

    return state, acceptance_rate

@partial(jax.jit, static_argnums=(1,2,3,8,))
def mcmove_jax(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, num_steps=100):
    keys = random.split(rng_key, 3 * num_steps + 1)
    rng_key = keys[0]
    interaction_keys_a = keys[1::3]
    interaction_keys_b = keys[2::3]
    flip_decision_keys = keys[3::3]

    def scan_fn(carry, keys):
        state, total_acceptance = carry
        key_a, key_b, flip_decision_key = keys  # 解包当前步骤的三个键
        state, acceptance_rate = step(
            state, key_a, key_b, flip_decision_key, 
            beta, num_samples, d, v_coup, h_coup, single
        )
        total_acceptance += acceptance_rate
        return (state, total_acceptance), state[0]

    initial_carry = (state, 0.0)
    final_carry, states_evo = lax.scan(
        scan_fn, 
        initial_carry, 
        (interaction_keys_a, interaction_keys_b, flip_decision_keys), 
        length=num_steps
    )

    flip_acceptance = final_carry[1] / num_steps
    return final_carry[0], rng_key, flip_acceptance, states_evo

@partial(jax.jit, static_argnums=(1,2,3,10,12,))    
def estimate_energy_derivative(state, beta, num_samples, d, rng_key, v_coup, h_coup, single, final_v, derivative_v, logical_op_row, l, mcmove_steps=1000):
    # 设置耦合基于当前 lambda 值
    v_coup_current = l * final_v + (1 - l) * v_coup  # 调整以反映您的具体耦合设置

    # 使用 mcmove 更新状态
    state, rng_key, flip_acceptance_rate, states_record = mcmove_jax(state, beta, num_samples, d, rng_key, v_coup_current, h_coup, single, mcmove_steps)

    # 定义一个局部函数，用于执行能量导数的计算
    def compute_energy_derivative(state, derivative_v_fixed, logical_op_row_fixed):
        # 计算每个模拟的能量导数
        energy_derivatives = jnp.sum(derivative_v_fixed * state[:, logical_op_row_fixed, :] *
                                     state[:, logical_op_row_fixed + 1, :], axis=1)
        return energy_derivatives

    # 计算并返回能量导数列表
    energy_derivatives = compute_energy_derivative(state, derivative_v, logical_op_row)

    # 返回能量导数列表和接受率
    return energy_derivatives, flip_acceptance_rate, state, rng_key, states_record, v_coup_current



