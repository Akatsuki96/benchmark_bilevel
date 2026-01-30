import jax.numpy as jnp
import jax


def _compute_grad_ffd(forward_values, current_values, directions, h):
    b, l = directions.shape[:2]
    diff = (forward_values - current_values) / h
    grad_contrib = diff[..., None] * directions
    return jnp.sum(grad_contrib, axis=(0, 1)) / (b * l)

def _hvp_11(forward_values, backward_values, current_value, directions, v, h):
    b,l = directions.shape[:2]
    second_diff = (forward_values + backward_values - 2 * current_value) / (2.0 * h**2)  
    w_dot_v = jnp.sum(directions * v, axis=-1, keepdims=True)  
    hessian_block = w_dot_v * directions - v                
    grad_contrib = second_diff[..., None] * hessian_block  
    return jnp.sum(grad_contrib, axis=(0,1)) / (b * l)     

def _hvp_12(forward_values, backward_values, current_value, inner_directions, outer_directions, v, h):

    b, l = forward_values.shape

    second_diff = (forward_values + backward_values - 2 * current_value ) / (2.0 * h**2) 
    w_dot_v = jnp.sum(inner_directions * v, axis=-1, keepdims=True)  
    uv_product = outer_directions * w_dot_v                          
    grad_contrib = second_diff[..., None] * uv_product  
    return jnp.sum(grad_contrib, axis=(0,1)) / (b * l)  
    

def get_random_directions(key, b, l, dim):
    directions = jax.random.normal(key, shape=(b, l, dim))
    _, key = jax.random.split(key)
    return directions, key

def get_random_batch(key, b, n_samples):
    batch_indices = jax.random.randint(key, shape=(b,), minval=0, maxval=n_samples)
    _, key = jax.random.split(key)
    return batch_indices, key