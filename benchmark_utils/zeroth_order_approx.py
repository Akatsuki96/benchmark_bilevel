import jax.numpy as jnp
import jax


def _compute_grad_ffd(forward_values, current_values, directions, h):
    b, l = directions.shape[:2]
    diff = (forward_values - current_values) / h
    grad_contrib = diff[..., None] * directions
    return jnp.sum(grad_contrib, axis=(0, 1)) / (b * l)
#    return jnp.sum(((forward_values - current_values) / h)[:,None] * directions, axis=0) / directions.shape[0]

def _hvp_11(forward_values, backward_values, current_value, directions, v, h):
    b,l = directions.shape[:2]
    second_diff = (forward_values + backward_values - 2 * current_value) / (2.0 * h**2)  
    w_dot_v = jnp.sum(directions * v, axis=-1, keepdims=True)  
    hessian_block = w_dot_v * directions - v                
    grad_contrib = second_diff[..., None] * hessian_block  
    return jnp.sum(grad_contrib, axis=(0,1)) / (b * l)     

    # bi = (forward_values + backward_values - 2 * current_value) / (2 * h **2) 
    # proj = directions @ v
    # term = directions * proj[:, None]              
    # return jnp.sum(bi[:, None] * (term - v), axis=0) / directions.shape[0]


def _hvp_12(forward_values, backward_values, current_value, inner_directions, outer_directions, v, h):

    b, l = forward_values.shape

    second_diff = (forward_values + backward_values - 2 * current_value ) / (2.0 * h**2) 
    w_dot_v = jnp.sum(inner_directions * v, axis=-1, keepdims=True)  
    uv_product = outer_directions * w_dot_v                          
    grad_contrib = second_diff[..., None] * uv_product  
    return jnp.sum(grad_contrib, axis=(0,1)) / (b * l)  
    
    # bi = (forward_values + backward_values - 2 * current_value) / (2 * h **2) 
    # proj = outer_directions @ w 
    # return jnp.sum(bi[:, None] * inner_directions * proj[:, None], axis=0) / inner_directions.shape[0]
