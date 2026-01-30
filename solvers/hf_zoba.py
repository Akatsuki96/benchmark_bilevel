from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr, init_lr_scheduler
    from benchmark_utils.zeroth_order_utils import _compute_grad_ffd, get_random_directions, get_random_batch

    import jax
    import jax.numpy as jnp
    
    



class Solver(StochasticJaxSolver):
    """Hessian-free Zeroth-Order Bilevel Algorithm (HF-ZOBA).

    M. Rando and S. Vaiter, "ZOBA: An Efficient Single-loop Zeroth-order 
    Bilevel Optimization Algorithm", ArXiv 2026."""
    name = 'HF-ZOBA'


    parameters = {
        'step_size': [0.01], # stepsize for z,v
        'outer_ratio': [2.0], # stepsize for x => stepsize / outer_ratio
        'h' : [1e-3], # finite difference discretization parameter 
        'h_bar' : [1e-3], # finite difference discretization parameter for hessian-vector products approximation
        'l1' : [100], # number of directions for inner gradient/hessians approximations
        'l2' : [1], # number of directions for outer gradient
        'b1' : [25], # batch size for inner function evaluations
        'b2' : [10], # batch size for outer function evaluations
        **StochasticJaxSolver.parameters
    }
    batch_size = 1

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        self.rnd_state_key = jax.random.PRNGKey(0)
        self.ffd = jax.jit(_compute_grad_ffd)

        v = jnp.zeros_like(self.inner_var)

        # Init lr scheduler
        step_sizes = jnp.array(
            [self.step_size, self.step_size / self.outer_ratio]
        )
        exponents = jnp.array(
            [0.5, 0.5]
        )
        state_lr = init_lr_scheduler(step_sizes, exponents)
        return dict(
            inner_var=self.inner_var, outer_var=self.outer_var, v=v,
            state_lr=state_lr, rnd_state_key=self.rnd_state_key,
            state_inner_sampler=self.state_inner_sampler,
            state_outer_sampler=self.state_outer_sampler,
        )


    def get_step(self, inner_sampler, outer_sampler):





        def zoba_one_iter(carry, _):

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            l_max = max(self.l1, self.l2)
            b_max = max(self.b1, self.b2)

            # Sample directions and batches

            inner_directions, subkey = get_random_directions(carry['rnd_state_key'], b_max, l_max, carry['inner_var'].shape[0])
            outer_directions, subkey = get_random_directions(subkey, b_max, l_max, carry['outer_var'].shape[0])

            inner_samples, subkey = get_random_batch(subkey, self.b1, self.n_inner_samples)
            outer_samples, subkey = get_random_batch(subkey, self.b2, self.n_outer_samples)
            
            # Define function mapping for inner target

            fz_values_inner   = jax.vmap(jax.vmap(lambda x, xi : self.f_inner(x, carry['outer_var'], start=xi), in_axes=(0, None)), in_axes=(0, 0))
            fx_values_inner   = jax.vmap(jax.vmap(lambda x, xi : self.f_inner(carry['inner_var'], x, start=xi), in_axes=(0, None)), in_axes=(0, 0))

            fz_values_outer   = jax.vmap(jax.vmap(lambda x, zeta : self.f_outer(x, carry['outer_var'], start=zeta), in_axes=(0, None)), in_axes=(0, 0))
            fx_values_outer   = jax.vmap(jax.vmap(lambda x, zeta : self.f_outer(carry['inner_var'], x, start=zeta), in_axes=(0, None)), in_axes=(0, 0))

            fxv_values_inner   = jax.vmap(jax.vmap(lambda x, xi : self.f_inner(carry['inner_var'] + self.h_bar * carry['v'], x, start=xi), in_axes=(0, None)), in_axes=(0, 0))
            
            # Compute function values in current iterates for every sample i.e. g(z_k, x_k, xi_i) and f(z_k, x_k, zeta_i)
            
            current_f_inner = jax.vmap(lambda xi_i : self.f_inner(carry['inner_var'], carry['outer_var'], start=xi_i), in_axes=(0))(inner_samples)
            current_f_outer = jax.vmap(lambda zeta_i : self.f_outer(carry['inner_var'], carry['outer_var'], start=zeta_i), in_axes=(0))(outer_samples)

            inner_plus = carry['inner_var'] + self.h * inner_directions
            outer_plus = carry['outer_var'] + self.h * outer_directions


            inner_z_plus, inner_x_plus = inner_plus[:self.b1, :self.l1, :], outer_plus[:self.b1, :self.l1, :]            
            outer_plus_z, outer_plus_x = inner_plus[:self.b2, :self.l2, :], outer_plus[:self.b2, :self.l2, :]


            # Compute g(z_k + h w_ij, x_k, xi_i) and g(z_k - h w_ij, x_k, xi_i) for i in batch and j in directions

            fz_plus_inner   = fz_values_inner(inner_z_plus, inner_samples)
            fx_plus_inner   = fx_values_inner(inner_x_plus, inner_samples)


            # Compute f(z_k + h w_ij, x_k, zeta_i) and f(z_k, x_k + h u_ij, zeta_i) for i in batch and j in directions

            fz_plus_outer = fz_values_outer(outer_plus_z, outer_samples)
            fx_plus_outer = fx_values_outer(outer_plus_x, outer_samples)


            # Compute g(z_k + h_bar v_k + h w_ij, x_k, xi_i) and g(z_k + h_bar v_k, x_k, xi_i)
            fv_z_plus_inner = fz_values_inner(carry['inner_var'] + self.h_bar * carry['v'] + self.h * inner_directions[:self.b1, :self.l1, :], inner_samples)
            fv_z_current_inner = jax.vmap(lambda xi_i : self.f_inner(carry['inner_var'] + self.h_bar * carry['v'], carry['outer_var'], start=xi_i), in_axes=(0))(inner_samples)
            fxv_plus_inner = fxv_values_inner(carry['outer_var'] + self.h * outer_directions[:self.b1, :self.l1, :] , inner_samples)
            

            # Compute gradients approximations
            grad_inner_z_v = self.ffd(fv_z_plus_inner, fv_z_current_inner[:, None], inner_directions[:self.b1, :self.l1, :], self.h)
            grad_inner_x_v = self.ffd(fxv_plus_inner,  fv_z_current_inner[:, None], outer_directions[:self.b1, :self.l1, :], self.h)
            
            grad_inner_z = self.ffd(fz_plus_inner, current_f_inner[:, None], inner_directions[:self.b1, :self.l1, :], self.h)
            grad_inner_x = self.ffd(fx_plus_inner, current_f_inner[:, None], outer_directions[:self.b1, :self.l1, :], self.h)

            grad_z_outer = self.ffd(fz_plus_outer, current_f_outer[:, None], inner_directions[:self.b2, :self.l2, :], self.h)
            grad_x_outer = self.ffd(fx_plus_outer, current_f_outer[:, None], outer_directions[:self.b2, :self.l2, :], self.h)
            
            # Compute search directions
            D_z = grad_inner_z
            D_v = ((grad_inner_z_v - grad_inner_z) / self.h_bar) + grad_z_outer
            D_x = ((grad_inner_x_v - grad_inner_x)/self.h_bar) + grad_x_outer

            # Update iterates

            carry['inner_var'] -= inner_step_size * D_z
            carry['v'] -= inner_step_size * D_v
            carry['outer_var'] -= outer_step_size * D_x
            
            # Update random key

            carry['rnd_state_key'] = subkey
            jax.debug.print("Function Value = {}", self.f_outer(carry['inner_var'], carry['outer_var']))

            return carry, _


        return zoba_one_iter
