from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """Zeroth-order SOBA"""
    name = 'ZOBA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [0.005],#,0.001], # stepsize for z,v
        'outer_ratio': [1.0],#, 2.0], # stepsize for x => stepsize / outer_ratio
        'h' : [1e-3], # smoothing parameter eta = h 
        'l1' : [25], # number of directions for outer gradient approximation
        'l2' : [50], # number of directions for inner gradient/hessians
        'h_outer' : [1e-4], #[1e-3],
        'zero_outer_smoothing' : [0],#, 1],
#        'batch_size': [64],
        'batch_size': [1],
        **StochasticJaxSolver.parameters
    }

    def init(self):
        # Init variables
        self.inner_var = self.inner_var0.copy()
        self.outer_var = self.outer_var0.copy()
        self.rnd_state_key = jax.random.PRNGKey(0)
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

        def _compute_grad_ffd(forward_values, current_values, directions, h):
            return jnp.sum(((forward_values - current_values) / h)[:,None] * directions, axis=0) / directions.shape[0]

        def _hvp_11(forward_values, backward_values, current_value, directions, v):
            bi = (forward_values + backward_values - 2 * current_value) / (2 * self.h * self.h) 
            proj = directions @ v
            term = directions * proj[:, None]              
            return jnp.sum(bi[:, None] * (term - v), axis=0) / directions.shape[0]


        def _hvp_12(forward_values, backward_values, current_value, inner_directions, outer_directions, w):
            
            bi = (forward_values + backward_values - 2 * current_value) / (self.h * self.h) 
            proj = outer_directions @ w 
            return jnp.sum(bi[:, None] * inner_directions * proj[:, None], axis=0) / inner_directions.shape[0]


        ffd = jax.jit(_compute_grad_ffd)
        hvp_b1 = jax.jit(_hvp_11)
        hvp_cross = jax.jit(_hvp_12)


        def zoba_one_iter(carry, _):

            _, subkey_inner = jax.random.split(carry['rnd_state_key'])
            _, subkey_outer = jax.random.split(subkey_inner)

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )


            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )


            l_max = max(self.l1, self.l2)
            # Build direction matrices
            inner_directions = jax.random.normal(subkey_inner, shape=(l_max, carry['inner_var'].shape[0]))
            outer_directions = jax.random.normal(subkey_outer, shape=(l_max, carry['outer_var'].shape[0]))


            # Current function values
            current_f_inner = self.f_inner(carry['inner_var'], carry['outer_var'], start_inner)
            current_f_outer = self.f_outer(carry['inner_var'], carry['outer_var'], start_outer)

            # Compute function values for building gradients and hessians surrogates (for g_1 and H_xx)
            f_plus_values_inner   = jax.vmap(lambda x : self.f_inner(x, carry['outer_var'], start_inner))(carry['inner_var'].reshape(1, -1) + self.h * inner_directions)
            f_minus_values_inner  = jax.vmap(lambda x : self.f_inner(x, carry['outer_var'], start_inner))(carry['inner_var'].reshape(1, -1) - self.h * inner_directions)

            ## For hessian cross block
            f_plus_H  = jax.vmap(lambda z,x : self.f_inner(z,x, start_inner))(carry['inner_var'].reshape(1, -1) + self.h * inner_directions, carry['outer_var'].reshape(1, -1) + self.h * outer_directions)
            f_minus_H = jax.vmap(lambda z,x : self.f_inner(z,x, start_inner))(carry['inner_var'].reshape(1, -1) - self.h * inner_directions, carry['outer_var'].reshape(1, -1) - self.h * outer_directions)

            ## For g_f,1 and g_f,2

            if self.zero_outer_smoothing == 1:
                f_plus_values_outer_1 = jax.vmap(lambda x : self.f_outer(x, carry['outer_var'], start_outer))(carry['inner_var'].reshape(1, -1) + self.h_outer * inner_directions[:self.l1, :])
                f_plus_values_outer_2 = jax.vmap(lambda x : self.f_outer(carry['inner_var'], x, start_outer))(carry['outer_var'].reshape(1, -1) + self.h_outer * outer_directions[:self.l1, :])

            else:
                f_plus_values_outer_1 = jax.vmap(lambda z,x : self.f_outer(z,x, start_outer))(carry['inner_var'].reshape(1, -1) + self.h_outer * inner_directions[:self.l1, :], carry['outer_var'].reshape(1, -1) + self.h_outer * outer_directions[:self.l1, :])
#            f_plus_values_outer_2 = jax.vmap(lambda x : self.f_outer(carry['inner_var'], x, start_outer))(carry['inner_var'].reshape(1, -1) + self.h * inner_directions[:self.l1, :], carry['outer_var'].reshape(1, -1) + self.h * outer_directions[:self.l1, :])


            # Comupte gradient approximations of inner and outer functions
            g_inner   = ffd(f_plus_values_inner, f_minus_values_inner, inner_directions, 2*self.h)
            g_outer_1 = ffd(f_plus_values_outer_1, current_f_outer, inner_directions[:self.l1, :], self.h_outer)

            if self.zero_outer_smoothing == 1:
                g_outer_2 = ffd(f_plus_values_outer_2, current_f_outer, outer_directions[:self.l1, :], self.h_outer)
            else:
                # g_outer_1 = ffd(f_plus_values_outer_1, current_f_outer, inner_directions[:self.l1, :], self.h)
                g_outer_2 = ffd(f_plus_values_outer_1, current_f_outer, outer_directions[:self.l1, :], self.h_outer)
            # Compute Hessian-vector products


            Hvp_11 = hvp_b1(f_plus_values_inner, f_minus_values_inner, current_f_inner, inner_directions, carry['v'])

            Hvp_12 = hvp_cross(f_plus_H, f_minus_H, current_f_inner, inner_directions, outer_directions, carry['v'])


            carry['inner_var'] -= inner_step_size * g_inner
            carry['v'] -= inner_step_size * (Hvp_11 + g_outer_1)
            carry['outer_var'] -= outer_step_size * (Hvp_12 + g_outer_2)

            carry['rnd_state_key'] = subkey_outer
            jax.debug.print("Function Value = {}", self.f_outer(carry['inner_var'], carry['outer_var']))

            return carry, _


        return zoba_one_iter
