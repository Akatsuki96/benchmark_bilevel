from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context
from math import sqrt
with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """Zeroth-order SOBA"""
    name = 'OptZMDSBA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [0.0025],#,0.001], # stepsize for z,v
        'outer_ratio': [1.0],#, 2.0], # stepsize for x => stepsize / outer_ratio
        'l' : [25], # number of directions for outer gradient approximation
        
        'h_outer' : [1e-3], #[1e-3],
        'tb' : [5], # num inner iterations
        'mu' : [1e-3],
        'beta_0' : [0.0001],
        'lam' : [10.0],

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
            return jnp.sum(((forward_values - current_values) / h)[:, None] * directions, axis=0) / directions.shape[0]
#            return  ((forward_values - current_values) / h) * directions / directions.shape[0]

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


        def optzmdsba_one_iter(carry, _):

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



            # Build direction matrices
            for t in range(self.tb):
                _, subkey_inner = jax.random.split(subkey_inner)
                _, subkey_outer = jax.random.split(subkey_outer)

                start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                    carry['state_inner_sampler']
                )
                start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                    carry['state_outer_sampler']
                )

                inner_directions = jax.random.normal(subkey_inner, shape=(1, carry['inner_var'].shape[0]))
                current_f_inner_y = self.f_inner(carry['inner_var'], carry['outer_var'], start_inner)
                current_f_inner_z = self.f_inner(carry['v'], carry['outer_var'], start_inner)
                current_f_outer_z = self.f_outer(carry['v'], carry['outer_var'], start_outer)


#                next_f_inner_y = self.f_inner(carry['inner_var'] + self.mu * inner_directions, carry['outer_var'], start_inner)
                next_f_inner_y = jax.vmap(lambda x : self.f_inner(x, carry['outer_var'], start_inner))(carry['inner_var'].reshape(1, -1) + self.mu * inner_directions)
                next_f_inner_z = jax.vmap(lambda x : self.f_inner(x, carry['outer_var'], start_inner))(carry['v'].reshape(1, -1) + self.mu * inner_directions)
                next_f_outer_z = jax.vmap(lambda x : self.f_outer(x, carry['outer_var'], start_outer))(carry['v'].reshape(1, -1) + self.mu * inner_directions)

                g_inner_y = ffd(next_f_inner_y, current_f_inner_y, inner_directions, self.mu)
                g_inner_z = ffd(next_f_inner_z, current_f_inner_z, inner_directions, self.mu)
                g_outer_z = ffd(next_f_outer_z, current_f_outer_z, inner_directions, self.mu)

#                jax.debug.print("V SHAPE : {} G SHAPE: {}", carry['v'].shape, g_inner_z.shape)
                carry['v'] -= (self.beta_0/sqrt(t + 1)) * g_inner_z.flatten()
                carry['inner_var'] -= (self.beta_0 / sqrt(t + 1)) * ((1/self.lam) * g_outer_z + g_inner_y).flatten()

            outer_directions = jax.random.normal(subkey_outer, shape=(self.l, carry['outer_var'].shape[0]))

            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )
            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )

            # Current function values
            current_f_outer = self.f_outer(carry['inner_var'], carry['outer_var'], start_outer)
            current_f_inner_y = self.f_inner(carry['inner_var'], carry['outer_var'], start_inner)
            current_f_inner_z = self.f_inner(carry['v'], carry['outer_var'], start_inner)


            next_f_inner_y = jax.vmap( lambda x : self.f_inner(carry['inner_var'], x, start_inner))(carry['outer_var'].reshape(1, -1)+ self.mu * outer_directions)
            next_f_inner_z = jax.vmap( lambda x : self.f_inner(carry['v'], x, start_inner))(carry['outer_var'].reshape(1, -1)+ self.mu * outer_directions)
            next_f_outer_z = jax.vmap( lambda x : self.f_outer(carry['v'], x, start_outer))(carry['outer_var'].reshape(1, -1)+ self.mu * outer_directions)

            g_outer = ffd(next_f_outer_z, current_f_outer, outer_directions, self.mu).flatten()
            g_inner_y = ffd(next_f_inner_y, current_f_inner_y, outer_directions, self.mu).flatten()
            g_inner_z = ffd(next_f_inner_z, current_f_inner_z, outer_directions, self.mu).flatten()


            carry['outer_var'] -= outer_step_size * (g_outer + self.lam * (g_inner_z - g_inner_y))

            carry['rnd_state_key'] = subkey_outer
            jax.debug.print("Function Value = {}", self.f_outer(carry['inner_var'], carry['outer_var']))

            return carry, _


        return optzmdsba_one_iter
