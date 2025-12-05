from benchmark_utils.stochastic_jax_solver import StochasticJaxSolver

from benchopt import safe_import_context

from math import sqrt

with safe_import_context() as import_ctx:
    from benchmark_utils.learning_rate_scheduler import update_lr
    from benchmark_utils.learning_rate_scheduler import init_lr_scheduler

    import jax
    import jax.numpy as jnp


class Solver(StochasticJaxSolver):
    """ZMDSBA"""
    name = 'ZMDSBA'

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'step_size': [0.00001], # beta_k
        'outer_ratio': [1.0], # alpha_k
        'beta_0' : [0.0005],
        'gamma_0' : [0.0005], # stepsize for hessian inverse approximation
        'eta_1' : [1e-3], # smoothing outer 1
        'eta_2' : [1e-3], # smoothing outer 2
        'mu_1' : [1e-3], # smoothing inner 1
        'mu_2' : [1e-3], # smoothing inner 2
        's_k' : [200], # number of directions for cross hessian block and outer gradient
        't_k' : [20], # number of iteration of inner loop
        'b_k' : [1], # number of iteration for computing inverse of hessian
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

    def _get_inner_sol_approx(self, inner_var, outer_var, inner_sampler, inner_state_sampler, rnd_state_key):

        for t in range(self.t_k):
            start_inner, *_, inner_state_sampler = inner_sampler(
                inner_state_sampler
            )
            f_current = self.f_inner(inner_var, outer_var, start_inner)
            v = jax.random.normal(rnd_state_key, shape=(inner_var.shape[0],))
            _, rnd_state_key = jax.random.split(rnd_state_key)
            f_next = self.f_inner(inner_var + self.mu_2 * v, outer_var, start_inner) 
            g = ((f_next - f_current)/self.mu_2) * v
            inner_var -= (self.beta_0 / sqrt(t + 1)) * g
        return inner_var, inner_state_sampler, rnd_state_key


    def _szhia(self, v0, inner_var, outer_var, inner_sampler, inner_state_sampler, outer_sampler, outer_state_sampler, rnd_state_key):

        # x, y_upd, eta_1, eta_2, mu_1, mu_2, b_k
        v0 = jnp.zeros((outer_var.shape[0],))
        for t in range(self.b_k):
            # STEP 1: sample batches
            start_inner, *_, inner_state_sampler = inner_sampler(inner_state_sampler)
            start_outer, *_, outer_state_sampler = outer_sampler(outer_state_sampler)
            # STEP 2: sample inner/outer directions 
            # -> for hessian
            v1 = jax.random.normal(rnd_state_key, shape=(inner_var.shape[0],)) # direction for inner
            _, rnd_state_key = jax.random.split(rnd_state_key)
            v2 = jax.random.normal(rnd_state_key, shape=(outer_var.shape[0],)) # direction for outer
            _,rnd_state_key = jax.random.split(rnd_state_key)
            # -> for gradient 
            u1 = jax.random.normal(rnd_state_key, shape=(inner_var.shape[0],)) # direction for inner
            _,rnd_state_key = jax.random.split(rnd_state_key)
            u2 = jax.random.normal(rnd_state_key, shape=(outer_var.shape[0],)) # direction for outer
            _,rnd_state_key = jax.random.split(rnd_state_key)

            # STEP 3: compute direction
            f_inner_plus    = self.f_inner(inner_var  + self.mu_2 *v1, outer_var + self.eta_2 * v2, start_inner)
            f_inner_minus   = self.f_inner(inner_var - self.mu_2 *v1, outer_var - self.eta_2 * v2, start_inner)
            f_inner_current = self.f_inner(inner_var, outer_var, start_inner)

            f_outer_plus    = self.f_outer(inner_var + self.mu_1 * u1, outer_var + self.eta_1 *u2, start_outer)
            f_outer_current = self.f_outer(inner_var, outer_var, start_outer)

            bi = (f_inner_plus + f_inner_minus - 2*f_inner_current) / (2 * self.mu_2*self.mu_2)
            g = ((jnp.linalg.outer(v1, v1) - jnp.eye(v1.shape[0])) *bi) @ v0  - ((f_outer_plus - f_outer_current) / self.mu_1) * u1 
            v0 -= (self.gamma_0 / sqrt(t + 1)) * g
        return v0, inner_state_sampler, outer_state_sampler, rnd_state_key




    def get_step(self, inner_sampler, outer_sampler):


        def zdsba_one_iter(carry, _):

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # STEP 1: Compute approximate solution of inner problem
            carry['inner_var'], carry['state_inner_sampler'], carry['rnd_state_key'] = self._get_inner_sol_approx(carry['inner_var'], carry['outer_var'], inner_sampler, carry['state_inner_sampler'], carry['rnd_state_key'])


            # STEP 2: Compute gradient approximation of outer target
            g_outer = jnp.zeros_like(carry['outer_var'])
            H_cross = jnp.zeros((carry['inner_var'].shape[0], carry['outer_var'].shape[0]))
            for i in range(self.s_k):
                start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                    carry['state_outer_sampler']
                )

                v = jax.random.normal(carry['rnd_state_key'], shape=(carry['outer_var'].shape[0],))
                _,carry['rnd_state_key'] = jax.random.split(carry['rnd_state_key'])
                f_outer_current = self.f_outer(carry['inner_var'], carry['outer_var'], start_outer)
                f_outer_next = self.f_outer(carry['inner_var'], carry['outer_var'] + self.eta_1 * v, start_outer)
                g_outer += ((f_outer_next - f_outer_current)/ self.eta_1) * v

                # STEP 3: Compute cross-hessian approximation
                cross_hess_sample, *_, carry['state_inner_sampler'] = inner_sampler(carry['state_inner_sampler'])

                v1 = jax.random.normal(carry['rnd_state_key'], shape=(carry['inner_var'].shape[0],)) # direction for inner
                _,carry['rnd_state_key'] = jax.random.split(carry['rnd_state_key'])
                v2 = jax.random.normal(carry['rnd_state_key'], shape=(carry['outer_var'].shape[0],)) # direction for outer
                _,carry['rnd_state_key'] = jax.random.split(carry['rnd_state_key'])


                f_inner_current = self.f_inner(carry['inner_var'], carry['outer_var'], cross_hess_sample)
                f_plus_H   = self.f_inner(carry['inner_var']  + self.mu_2 * v1,  carry['outer_var'] + self.eta_2 * v2, cross_hess_sample)
                f_minus_H  = self.f_inner(carry['inner_var']  - self.mu_2 * v1,  carry['outer_var'] - self.eta_2 * v2, cross_hess_sample)

                H_cross += ((f_plus_H + f_minus_H - 2 * f_inner_current) / (self.eta_2 * self.mu_2)) * jnp.outer(v1, v2)

            g_outer /= self.s_k
            H_cross /= self.s_k
            # STEP 4: Compute inverse of block-hessian times gradient of outer 

            carry['v'], carry['state_inner_sampler'], carry['state_outer_sampler'], carry['rnd_state_key'] = self._szhia(carry['v'], carry['inner_var'], carry['outer_var'], inner_sampler, carry['state_inner_sampler'], outer_sampler, carry['state_outer_sampler'], carry['rnd_state_key'])  

            # STEP 5: Update outer variable
            carry['outer_var'] -= outer_step_size * (g_outer - H_cross @ carry['v'])
            jax.debug.print("Function Value = {}", self.f_outer(carry['inner_var'], carry['outer_var']))

            return carry, _


        return zdsba_one_iter
