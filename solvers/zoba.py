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
        'step_size': [0.01],
        'outer_ratio': [1.0],
        'h' : [1e-3],
        'l1' : [10],
        'l2' : [25],
        'batch_size': [64],
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

        grad_inner = jax.grad(self.f_inner, argnums=0)
        grad_outer = jax.grad(self.f_outer, argnums=(0, 1))


        def _compute_grad_ffd(forward_values, current_values, directions):
            return jnp.sum(((forward_values - current_values) / self.h)[:,None] * directions, axis=0) / directions.shape[0]

        # def _compute_hess_block_1(forward_values, backward_values, current_value, directions):

        #     bi = (forward_values + backward_values - 2 * current_value) / (2 * self.h * self.h) 
        #     outer_terms = directions[:, :, None] * directions[:, None, :]

        #     H_b1 = jnp.einsum('n,nij->ij', bi, outer_terms - jnp.eye(directions.shape[1]))
        #     H_b1 /= directions.shape[0]

        #     # H = jnp.zeros((directions.shape[1], directions.shape[1]))

        #     # for i in range(directions.shape[0]):
        #     #     bi = (forward_values[i] + backward_values[i] - 2 * current_value) / (2 * self.h * self.h)
        #     #     H += bi * (jnp.outer(directions[i], directions[i]) - jnp.eye(directions.shape[1]))

        #     # H /= directions.shape[0]
        #     # jax.debug.print("IS SAME APPROX: {}", jnp.linalg.norm(H - H1, ord=2))

        #     return H_b1

        def _hvp_11(forward_values, backward_values, current_value, directions, v):
            bi = (forward_values + backward_values - 2 * current_value) / (2 * self.h * self.h) 
            proj = directions @ v
            term = directions * proj[:, None]              
            return jnp.sum(bi[:, None] * (term - v), axis=0) / directions.shape[0]


        def _hvp_12(forward_values, backward_values, current_value, inner_directions, outer_directions, w):
            
            bi = (forward_values + backward_values - 2 * current_value) / (self.h * self.h) 
            proj = outer_directions @ w 
            return jnp.sum(bi[:, None] * inner_directions * proj[:, None], axis=0) / inner_directions.shape[0]


        
        # def _compute_hess_cross_block(forward_values, backward_values, current_value, inner_directions, outer_directions):

        #     # scalar weights for each sample
        #     bi = (forward_values + backward_values - 2 * current_value) / (self.h * self.h)  # shape (N,)

        #     # Outer products for all samples: shape (N, d_in, d_out)
        #     outer_terms = inner_directions[:, :, None] * outer_directions[:, None, :]

        #     # Weighted sum across samples
        #     H_cross = jnp.einsum('n,nij->ij', bi, outer_terms)

        #     # Average across directions
        #     H_cross /= inner_directions.shape[0]
        #     # H = jnp.zeros((inner_directions.shape[1], outer_directions.shape[1]))

        #     # for i in range(inner_directions.shape[0]):
        #     #     bi = (forward_values[i] + backward_values[i] - 2 * current_value) / (self.h * self.h)
        #     #     H += bi * jnp.outer(inner_directions[i], outer_directions[i])
        #     #     #jax.debug.print("[{}] H : {}", i, H)
        #     # H /= inner_directions.shape[0]
        #     # jax.debug.print("IS SAME APPROX: {}", jnp.linalg.norm(H - H_cross, ord=2))
        #     return H_cross


        ffd = jax.jit(_compute_grad_ffd)
        hvp_b1 = jax.jit(_hvp_11)
        hvp_cross = jax.jit(_hvp_12)


        def soba_one_iter(carry, _):

            _, subkey_inner = jax.random.split(carry['rnd_state_key'])
            _, subkey_outer = jax.random.split(subkey_inner)

            (inner_step_size, outer_step_size), carry['state_lr'] = update_lr(
                carry['state_lr']
            )

            # Step.1 - get all gradients and compute the implicit gradient.
            start_inner, *_, carry['state_inner_sampler'] = inner_sampler(
                carry['state_inner_sampler']
            )

            start_outer, *_, carry['state_outer_sampler'] = outer_sampler(
                carry['state_outer_sampler']
            )



        #     grad_inner_var, vjp_train = jax.vjp(
        #         lambda z, x: grad_inner(z, x, start_inner), carry['inner_var'],
        #         carry['outer_var']
        #     )
        #     hvp, cross_v = vjp_train(carry['v'])
        #     grad_in_outer, grad_out_outer = grad_outer(
        #        carry['inner_var'], carry['outer_var'], start_outer
        #    )

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
            f_plus_values_outer_1 = jax.vmap(lambda x : self.f_outer(x, carry['outer_var'], start_outer))(carry['inner_var'].reshape(1, -1) + self.h * inner_directions[:self.l1, :])
            f_plus_values_outer_2 = jax.vmap(lambda x : self.f_outer(carry['inner_var'], x, start_outer))(carry['outer_var'].reshape(1, -1) + self.h * outer_directions[:self.l1, :])


            # Comupte gradient approximations of inner and outer functions
            g_inner   = ffd(f_plus_values_inner[:self.l1], current_f_inner, inner_directions[:self.l1, :])
            g_outer_1 = ffd(f_plus_values_outer_1, current_f_outer, inner_directions[:self.l1, :])
            g_outer_2 = ffd(f_plus_values_outer_2, current_f_outer, outer_directions[:self.l1, :])


            # Compute Hessian-vector products
#            H_block_11 = hess_b1(f_plus_values_inner, f_minus_values_inner, current_f_inner, inner_directions) @ carry['v']

            Hvp_11 = hvp_b1(f_plus_values_inner, f_minus_values_inner, current_f_inner, inner_directions, carry['v'])



#            H_cross = hess_bcross(f_plus_H, f_minus_H, current_f_inner, inner_directions, outer_directions) @ carry['v']

            Hvp_12 = hvp_cross(f_plus_H, f_minus_H, current_f_inner, inner_directions, outer_directions, carry['v'])

            # jax.debug.print("Implicit vs Exp: {}", jnp.linalg.norm(Hvp_12 - H_cross))
            # jax.debug.print("HB 1 error: {}", jnp.linalg.norm(H_block_11 - hvp) / jnp.linalg.norm(hvp))
            # jax.debug.print("HB imp 1 error: {}", jnp.linalg.norm(H_block_11_vp - hvp) / jnp.linalg.norm(hvp))
            # jax.debug.print("HB Cross error: {}", jnp.linalg.norm(H_cross - cross_v) / jnp.linalg.norm(cross_v))
            # jax.debug.print("HB Cross Impl error: {}", jnp.linalg.norm(Hvp_12 - cross_v) / jnp.linalg.norm(cross_v))
            # jax.debug.print("Inner grad error: {}", jnp.linalg.norm(g_inner - grad_inner_var) / jnp.linalg.norm(grad_inner_var))
            # jax.debug.print("Grad in outer error: {}", jnp.linalg.norm(g_outer_1 - grad_in_outer) / jnp.linalg.norm(grad_in_outer))
            # jax.debug.print("Grad out outer error: {}", jnp.linalg.norm(g_outer_2 - grad_out_outer) / jnp.linalg.norm(grad_out_outer))

            carry['inner_var'] -= inner_step_size * g_inner
            carry['v'] -= inner_step_size * (Hvp_11 + g_outer_1)
            carry['outer_var'] -= outer_step_size * (Hvp_12 + g_outer_2)

            carry['rnd_state_key'] = subkey_outer
            jax.debug.print("Function Value = {}", self.f_outer(carry['inner_var'], carry['outer_var']))

            return carry, _


        return soba_one_iter
