import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .solver import Solver


class NewtonRaphsonSolver(Solver):
    def __init__(self, solver_input_block,
                 n_nodes, n_dof_per_node, connectivity,
                 assemble_residual,
                 assemble_tangent,
                 apply_dirichlet_bcs_to_solution_func,
                 apply_dirichlet_bcs_to_residual_func,
                 apply_dirichlet_bcs_to_tangent_func):
        super(NewtonRaphsonSolver, self).__init__(solver_input_block, n_nodes, n_dof_per_node,
                                                  connectivity)

        self.apply_dirichlet_bcs_to_solution_func = apply_dirichlet_bcs_to_solution_func
        self.apply_dirichlet_bcs_to_residual_func = apply_dirichlet_bcs_to_residual_func
        self.apply_dirichlet_bcs_to_tangent_func = apply_dirichlet_bcs_to_tangent_func
        self.assemble_residual = assemble_residual
        self.assemble_tangent = assemble_tangent

        if self.solver_input_block['linear_solver'].lower() == 'gmres':
            self.linear_solver = jit(jax.scipy.sparse.linalg.gmres)
        elif self.solver_input_block['linear_solver'].lower() == 'cg':
            self.linear_solver = jit(jax.scipy.sparse.linalg.cg)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported linear solver: %s in NewtonRaphsonSolver' %
                                self.solver_input_block['linear_solver'])

        # TODO figure out how to do this right
        # self.jit_solve = jit(self.solve)

    def solve(self,
              u_old, n_dirichlet_bcs, dirichlet_bcs_nodes, dirichlet_bcs_values,
              time_step, t, delta_t):

        # make an initial guess for newton iterations
        #
        u = jnp.zeros_like(u_old)

        # loop over the maximum number of iterations
        #
        n = 0
        self.print_solver_heading(time_step)
        while n <= self.solver_input_block['maximum_iterations']:
            # enforce bcs on u
            #
            try:
                u, _, _ = jax.lax.fori_loop(0, n_dirichlet_bcs,
                                            self.apply_dirichlet_bcs_to_solution_func,
                                            (u, dirichlet_bcs_nodes, dirichlet_bcs_values))
            except IndexError:
                pass

            # assemble residual and tangent
            #
            residual = self.assemble_residual(u, u_old, t, delta_t)
            try:
                residual, _ = jax.lax.fori_loop(0, n_dirichlet_bcs,
                                                self.apply_dirichlet_bcs_to_residual_func,
                                                (residual, dirichlet_bcs_nodes))
            except IndexError:
                pass

            residual_error = jnp.linalg.norm(residual)

            # check error on residual
            #
            if residual_error < self.solver_input_block['residual_tolerance']:
                print('Converged on residual: |R| = {0:.8e}'.format(residual_error.ravel()[0]))
                break

            # if not converged on residual, calculate tangent and newton step
            #
            tangent = jacfwd(self.assemble_residual, argnums=0)(u, u_old, t, delta_t)

            # enforce bcs on residual and tangent
            #
            try:
                tangent, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes), self.apply_dirichlet_bcs_to_tangent_func,
                                                  (tangent, dirichlet_bcs_nodes, dirichlet_bcs_values))
            except IndexError:
                pass

            delta_u, _ = self.linear_solver(tangent, -residual)
            u = jax.ops.index_add(u, jax.ops.index[:], delta_u)

            increment_error = jnp.linalg.norm(delta_u)

            if increment_error < self.solver_input_block['increment_tolerance']:
                print('Converged on increment: |du| = {0:.8e}'.format(increment_error.ravel()[0]))
                break

            self.print_solver_state(n, residual_error.ravel(), increment_error.ravel())

            n = n + 1

        return u
