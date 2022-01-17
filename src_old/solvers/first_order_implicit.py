import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd

import time_control
from .solver import Solver


class FirstOrderImplicit(Solver):
    """
    Solve the following equation
    M\dot{c} + Kc = 0
    M(c_tau - c_t) / delta_t + Kc_tau = 0
    Mc_tau + delta_t * Kc_tau = Mc_t
    (M + delta_t * K) c_tau = Mc_t
    c_tau = Mc_t / (M + delta_t * K)
    """
    def __init__(self,
                 solver_input_block,
                 n_nodes, n_dof_per_node, connectivity,
                 apply_dirichlet_bcs_to_solution_func,
                 apply_dirichlet_bcs_to_residual_func,
                 apply_dirichlet_bcs_to_tangent_func,
                 assemble_residual,
                 assemble_stiffness_matrix,
                 assemble_mass_matrix):
        super(FirstOrderImplicit, self).__init__(solver_input_block, n_nodes, n_dof_per_node,
                                                 connectivity)

        self.apply_dirichlet_bcs_to_solution_func = apply_dirichlet_bcs_to_solution_func
        self.apply_dirichlet_bcs_to_residual_func = apply_dirichlet_bcs_to_residual_func
        self.apply_dirichlet_bcs_to_tangent_func = apply_dirichlet_bcs_to_tangent_func
        self.assemble_residual = jit(assemble_residual)
        self.assemble_stiffness_matrix = jit(assemble_stiffness_matrix)
        self.assemble_mass_matrix = jit(assemble_mass_matrix)

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

    def solve(self, time_step, u_0, dirichlet_bcs_nodes, dirichlet_bcs_values):

        # initialize the solution vectors
        #
        u_solve = jnp.zeros_like(u_0)
        residual_solve = jnp.zeros_like(u_solve)
        delta_u_solve = jnp.zeros_like(u_solve)
        u_old = u_0

        u_old, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes),
                                        self.apply_dirichlet_bcs_to_solution_func,
                                        (u_old, dirichlet_bcs_nodes, dirichlet_bcs_values))

        # loop over the maximum number of iterations
        #
        n = 0
        self.print_solver_heading(time_step)
        while n <= self.solver_input_block['maximum_iterations']:

            def solver_function(values):
                residual, delta_u, u = values

                # force u to satisfy dirichlet conditions on the bcs
                #
                # u, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes),
                #                             self.apply_dirichlet_bcs_to_solution_func,
                #                             (u, dirichlet_bcs_nodes, dirichlet_bcs_values))

                # residual = self.assemble_linear_system(u)
                # tangent = self.tangent_function(u)
                # A =
                residual = self.assemble_residual(u)
                K = self.assemble_stiffness_matrix()
                M = self.assemble_mass_matrix()
                A = M + time_step * K
                b = jnp.matmul(M, u_old)
                u_new, _ = self.linear_solver(A, b,
                                              maxiter=self.solver_input_block['maximum_linear_solver_iterations'])
                # u = jnp.matmul(jnp.linalg.inv(A), b)
                # tangent, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes), self.apply_dirichlet_bcs_to_tangent_func,
                #                                   (tangent, dirichlet_bcs_nodes, dirichlet_bcs_values))

                # solve for solution increment
                #
                # delta_u, _ = self.linear_solver(tangent, residual,
                #                                 maxiter=self.solver_input_block['maximum_linear_solver_iterations'])
                delta_u = u_new - u_old
                # update the solution increment, note where the minus sign is
                #
                u = jax.ops.index_add(u, jax.ops.index[:], delta_u)
                # u = jax.ops.index_update(u, jax.ops.index[:], u_new)
                return residual, delta_u, u

            output = solver_function((residual_solve, delta_u_solve, u_solve))
            n = n + 1
            residual_solve, delta_u_solve, u_solve = output

            residual_error, increment_error = jnp.linalg.norm(residual_solve), jnp.linalg.norm(delta_u_solve)

            self.print_solver_state(n, residual_error.ravel(), increment_error.ravel())

            if residual_error < self.solver_input_block['residual_tolerance']:
                print('Converged on residual: |R| = {0:.8e}'.format(residual_error.ravel()[0]))
                break

            if increment_error < self.solver_input_block['increment_tolerance']:
                print('Converged on increment: |du| = {0:.8e}'.format(increment_error.ravel()[0]))
                break

        return u_solve
