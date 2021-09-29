import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .solver import Solver


class NewtonRaphsonSolver(Solver):
    def __init__(self, solver_input_block, n_nodes, n_dof_per_node, connectivity,
                 apply_dirichlet_bcs_to_solution_func, apply_dirichlet_bcs_to_residual_func,
                 apply_dirichlet_bcs_to_tangent_func,
                 assemble_linear_system):
        super(NewtonRaphsonSolver, self).__init__(solver_input_block, n_nodes, n_dof_per_node,
                                                  connectivity)

        self.apply_dirichlet_bcs_to_solution_func = apply_dirichlet_bcs_to_solution_func
        self.apply_dirichlet_bcs_to_residual_func = apply_dirichlet_bcs_to_residual_func
        self.apply_dirichlet_bcs_to_tangent_func = apply_dirichlet_bcs_to_tangent_func
        self.assemble_linear_system = jit(assemble_linear_system)
        self.tangent_function = jit(jacfwd(self.assemble_linear_system))

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

        # loop over the maximum number of iterations
        #
        n = 0
        self.print_solver_heading(time_step)
        while n <= self.solver_input_block['maximum_iterations']:

            def solver_function(values):
                residual, delta_u, u = values

                # force u to satisfy dirichlet conditions on the bcs
                #
                try:
                    u, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes),
                                                self.apply_dirichlet_bcs_to_solution_func,
                                                (u, dirichlet_bcs_nodes, dirichlet_bcs_values))
                except IndexError:
                    pass

                residual = self.assemble_linear_system(u)
                tangent = self.tangent_function(u)

                try:
                    tangent, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes), self.apply_dirichlet_bcs_to_tangent_func,
                                                      (tangent, dirichlet_bcs_nodes, dirichlet_bcs_values))
                except IndexError:
                    pass

                # solve for solution increment
                #
                delta_u, _ = self.linear_solver(tangent, residual,
                                                maxiter=self.solver_input_block['maximum_linear_solver_iterations'])

                # update the solution increment, note where the minus sign is
                #
                u = jax.ops.index_add(u, jax.ops.index[:], -delta_u)

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
