import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .solver import Solver


class FirstOrderExplicit(Solver):
    def __init__(self,
                 solver_input_block,
                 n_nodes, n_dof_per_node, connectivity,
                 apply_dirichlet_bcs_to_solution_func,
                 apply_dirichlet_bcs_to_residual_func,
                 apply_dirichlet_bcs_to_tangent_func,
                 assemble_linear_system,
                 assemble_mass_matrix):
        super(FirstOrderExplicit, self)\
            .__init__(solver_input_block, n_nodes, n_dof_per_node, connectivity)

        self.apply_dirichlet_bcs_to_solution_func = apply_dirichlet_bcs_to_solution_func
        self.apply_dirichlet_bcs_to_residual_func = apply_dirichlet_bcs_to_residual_func
        self.apply_dirichlet_bcs_to_tangent_func = apply_dirichlet_bcs_to_tangent_func
        self.assemble_linear_system = jit(assemble_linear_system)
        self.assemble_mass_matrix = jit(assemble_mass_matrix)
        self.mass_matrix_inv = jnp.linalg.inv(self.assemble_mass_matrix())

    def solver_heading(self):
        print('%s\t\t%\t\t%s\t\t%s' % ('Increment', 'Time', 'Residual', 'Increment'))

    def solve(self, time_control, u_old, dirichlet_bcs_nodes, dirichlet_bcs_values):
        # force u to satisfy dirichlet conditions on the bcs
        #
        u_old, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes),
                                        self.apply_dirichlet_bcs_to_solution_func,
                                        (u_old, dirichlet_bcs_nodes, dirichlet_bcs_values))

        R = self.assemble_linear_system(u_old)
        # M = self.assemble_mass_matrix()
        # delta_u = time_step * jnp.matmul(jnp.linalg.inv(M), R)
        delta_u = time_control.time_increment * jnp.matmul(self.mass_matrix_inv, R)
        u = u_old + delta_u

        residual = (u - u_old) / time_control.time_increment - R

        residual_error = jnp.linalg.norm(residual)
        increment_error = jnp.linalg.norm(delta_u)

        if time_control.time_step_number % 100 == 0:
            print('{0:}\t\t{1:.8e}\t\t{2:.8e}\t\t{3:.8e}'.format(time_control.time_step_number,
                                                                 time_control.t,
                                                                 residual_error.ravel()[0],
                                                                 increment_error.ravel()[0]))

        if time_control.time_step_number % 10000 == 0:
            self.print_solver_heading(time_control.time_step_number)


        return u
