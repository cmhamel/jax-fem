import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .solver import Solver


class FirstOrderExplicit(Solver):
    """
    Need to solve the following
    M\dot{u} + Ku = F, for now assume F = 0 then
    M\dot{u} + Ku = 0
    M(u_tau - u_t) + dt K u_t = 0
    Mu_tau = Mu_t - dt K u_t
    u_tau = u_t - dt M^-1 K u_t
    """
    def __init__(self,
                 solver_input_block,
                 n_nodes, n_dof_per_node, connectivity,
                 apply_dirichlet_bcs_to_solution_func,
                 apply_dirichlet_bcs_to_residual_func,
                 apply_dirichlet_bcs_to_tangent_func,
                 # assemble_linear_system,
                 assemble_stiffness_matrix,
                 assemble_mass_matrix,
                 property):
        super(FirstOrderExplicit, self)\
            .__init__(solver_input_block, n_nodes, n_dof_per_node, connectivity)

        self.apply_dirichlet_bcs_to_solution_func = apply_dirichlet_bcs_to_solution_func
        self.apply_dirichlet_bcs_to_residual_func = apply_dirichlet_bcs_to_residual_func
        self.apply_dirichlet_bcs_to_tangent_func = apply_dirichlet_bcs_to_tangent_func
        # self.assemble_linear_system = jit(assemble_linear_system)
        # self.assemble_stiffness_matrix = jit(jacfwd(self.assemble_linear_system))
        self.assemble_stiffness_matrix = jit(assemble_stiffness_matrix)
        self.assemble_mass_matrix = jit(assemble_mass_matrix)
        self.mass_matrix = self.assemble_mass_matrix()
        self.property = property

        # lumped mass matrix using the row method
        #
        self.lumped_mass_matrix = jnp.sum(self.mass_matrix, axis=1)
        self.max_eigenvalue = jnp.max(self.lumped_mass_matrix)
        # self.delta_t = 2 * self.property / self.max_eigenvalue

        # print(self.delta_t)
        # import sys
        # sys.exit()

        # self.jit_solve = jit(self.solve)

    def solver_heading(self):
        print('%s\t\t%\t\t%s\t\t%s' % ('Increment', 'Time', 'Residual', 'Increment'))

    def solve(self, time_control, u_old, dirichlet_bcs_nodes, dirichlet_bcs_values):
        # force u to satisfy dirichlet conditions on the bcs
        #
        u_old, _, _ = jax.lax.fori_loop(0, len(dirichlet_bcs_nodes),
                                        self.apply_dirichlet_bcs_to_solution_func,
                                        (u_old, dirichlet_bcs_nodes, dirichlet_bcs_values))

        # R = self.assemble_linear_system(u_old)
        K = self.assemble_stiffness_matrix()
        delta_t = time_control.time_increment

        delta_u = -delta_t * jnp.matmul(K, u_old) / self.lumped_mass_matrix
        u = u_old + delta_u

        # residual = (u - u_old) / time_control.time_increment - R
        residual = u - u_old

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
