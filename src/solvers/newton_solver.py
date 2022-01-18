import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .solver_base_class import SolverBaseClass


class NewtonSolver(SolverBaseClass):
    def __init__(self,
                 solver_input_settings: dict,
                 variables: list,
                 kernels: list,
                 boundary_conditions: list) -> None:
        super(NewtonSolver, self).__init__(solver_input_settings, variables, kernels, boundary_conditions)
        self.residual_tolerance = self.solver_input_settings['residual_tolerance']
        self.increment_tolerance = self.solver_input_settings['increment_tolerance']

        self.update_solution = jit(self.update_solution)

    def __str__(self) -> str:
        string = 'NewtonSolver(SolverBaseClass):\n'
        string = string + '\tResidual tolerance  = %s\n' % self.residual_tolerance
        string = string + '\tIncrement tolerance = %s\n' % self.increment_tolerance
        string = string + '\tLinear solver       = %s\n' % self.linear_solver_input_settings['type']
        return string

    def calculate_total_residual(self,
                                 nodal_coordinates: jnp.ndarray,
                                 element_connectivity: jnp.ndarray,
                                 u: jnp.ndarray) -> jnp.ndarray:

        # TODO: need to update for coupled problems
        #
        # calculate residual
        #
        residual = jnp.zeros_like(u)
        for kernel in self.kernels:
            residual = residual + kernel.calculate_residual(nodal_coordinates,
                                                            element_connectivity,
                                                            u)

        # modify residual to satisfy bcs
        #
        for bc in self.boundary_conditions:
            residual = bc.modify_residual_vector_to_satisfy_boundary_conditions(residual)

        return residual

    def update_solution(self,
                        nodal_coordinates: jnp.ndarray,
                        element_connectivity: jnp.ndarray,
                        u: jnp.ndarray) -> tuple:

        # TODO: need to update based on the variable on the dirichlet BC
        #
        for bc in self.boundary_conditions:
            u = bc.modify_solution_vector_to_satisfy_boundary_conditions(u)

        residual = self.calculate_total_residual(nodal_coordinates,
                                                 element_connectivity,
                                                 u)
        tangent = jacfwd(self.calculate_total_residual, argnums=2)(nodal_coordinates,
                                                                   element_connectivity,
                                                                   u)

        for bc in self.boundary_conditions:
            tangent = bc.modify_tangent_matrix_to_satisfy_boundary_conditions(tangent)

        delta_u, _ = jax.scipy.sparse.linalg.gmres(tangent, residual)
        u = u - delta_u

        residual_norm = jnp.linalg.norm(residual)
        delta_u_norm = jnp.linalg.norm(delta_u)

        return u, residual_norm, delta_u_norm


