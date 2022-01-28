import functools
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
                 boundary_conditions: list,
                 residual_methods: list,
                 tangent_methods: list) -> None:
        super(NewtonSolver, self).__init__(solver_input_settings,
                                           variables, kernels, boundary_conditions,
                                           residual_methods, tangent_methods)
        self.residual_tolerance = self.solver_input_settings['residual_tolerance']
        self.increment_tolerance = self.solver_input_settings['increment_tolerance']

        self.update_solution = jit(self.update_solution)

    def __str__(self) -> str:
        string = '  ' + __class__.__name__ + ':\n'
        string = string + '    Residual tolerance  = %s\n' % self.residual_tolerance
        string = string + '    Increment tolerance = %s\n' % self.increment_tolerance
        string = string + '    Linear solver       = %s\n' % self.linear_solver_input_settings['type']
        return string

    def update_solution(self,
                        nodal_coordinates: jnp.ndarray,
                        element_connectivity: jnp.ndarray,
                        u: jnp.ndarray) -> tuple:
        for bc in self.boundary_conditions:
            u = bc.modify_solution_vector_to_satisfy_boundary_conditions(u)

        residual = self.residual_methods[0](nodal_coordinates, element_connectivity, u)
        tangent = self.tangent_methods[0](nodal_coordinates, element_connectivity, u)

        delta_u, _ = jax.scipy.sparse.linalg.gmres(tangent, residual)
        u = u - delta_u

        residual_norm = jnp.linalg.norm(residual)
        delta_u_norm = jnp.linalg.norm(delta_u)

        return u, residual_norm, delta_u_norm
