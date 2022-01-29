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
                 bc_update_solution_methods: list,
                 residual_methods: list,
                 tangent_methods_diagonal: list,
                 tangent_methods_off_diagonal=None) -> None:
        super(NewtonSolver, self).__init__(solver_input_settings,
                                           variables,
                                           kernels,
                                           boundary_conditions,
                                           bc_update_solution_methods,
                                           residual_methods,
                                           tangent_methods_diagonal,
                                           tangent_methods_off_diagonal)
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

        # update solution vectors to match the dirichlet bcs
        #
        for n, variable in enumerate(self.variables):
            u = jax.ops.index_update(u, jax.ops.index[:, n],
                                     self.bc_update_solution_methods[n](u[:, n]))

        # calculate residual
        # TODO: this only works for uncoupled problems so far, need to figure
        # TODO: out how to properly construct lambdas from coupled kernels
        #
        residual = jnp.zeros((u.shape[0] * u.shape[1]))
        tangent = jnp.zeros((residual.shape[0], residual.shape[0]))
        residual_norms = []
        for n, residual_method in enumerate(self.residual_methods):
            temp_residual = residual_method(nodal_coordinates, element_connectivity, u[:, n])
            residual_norms.append(jnp.linalg.norm(temp_residual))
            residual = jax.ops.index_update(residual,
                                            jax.ops.index[n * u.shape[0]:(n + 1) * u.shape[0]],
                                            temp_residual)

            temp_tangent = jacfwd(residual_method, argnums=2)(nodal_coordinates, element_connectivity, u[:, n])

            # TODO: hook up to tangent update bc methods that are constructed from lambdas
            #
            for bc in self.boundary_conditions:
                if bc.variable == self.variables[n]:
                    temp_tangent = bc.modify_tangent_matrix_to_satisfy_boundary_conditions(temp_tangent)
            tangent = jax.ops.index_update(tangent,
                                           jax.ops.index[n * u.shape[0]:(n + 1) * u.shape[0],
                                                         n * u.shape[0]:(n + 1) * u.shape[0]],
                                           temp_tangent)

        # calculate individual diagonal tangents
        #
        # TODO: This only works works for uncoupled kernels currently, need to update everything to include
        # TODO: couple kernels

        # TODO: add the hooks for off diagonal tangent assembly
        #

        # TODO: hook up to linear solver factory
        #
        delta_u_vec = jax.scipy.linalg.solve(tangent, residual)

        # unvectorize solution element
        #
        delta_u = jnp.zeros_like(u)
        for n in range(u.shape[1]):
            delta_u = jax.ops.index_update(delta_u, jax.ops.index[:, n],
                                           delta_u_vec[n * u.shape[0]:(n + 1) * u.shape[0]])

        # calculate some norms
        #
        residual_norm = jnp.linalg.norm(residual)
        delta_u_vec_norm = jnp.linalg.norm(delta_u_vec)
        delta_u_norms = jnp.linalg.norm(delta_u, axis=0)

        # update solution array and return
        #
        u_new = u - delta_u

        return u_new, residual_norm, residual_norms, delta_u_vec_norm, delta_u_norms
