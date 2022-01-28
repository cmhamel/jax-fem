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

        # update solution vectors to match the dirichlet bcs
        #
        for n, variable in enumerate(self.variables):
            for bc in self.boundary_conditions:
                if bc.variable == variable:
                    u = jax.ops.index_update(u, jax.ops.index[:, n],
                                             bc.modify_solution_vector_to_satisfy_boundary_conditions(u[:, n]))

        # vectorize solution vectors
        #
        u_vec = u[:, 0]
        if u.shape[1] > 1:
            for n in range(1, u.shape[1]):
                u_vec = jnp.hstack((u_vec, u[:, n]))

        residual = jnp.zeros_like(u_vec)
        for n, residual_method in enumerate(self.residual_methods):
            if residual_method is not None:
                temp_residual = residual_method(nodal_coordinates, element_connectivity, u[:, n])
                residual = jax.ops.index_update(residual,
                                                jax.ops.index[n * u.shape[0]:(n + 1) * u.shape[0]],
                                                temp_residual)

        # calculate individual tangents
        #
        # TODO: This only works works for uncoupled kernels currently, need to update everything to include
        # TODO: couple kernels
        #
        tangent = jnp.zeros((residual.shape[0], residual.shape[0]))
        for n, tangent_method in enumerate(self.tangent_methods):
            if tangent_method is not None:
                temp_tangent = tangent_method(nodal_coordinates, element_connectivity, u[:, n])
                tangent = jax.ops.index_update(tangent,
                                               jax.ops.index[n * u.shape[0]:(n + 1) * u.shape[0],
                                                             n * u.shape[0]:(n + 1) * u.shape[0]],
                                               temp_tangent)

        delta_u_vec, _ = jax.scipy.sparse.linalg.gmres(tangent, residual)
        u_vec = u_vec - delta_u_vec

        residual_norm = jnp.linalg.norm(residual)
        delta_u_vec_norm = jnp.linalg.norm(delta_u_vec)

        # unvectorize
        #
        u_new = jnp.zeros_like(u)
        for n in range(u.shape[1]):
            u_new = jax.ops.index_update(u_new, jax.ops.index[:, n], u_vec[n * u.shape[0]:(n + 1) * u.shape[0]])

        return u_new, residual_norm, delta_u_vec_norm

    def update_solution_old_but_works(self,
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
