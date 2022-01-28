import jax
import jax.numpy as jnp
from .boundary_condition_base_class import BoundaryConditionBaseClass


class ConstantDirichletBoundaryCondition(BoundaryConditionBaseClass):
    def __init__(self, boundary_condition_input_settings: dict) -> None:
        super(ConstantDirichletBoundaryCondition, self).__init__(boundary_condition_input_settings)
        self.constant = self.boundary_condition_input_settings['constant']

    def __str__(self) -> str:
        string = '    ' + __class__.__name__ + ':\n'
        string = string + '      Node sets = '
        for node_set in self.node_sets:
            string = string + str(node_set) + ' '
        string = string + '\n'
        string = string + '      Variable  = %s\n' % self.variable
        string = string + '      Constant  = %s\n' % self.constant
        return string

    def modify_solution_vector_to_satisfy_boundary_conditions(self, u: jnp.ndarray) -> jnp.ndarray:
        u = jax.ops.index_update(u, jax.ops.index[self.node_set_nodes], self.constant)
        return u

    def modify_residual_vector_to_satisfy_boundary_conditions(self, residual: jnp.ndarray) -> jnp.ndarray:
        residual = jax.ops.index_update(residual, jax.ops.index[self.node_set_nodes], 0.0)
        return residual

    def modify_tangent_matrix_to_satisfy_boundary_conditions(self, tangent: jnp.ndarray) -> jnp.ndarray:
        # simple penalty like method
        #
        def loop_body(i, temp_tangent):
            return jax.ops.index_update(temp_tangent, jax.ops.index[self.node_set_nodes[i],
                                                                    self.node_set_nodes[i]], 1.0)

        tangent = jax.lax.fori_loop(0, len(self.node_set_nodes), loop_body, tangent)
        return tangent
