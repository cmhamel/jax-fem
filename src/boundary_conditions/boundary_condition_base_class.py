import jax
import jax.numpy as jnp
from mesh import MeshBaseClass


class BoundaryConditionBaseClass:
    def __init__(self,
                 boundary_condition_input_settings: dict) -> None:
        self.boundary_condition_input_settings = boundary_condition_input_settings
        self.node_sets = self.boundary_condition_input_settings['node_sets']
        self.node_set_nodes = None

    def populate_node_set_nodes_from_mesh(self, mesh: MeshBaseClass) -> None:
        node_set_nodes = jnp.array([], dtype=jnp.int64)
        for node_set in self.node_sets:
            index = jnp.argwhere(mesh.node_sets == node_set)[0][0]
            node_set_nodes = jnp.hstack((node_set_nodes, mesh.node_set_nodes[index]))

        self.node_set_nodes = node_set_nodes
