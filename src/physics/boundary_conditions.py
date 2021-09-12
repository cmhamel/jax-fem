from pre_processing import GenesisMesh
import jax.numpy as jnp


class BoundaryCondition:
    def __init__(self):
        pass
    

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self, node_set_name, node_set_nodes, bc_type, value, time_end=1.0):
        super(DirichletBoundaryCondition, self).__init__()
        self.node_set_name = node_set_name
        self.node_set_nodes = node_set_nodes
        self.bc_type = bc_type
        self.time_end = time_end
        self.value = value
        self.values = self.update_bc_values()
        
    def update_bc_values(self, time=0.0):
        if self.bc_type == 'constant':
            values = self.value * jnp.ones(self.node_set_nodes.shape, dtype=jnp.float64)
        elif self.bc_type == 'ramp':
            values = self.value * (time / self.time_end) * jnp.ones(self.node_set_nodes.shape, dtype=jnp.float64)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported type in DirichletBoundaryCondition')

        return values


class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self):
        super(NeumannBoundaryCondition, self).__init__()
    