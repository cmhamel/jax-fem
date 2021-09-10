from pre_processing import GenesisMesh
import jax.numpy as jnp


class BoundaryCondition:
    def __init__(self):
        pass
    

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self, node_set_name, node_set_nodes, bc_type, value):
        super(DirichletBoundaryCondition, self).__init__()
        self.node_set_name = node_set_name
        self.node_set_nodes = node_set_nodes
        self.bc_type = bc_type
        # self.value = value

        if bc_type == 'constant':
            self.values = value * jnp.ones(self.node_set_nodes.shape, dtype=jnp.float64)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported type in DirichletBoundaryCondition')
        
        
class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self):
        super(NeumannBoundaryCondition, self).__init__()
    