from pre_processing import GenesisMesh
import jax.numpy as jnp


class BoundaryCondition:
    def __init__(self):
        pass
    

class DirichletBoundaryCondition(BoundaryCondition):
    def __init__(self,
                 dirichlet_bc_input_block,
                 node_set_name, node_set_nodes,
                 time_end=1.0, dof_number=0, coordinates=None):
        super(DirichletBoundaryCondition, self).__init__()

        # options
        #
        self.dirichlet_bc_input_block = dirichlet_bc_input_block
        self.bc_type = self.dirichlet_bc_input_block['type']

        # mesh stuff
        #
        self.node_set_name = node_set_name
        self.node_set_nodes = node_set_nodes

        # optional inputs from the code
        #
        self.time_end = time_end
        self.dof_number = dof_number
        self.coordinates = coordinates

        # calculate bc values at the nodes
        #
        self.values = self.update_bc_values()
        
    def update_bc_values(self, time=0.0):
        if self.bc_type.lower() == 'constant':
            scale = self.dirichlet_bc_input_block['value']
            values = scale * jnp.ones(self.node_set_nodes.shape, dtype=jnp.float64)
        elif self.bc_type.lower() == 'ramp':
            scale = self.dirichlet_bc_input_block['value']
            values = scale * (time / self.time_end) * jnp.ones(self.node_set_nodes.shape, dtype=jnp.float64)
        elif self.bc_type.lower() == 'gaussian':
            beam_amplitude = self.dirichlet_bc_input_block['beam_amplitude']
            beam_dimater = self.dirichlet_bc_input_block['beam_diameter']
            beam_center = self.dirichlet_bc_input_block['beam_center']

            # TODO: hardcoded for vertical stuff
            #
            xs = self.coordinates[:, 0]
            values = beam_amplitude * jnp.exp(-(xs - beam_center)**2 / (2 * beam_dimater)**2)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported type in DirichletBoundaryCondition')

        return values


class NeumannBoundaryCondition(BoundaryCondition):
    def __init__(self):
        super(NeumannBoundaryCondition, self).__init__()
    