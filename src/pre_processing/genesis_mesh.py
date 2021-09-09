from jax import jit
import jax.numpy as jnp
from exodus3 import exodus
from .mesh import Mesh


class GenesisMesh(Mesh):
    def __init__(self, genesis_file, blocks, node_sets, side_sets, summarize=False):
        super(GenesisMesh, self).__init__()
        self.genesis_file = genesis_file
        self.blocks = blocks
        self.node_sets = node_sets
        self.side_sets = side_sets

        self.exo = exo = exodus(self.genesis_file, array_type='numpy')

        if summarize:
            exo.summarize()

        # initialize mesh arrays
        #
        self.n_dimensions = self.read_number_of_dimensions()
        self.nodal_coordinates = self.read_nodal_coordinates()
        self.element_connectivity, self.n_elements_in_block, self.n_nodes_per_element = \
            self.read_element_connectivity(1)  # TODO add multiple blocks

    def __del__(self):
        self.genesis_file = None
        self.exo.close()
        self.exo = None
        self.nodal_coordinates = None
        self.element_connectivity = None

    def __str__(self):
        print('Genesis file = %s' % self.genesis_file)
        print('Blocks:')
        for block in self.blocks:
            print('\t%s' % block)
        print('\n')
        print('Node sets:')
        for node_set in self.node_sets:
            print('\t%s' % node_set)
        print('\n')
        print('Side sets:')
        for side_set in self.side_sets:
            print('\t%s' % side_set)
        print('\n')

    def read_number_of_dimensions(self):
        n_dimensions = self.exo.read_number_of_dimensions
        return n_dimensions

    def read_nodal_coordinates(self):

        # read nodal coordinates
        #
        x_coordinates, y_coordinates, z_coordinates = self.exo.get_coords()

        if self.number_of_dimensions == 1:
            nodal_coordinates = x_coordinates.reshape((-1, 1))
        elif self.number_of_dimensions == 2:
            nodal_coordinates = jnp.concatenate((x_coordinates.reshape((-1, 1)),
                                                 y_coordinates.reshape((-1, 1))), 1)
        elif self.number_of_dimensions == 3:
            nodal_coordinates = jnp.concatenate((x_coordinates.reshape((-1, 1)),
                                                 y_coordinates.reshape((-1, 1)),
                                                 z_coordinates.reshape((-1, 1))), 1)
        else:
            assert False, 'unsupported number of dimensions currently'

        return nodal_coordinates

    def read_element_connectivity(self, block_id):
        element_connectivity, n_elements_in_block, n_nodes_per_element = \
            self.exo.get_elem_connectivity(block_id)

        return element_connectivity, n_elements_in_block, n_nodes_per_element

    def read_node_set_nodes(self):
        pass

    def read_side_set_elements_and_faces(self):
        pass
