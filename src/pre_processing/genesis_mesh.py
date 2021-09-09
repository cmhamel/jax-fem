from jax import jit
import jax.numpy as jnp
from exodus3 import exodus


class GenesisMesh:
    def __init__(self, genesis_file, summarize=False):

        self.genesis_file = genesis_file
        self.exo = exo = exodus(self.genesis_file, array_type='numpy')

        if summarize:
            exo.summarize()

    def read_nodal_coordinates(self, number_of_dimensions):

        # read nodal coordinates
        #
        x_coordinates, y_coordinates, z_coordinates = self.exo.get_coords()

        if number_of_dimensions == 2:
            nodal_coordinates = jnp.concatenate((x_coordinates.reshape((-1, 1)),
                                                 y_coordinates.reshape((-1, 1))), 1)
        else:
            assert False, 'unsupported number of dimensions currently'

        return nodal_coordinates

    def read_element_connectivity(self, block_id):
        element_connectivity, n_elements_in_block, n_nodes_per_element = \
            self.exo.get_elem_connectivity(block_id)

        return element_connectivity, n_elements_in_block, n_nodes_per_element
