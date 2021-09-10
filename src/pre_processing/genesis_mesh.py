from jax import jit
import jax.numpy as jnp
from exodus3 import exodus
from .mesh import Mesh
from util import suppress_stdout


class GenesisMesh(Mesh):
    """
    Data structure for a genesis me, it will only read in things
    that are necessary for the calculation given in the input deck
    """
    def __init__(self, n_dimensions,
                 genesis_file,
                 blocks, node_sets, side_sets,
                 summarize=False):
        """
        :param n_dimensions: number of dimensions given in input deck
        :param genesis_file: genesis file given in input deck
        :param blocks: block numbers supplied by input deck
        :param node_sets: node set names supplied by the input deck
        :param side_sets: side set names supplied by the input deck
        :param summarize: a flag to exodus to print the contents of the file
        """
        super(GenesisMesh, self).__init__()

        # variables specific to the genesis mesh structure
        #
        self.n_dimensions = n_dimensions
        self.genesis_file = genesis_file
        self.blocks = blocks
        self.node_sets = node_sets
        self.side_sets = side_sets

        # with suppress_stdout:
        # with suppress_stdout():
        self.exo = exodus(self.genesis_file, array_type='numpy')

        if summarize:
            self.exo.summarize()

        # check to make sure the node sets and side sets are valid
        #
        if len(self.node_sets) > 0:
            exo_node_set_names = self.exo.get_node_set_names()
            for node_set in self.node_sets:
                assert node_set in exo_node_set_names

        if len(self.side_sets) > 0:
            exo_side_set_names = self.exo.get_side_set_names()
            for side_set in self.side_sets:
                assert side_set in exo_side_set_names

        # initialize mesh arrays
        #
        self.nodal_coordinates = jnp.array(self.read_nodal_coordinates())
        self.element_connectivities = []
        self.n_elements_in_blocks = []
        self.n_nodes_per_element = []

        for block in self.blocks:
            element_connectivity, n_elements_in_block, n_nodes_per_element = \
                self.read_element_connectivity(block)  # TODO add multiple blocks

            # subtract one to make it zero indexed
            #
            self.element_connectivities.append(jnp.array(element_connectivity) - 1)
            self.n_elements_in_blocks.append(n_elements_in_block)
            self.n_nodes_per_element.append(n_nodes_per_element)

        # get stuff for node sets
        #
        self.node_set_nodes = self.read_node_set_nodes()

        # with suppress_stdout:
        #     exo.close()

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

    def read_nodal_coordinates(self):

        # read nodal coordinates
        #
        x_coordinates, y_coordinates, z_coordinates = self.exo.get_coords()

        if self.n_dimensions == 1:
            nodal_coordinates = x_coordinates.reshape((-1, 1))
        elif self.n_dimensions == 2:
            nodal_coordinates = jnp.concatenate((x_coordinates.reshape((-1, 1)),
                                                 y_coordinates.reshape((-1, 1))), 1)
        elif self.n_dimensions == 3:
            nodal_coordinates = jnp.concatenate((x_coordinates.reshape((-1, 1)),
                                                 y_coordinates.reshape((-1, 1)),
                                                 z_coordinates.reshape((-1, 1))), 1)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Bad number df dimensions in GenesisMesh')

        return nodal_coordinates

    def read_element_connectivity(self, block_id):
        element_connectivity, n_elements_in_block, n_nodes_per_element = \
            self.exo.get_elem_connectivity(block_id)

        return element_connectivity, n_elements_in_block, n_nodes_per_element

    def read_node_set_nodes(self):
        exo_node_set_names = self.exo.get_node_set_names()
        exo_node_set_ids = self.exo.get_node_set_ids()
        node_set_nodes = []
        for node_set in self.node_sets:
            id = exo_node_set_ids[exo_node_set_names.index(node_set)]
            node_set_nodes.append(self.exo.get_node_set_nodes(id) - 1)
        return node_set_nodes

    def read_side_set_elements_and_faces(self):
        pass
        # TODO finish this out when you implement Neumann boundary conditions
