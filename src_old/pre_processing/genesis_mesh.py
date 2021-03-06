import jax.ops
from jax import jit
import jax.numpy as jnp
import exodus3
from exodus3 import exodus
from .mesh import Mesh
from util import SuppressStdOutput
from util import general_tardigrade_error


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
        with SuppressStdOutput(suppress_stdout=True, suppress_stderr=False):
            self.exo = exodus(self.genesis_file, array_type='numpy')

        if summarize:
            self.exo.summarize()

        # check to make sure the node sets and side sets are valid
        #
        if len(self.node_sets) > 0:
            for node_set in self.node_sets:
                assert node_set in self.exo.get_node_set_ids()

        # TODO: fix the below to use numbers instead of names
        #
        # if len(self.side_sets) > 0:
        #     exo_side_set_names = self.exo.get_side_set_names()
        #     for side_set in self.side_sets:
        #         assert side_set in exo_side_set_names

        # initialize mesh arrays
        #
        self.nodal_coordinates = jnp.array(self.read_nodal_coordinates())
        self.element_connectivities = []
        self.n_elements_in_blocks = []
        self.n_nodes_per_element = []
        self.connectivity = self.read_collected_element_connectivity()

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
        with SuppressStdOutput(suppress_stdout=True, suppress_stderr=False):
            self.exo.close()

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
            general_tardigrade_error('Bad number of dimensions in GenesisMesh')

        return nodal_coordinates

    def read_element_connectivity(self, block_id):
        element_connectivity, n_elements_in_block, n_nodes_per_element = \
            self.exo.get_elem_connectivity(block_id)

        return element_connectivity, n_elements_in_block, n_nodes_per_element

    def read_collected_element_connectivity(self):
        """
        Read the collected element connectivity
        this in general will probably be easier to handle element wise
        TODO: figure out how to handle multiple blocks with this format
        TODO: probably need to get a list of elements in a given block
        TODO: and index that way for different materials
        the -1 is to shift it to be zero indexed for python
        :return: collected element connectivity
        """
        connectivity = []
        exodus3.collectElemConnectivity(self.exo, connectivity)
        connectivity = jnp.array(connectivity) - 1
        return connectivity

    # method to create a new connectivity matrix for multi-dof per node problems
    #
    def make_multiple_dof_connectivity(self, n_dof):

        # error checking and uninteresting cases
        #
        assert n_dof > 0
        if n_dof == 1:
            return self.connectivity

        connectivity = jnp.zeros((self.connectivity.shape[0], n_dof * self.connectivity.shape[1]), dtype=jnp.int32)

        for n in range(self.connectivity.shape[1]):
            connectivity = jax.ops.index_update(connectivity, jax.ops.index[:, n_dof * n],
                                                n_dof * self.connectivity[:, n])

        for n in range(n_dof * self.connectivity.shape[1]):
            if n % n_dof == 0:
                continue
            else:
                connectivity = jax.ops.index_update(connectivity, jax.ops.index[:, n], connectivity[:, n - 1] + 1)

        return connectivity

    def read_node_set_nodes(self):
        node_set_nodes = []
        for node_set in self.node_sets:
            node_set_nodes.append(self.exo.get_node_set_nodes(node_set) - 1)

        return node_set_nodes

    def modify_node_list_for_multiple_dofs(self, n_dof, node_set_number):
        assert n_dof > 0
        if n_dof == 1:
            return self.node_set_nodes[node_set_number]

        node_set_nodes = jnp.zeros(n_dof * self.node_set_nodes[node_set_number].shape[0], dtype=jnp.int32)

        node_set_nodes = jax.ops.index_update(node_set_nodes, jax.ops.index[::n_dof],
                                              n_dof * self.node_set_nodes[node_set_number])
        # for n in range(n_dof):
        #     node_set_nodes = jax.ops.index_update(node_set_nodes, jax.ops.index[n::2],
        #                                           )
        for n in range(len(node_set_nodes)):
            if n % n_dof == 0:
                continue
            else:
                node_set_nodes = jax.ops.index_update(node_set_nodes, jax.ops.index[n],
                                                      node_set_nodes[n - 1] + 1)

        return node_set_nodes

    def read_side_set_elements_and_faces(self):
        pass
        # TODO finish this out when you implement Neumann boundary conditions
