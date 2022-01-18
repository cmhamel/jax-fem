import numpy as np
import jax.numpy as jnp
import exodus3 as exodus
from .mesh_base_class import MeshBaseClass
from util import SuppressStdOutput


class GenesisMesh(MeshBaseClass):
    def __init__(self,
                 mesh_input_settings: dict,
                 number_of_dimensions: int,
                 summarize=False,
                 convert_to_jax_numpy=True) -> None:
        super(GenesisMesh, self).__init__(mesh_input_settings, number_of_dimensions)

        self.mesh_input_settings = self.mesh_input_settings['genesis_mesh']
        self.genesis_file = self.mesh_input_settings['genesis_file']
        self.summarize = summarize

        # set up exodus file
        #
        self.exo = self._read_genesis_file()

        self.nodal_coordinates = self._read_nodal_coordinates()
        self.element_connectivity = self._read_element_connectivity()

        self.blocks = self._read_blocks()
        self.node_sets = self._read_node_sets()
        self.node_set_nodes = self._read_node_set_nodes()

        self.number_of_nodes = self.nodal_coordinates.shape[0]
        self.number_of_elements = self.element_connectivity.shape[0]

        # self.block_elements = self._read_block_elements()

        if convert_to_jax_numpy:
            self.nodal_coordinates = jnp.array(self.nodal_coordinates)
            self.element_connectivity = jnp.array(self.element_connectivity, dtype=jnp.int64)
            for n in range(len(self.node_set_nodes)):
                self.node_set_nodes[n] = jnp.array(self.node_set_nodes[n], dtype=jnp.int64)

        # close exodus file
        #
        with SuppressStdOutput(suppress_stdout=True):
            self.exo.close()

    def __str__(self) -> str:
        string = 'GenesisMesh(MeshBaseClass):\n'
        string = string + '\tGenesis file name    = %s\n' % self.genesis_file
        string = string + '\tNumber of dimensions = %s\n' % self.number_of_dimensions
        string = string + '\tNumber of nodes      = %s\n' % self.number_of_nodes
        string = string + '\tNumber of elements   = %s\n' % self.number_of_elements
        string = string + '\tBlocks               = '
        for n in range(self.blocks.shape[0]):
            string = string + str(self.blocks[n]) + ' '
        string = string + '\n'
        string = string + '\tNode sets            = '
        for n in range(self.node_sets.shape[0]):
            string = string + str(self.node_sets[n]) + ' '
        string + string + '\n'

        return string

    def _read_genesis_file(self) -> exodus.exodus:
        with SuppressStdOutput(suppress_stdout=True):
            exo = exodus.exodus(self.genesis_file, array_type='numpy')

        if self.summarize:
            self.exo.summarize()

        return exo

    def _read_nodal_coordinates(self) -> np.ndarray:
        x_coords, y_coords, z_coords = self.exo.get_coords()

        if self.number_of_dimensions == 1:
            return x_coords.reshape((-1, 1))

        elif self.number_of_dimensions == 2:
            return np.hstack((x_coords.reshape((-1, 1)),
                              y_coords.reshape((-1, 1))))
        elif self.number_of_dimensions == 3:
            return jnp.hstack((x_coords.reshape((-1, 1)),
                               y_coords.reshape((-1, 1)),
                               z_coords.reshape((-1, 1))))

    def _read_element_connectivity(self) -> np.ndarray:
        connectivity = []
        exodus.collectElemConnectivity(self.exo, connectivity)
        connectivity = np.array(connectivity) - 1
        return connectivity

    def _read_blocks(self) -> np.ndarray:
        blocks = list()
        for block in self.mesh_input_settings['blocks']:
            blocks.append(block)
        return np.array(blocks)

    def _read_node_sets(self) -> np.ndarray:
        node_sets = list()
        for node_set in self.mesh_input_settings['node_sets']:
            node_sets.append(node_set)
        return np.array(node_sets)

    def _read_node_set_nodes(self) -> list:
        node_set_nodes = list()
        for n in range(self.node_sets.shape[0]):
            node_set_nodes.append(self.exo.get_node_set_nodes(self.node_sets[n]) - 1)
        return node_set_nodes

    # def _read_block_elements(self) -> np.ndarray:
    #     for block in self.blocks:
    #         print(block)
    #         element_attributes = self.exo.get_elem_attr(block)