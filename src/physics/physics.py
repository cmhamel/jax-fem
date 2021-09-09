from pre_processing import GenesisMesh


class Physics:
    def __init__(self, n_dimensions, mesh_input):
        self.n_dimensions = n_dimensions
        self.mesh_input = mesh_input

        # read mesh input settings
        #
        self.genesis_file = self.set_genesis_file()
        self.genesis_mesh = self.set_genesis_mesh()
        self.nodal_coordinates = self.set_nodal_coordinates()

        # TODO: modify to support multiple blocks
        #
        self.element_connectivity, self.n_elements_in_block, self.n_nodes_per_element = \
            self.set_element_connectivity(1)

    def __str__(self):
        string = '--- Generic physics base class ---\n'
        string = string + \
                 'Genesis file                = %s\n' % self.genesis_file
        string = string + \
                 'Number of elements in block = %s\n' % self.n_elements_in_block
        string = string + \
                 'Number of nodes per element = %s\n' % self.n_nodes_per_element

        return string

    def set_genesis_file(self):
        try:
            genesis_file = self.mesh_input['genesis_file']
        except KeyError:
            raise Exception('No genesis file found in physics: ')

        return genesis_file

    def set_genesis_mesh(self):
        return GenesisMesh(self.genesis_file)

    def set_nodal_coordinates(self):
        nodal_coordinates = \
            self.genesis_mesh.read_nodal_coordinates(self.n_dimensions)
        return nodal_coordinates

    def set_element_connectivity(self, block_id):
        # below assumes only one block in the entire problem
        # the pre-processing method is generic though and could be
        # used to read in the connectivity of each block
        #
        # TODO: generalize to multiple blocks
        #
        element_connectivity, n_nodes_per_block, n_nodes_per_element = \
            self.genesis_mesh.read_element_connectivity(1)

        return element_connectivity, n_nodes_per_block, n_nodes_per_element
