import jax

from pre_processing import GenesisMesh
from solvers import NewtonRaphsonSolver
from post_processing import PostProcessor
import jax.numpy as jnp
from jax import jit, partial


class Physics:
    def __init__(self, n_dimensions, physics_input):
        self.n_dimensions = n_dimensions
        self.physics_input = physics_input

        # grab some different blocks of the input deck
        #
        self.mesh_input_block = self.physics_input['mesh']
        self.blocks_input_block = self.physics_input['blocks']
        try:
            self.boundary_conditions_input_block = self.physics_input['boundary_conditions']
        except KeyError:
            self.boundary_conditions_input_block = {}

        try:
            self.initial_conditions_input_block = self.physics_input['initial_conditions']
        except KeyError:
            self.initial_conditions_input_block = {}

        # if explicit we don't have a solver block
        #
        try:
            self.solver_input_block = self.physics_input['solver']
        except KeyError:
            self.solver_input_block = None

        self.post_processing_block = self.physics_input['output']

        # read mesh input settings
        #
        self.genesis_file = self.read_genesis_file()

        # get the blocks
        #
        self.block_numbers = self.read_genesis_block_numbers()

        # loop over generic boundary conditions and get all node sets and sides
        #
        self.node_set_names = self.read_node_set_names()
        self.side_set_names = self.read_side_set_names()

        # set the mesh
        #
        self.genesis_mesh = self.set_genesis_mesh()

        # set up the post processor
        #
        self.post_processor = self.set_post_processor()

    def __str__(self):
        string = '--- Generic physics base class ---\n'
        string = string + \
                 'Genesis file = %s\n\n' % self.genesis_file

        for n in range(len(self.block_numbers)):
            string = string + \
                     '\tBlock number = %s\n' % self.block_numbers[n]
            string = string + \
                     '\t\tNumber of elements in block = %s\n' % self.genesis_mesh.n_elements_in_blocks[n]
            string = string + \
                     '\t\tNumber of nodes per element = %s\n' % self.genesis_mesh.n_nodes_per_element[n]

        string = string + '\tNode sets\n'
        for n in range(len(self.node_set_names)):
            string = string + \
                     '\t\t%s\n' % self.node_set_names[n]

        string = string + '\tSide sets\n'
        for n in range(len(self.side_set_names)):
            string = string + \
                     '\t\t%s\n' % self.side_set_names[n]

        return string

    def read_genesis_file(self):
        try:
            genesis_file = self.mesh_input_block['genesis_file']
        except KeyError:
            raise Exception('No genesis file found in physics: ')

        return genesis_file

    def read_genesis_block_numbers(self):
        block_numbers = []
        for block in self.blocks_input_block:
            block_numbers.append(block['block_number'])
        return block_numbers

    def read_node_set_names(self):
        node_sets = []
        for key in self.boundary_conditions_input_block.keys():
            bcs = self.boundary_conditions_input_block[key]
            for bc in bcs:
                if 'node_set' in bc.keys():
                    node_sets.append(bc['node_set'])

        return node_sets

    def read_side_set_names(self):
        side_sets = []
        for key in self.boundary_conditions_input_block.keys():
            bcs = self.boundary_conditions_input_block[key]
            for bc in bcs:
                if 'side_set' in bc.keys():
                    side_sets.append(bc['side_set'])

        return side_sets

    def set_genesis_mesh(self):
        return GenesisMesh(n_dimensions=self.n_dimensions,
                           genesis_file=self.genesis_file,
                           blocks=self.block_numbers,
                           node_sets=self.node_set_names,
                           side_sets=self.side_set_names,
                           summarize=False)

    def set_post_processor(self):
        print('Initializing PostProcessor')
        return PostProcessor(n_dimensions=self.n_dimensions,
                             post_processor_input_block=self.post_processing_block,
                             genesis_mesh=self.genesis_mesh)

    @partial(jit, static_argnums=(0,))
    def get_nodal_quantities_on_element(self, field, e):
        """
        :param field: field is the entire nodal array
        :param e:
        :return:
        """
        field_e = \
            field[self.genesis_mesh.element_connectivities[0]
                 [e * self.genesis_mesh.n_nodes_per_element[0]:
                  (e + 1) * self.genesis_mesh.n_nodes_per_element[0]]]

        return field_e
