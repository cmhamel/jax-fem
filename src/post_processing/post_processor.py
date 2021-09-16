import exodus3
from exodus3 import exodus
import jax.numpy as jnp
import numpy as np
import os
from util import SuppressStdOutput


class PostProcessor:
    def __init__(self, n_dimensions, genesis_mesh, post_processor_input_block):
        self.n_dimensions = n_dimensions
        self.genesis_mesh = genesis_mesh
        self.post_processor_input_block = post_processor_input_block
        self.output_file = self.post_processor_input_block['exodus_database']

        with SuppressStdOutput(suppress_stdout=True, suppress_stderr=False):
            self.exo = self.initialize_exodus_output_database()

        try:
            self.nodal_output_variables = \
                self.post_processor_input_block['requested_outputs']['nodal_variables']
            self.nodal_scalar_variables = self.nodal_output_variables['scalars']
            self.setup_nodal_scalar_variables()
        except KeyError:
            self.nodal_output_variables = []
            self.nodal_scalar_variables = []

    def initialize_exodus_output_database(self):

        # need to reopen the mesh genesis file and copy
        # it to name of the output exodus file
        #
        os.system('rm -f ' + self.output_file)

        mesh_exo = exodus(self.genesis_mesh.genesis_file)
        exo = mesh_exo.copy(self.output_file)
        return exo

    def setup_nodal_scalar_variables(self):
        # exodus3.add_variables(self.exo, nodal_vars=self.nodal_scalar_variables)
        self.exo.set_node_variable_number(len(self.nodal_scalar_variables))
        for n, output in enumerate(self.nodal_scalar_variables):
            self.exo.put_node_variable_name(output, n + 1)

    def write_nodal_scalar_variable(self, variable_name, time_step, nodal_values):
        self.exo.put_node_variable_values(variable_name, time_step, nodal_values)
