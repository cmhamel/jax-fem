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

        # scalar variables
        #
        # try:
        #     self.nodal_output_variables = \
        #         self.post_processor_input_block['requested_outputs']['nodal_variables']
        #     self.nodal_scalar_variables = self.nodal_output_variables['scalars']
        #     self.setup_nodal_scalar_variables()
        # except KeyError:
        #     self.nodal_output_variables = []
        #     self.nodal_scalar_variables = []

        # nodal variables
        #
        try:
            self.nodal_output_variables = self.post_processor_input_block['requested_outputs']['nodal_variables']
            # collect scalar variables
            #
            try:
                self.nodal_scalar_variables = self.nodal_output_variables['scalars']
            except KeyError:
                self.nodal_scalar_variables = []
            # collect vector variables
            #
            try:
                self.nodal_vector_variables = self.nodal_output_variables['vectors']
            except KeyError:
                self.nodal_vector_variables = []
        except KeyError:
            self.nodal_output_variables = []
            self.nodal_scalar_variables = []
            self.nodal_vector_variables = []

        # count all variables and set up total variables in exodus database
        #
        # self.exo.set_node_variable_number(len(self.nodal_scalar_variables) +
        #                                   self.n_dimensions * len(self.nodal_vector_variables))
        self.exo.set_node_variable_number(len(self.nodal_scalar_variables) + 3 * len(self.nodal_vector_variables))

        # associate variable names with numbers in exo database
        #
        variable_number = 0
        for n, output in enumerate(self.nodal_scalar_variables):
            self.exo.put_node_variable_name(output, variable_number + 1)
            variable_number = variable_number + 1

        for n, output in enumerate(self.nodal_vector_variables):
            # if self.n_dimensions == 1:
            #     self.exo.put_node_variable_name(output + '_x', variable_number + 1)
            # elif self.n_dimensions == 2:
            #     self.exo.put_node_variable_name(output + '_x', variable_number + 1)
            #     self.exo.put_node_variable_name(output + '_y', variable_number + 2)
            # else:
            #     assert False
            self.exo.put_node_variable_name(output + '_x', variable_number + 1)
            self.exo.put_node_variable_name(output + '_y', variable_number + 2)
            self.exo.put_node_variable_name(output + '_z', variable_number + 3)

            # variable_number = variable_number + self.n_dimensions
            variable_number = variable_number + 3

    def initialize_exodus_output_database(self):

        # need to reopen the mesh genesis file and copy
        # it to name of the output exodus file
        #
        os.system('rm -f ' + self.output_file)

        mesh_exo = exodus(self.genesis_mesh.genesis_file)
        exo = mesh_exo.copy(self.output_file)
        return exo

    # def setup_nodal_scalar_variables(self):
    #     # exodus3.add_variables(self.exo, nodal_vars=self.nodal_scalar_variables)
    #     # self.exo.set_node_variable_number(len(self.nodal_scalar_variables))
    #     for n, output in enumerate(self.nodal_scalar_variables):
    #         self.exo.put_node_variable_name(output, n + 1)
    #
    # def setup_nodal_vector_variables(self):
    #     # self.exo.set_node_variable_number(len())
    #     # for n, output in enumerate(self.noda)
    #     for n, output in enumerate(self.nodal_vector_variables):
    #         if self.n_dimensions == 1:
    #             self.exo.put_node_variable_name(output + '_x', n + 1)

    def write_nodal_scalar_variable(self, variable_name, time_step, nodal_values):
        self.exo.put_node_variable_values(variable_name, time_step, nodal_values)

    def write_nodal_vector_variable(self, variable_name, time_step, nodal_values):
        if self.n_dimensions == 1:
            self.exo.put_node_variable_values(variable_name + '_x', time_step, nodal_values[:, 0])
        elif self.n_dimensions == 2:
            self.exo.put_node_variable_values(variable_name + '_x', time_step, nodal_values[:, 0])
            self.exo.put_node_variable_values(variable_name + '_y', time_step, nodal_values[:, 1])
