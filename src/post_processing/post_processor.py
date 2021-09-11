import exodus3
from exodus3 import exodus
import jax.numpy as jnp
import numpy as np


class PostProcessor:
    def __init__(self, n_dimensions, genesis_mesh, post_processor_input_block):
        self.n_dimensions = n_dimensions
        self.genesis_mesh = genesis_mesh
        self.post_processor_input_block = post_processor_input_block
        self.output_file = self.post_processor_input_block['exodus_database']
        self.nodal_output_variables = \
            self.post_processor_input_block['requested_outputs']['nodal_variables']
        self.nodal_scalar_variables = self.nodal_output_variables['scalars']
        self.exo = self.initialize_exodus_output_database()

    def initialize_exodus_output_database(self):
        exo = exodus(self.output_file)
        exo.copy(self.genesis_mesh.genesis_file)
        return exo

    def setup_scalar_variables(self):
        pass

    def write_scalar_variable(self):
        pass
