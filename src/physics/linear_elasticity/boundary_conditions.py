import jax
import jax.numpy as jnp
from util import general_tardigrade_error


class DisplacementBoundaryCondition:
    def __init__(self, displacement_bc_input_block, n_dof, genesis_mesh, node_set_number):
        self.node_set_name = displacement_bc_input_block['node_set']
        self.type = displacement_bc_input_block['type']
        self.components = displacement_bc_input_block['components']

        # get the bc nodes
        #
        self.bc_nodes = genesis_mesh.modify_node_list_for_multiple_dofs(n_dof, node_set_number)

        # now prune the nodes based on the components
        #
        if self.components == ['x', 'y']:
            pass
        elif self.components == ['x']:
            # remove the first element and every other two
            #
            self.bc_nodes = jnp.delete(self.bc_nodes, jax.ops.index[::n_dof])
        elif self.components == ['y']:
            # remove the second element and every other two
            #
            self.bc_nodes = jnp.delete(self.bc_nodes, jax.ops.index[1::n_dof])
        else:
            general_tardigrade_error('Unsupported component in DisplacementBoundaryCondition')

        if displacement_bc_input_block['type'] == 'fixed':
            self.values = jnp.zeros(self.bc_nodes.shape[0], dtype=jnp.float64)
        elif displacement_bc_input_block['type'] == 'prescribed':
            try:
                value = displacement_bc_input_block['value']
                self.values = value * jnp.ones(self.bc_nodes.shape[0], dtype=jnp.float64)
            except KeyError:
                general_tardigrade_error('Need a value for a prescribed displacement boundary condition')
        else:
            general_tardigrade_error('Unsupported dispalcement boundary condition')

    def __str__(self):
        string = 'Displacement BC:' + '\n' + \
                 '\tNode set name = ' + str(self.node_set_name) + '\n' + \
                 '\tType          = ' + str(self.type) + '\n' + \
                 '\tComponents    = ' + str(self.components)
        return string

