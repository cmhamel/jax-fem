import jax
import jax.numpy as jnp
from jax import jit


class Solver:
    def __init__(self, solver_input_block=None,
                 n_nodes=None, n_dof_per_node=None,
                 connectivity=None):
        self.solver_input_block = solver_input_block
        self.n_nodes = n_nodes
        self.n_dof_per_node = n_dof_per_node
        self.connectivity = connectivity

    @staticmethod
    def print_solver_heading(time_step):
        print('-------------------------------------------------------')
        print('--- Time step %s' % time_step)
        print('-------------------------------------------------------')
        print('Iteration\t\t|R|\t\t|du|')

    @staticmethod
    def print_solver_state(increment, residual_error, increment_error):
        print('\t{0:4}\t\t{1:.8e}\t{2:.8e}'.format(increment, residual_error[0], increment_error[0]))




