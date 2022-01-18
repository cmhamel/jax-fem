import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import jacfwd
from mesh import MeshBaseClass
from mesh import mesh_factory
from kernels import kernel_factory
from boundary_conditions import boundary_condition_factory
from boundary_conditions import BoundaryConditionBaseClass
from solvers import solver_factory
from solvers import SolverBaseClass
from post_processors import post_processor_factory
from post_processors import PostProcessorBaseClass


class Analysis:
    def __init__(self, input_settings) -> None:

        # various dicts of input settings
        #
        self.input_settings = input_settings
        self.number_of_dimensions = input_settings['number_of_dimensions']
        self.mesh_input_settings = input_settings['mesh']
        self.variables_input_settings = input_settings['variables']
        self.kernels_input_settings = input_settings['kernels']
        self.boundary_conditions_input_settings = input_settings['boundary_conditions']
        self.solver_input_settings = input_settings['solver']
        self.post_processor_input_settings = input_settings['post_processor']

        # setup
        #
        self.mesh = self._setup_mesh()
        self.variables = self._setup_variables()
        self.kernels = self._setup_kernels()
        self.boundary_conditions = self._setup_boundary_conditions()
        self.solver = self._setup_solver()
        self.post_processor = self._setup_post_processor()

        # other stuff
        #
        self.number_of_variables = len(self.variables)
        self.number_of_kernels = len(self.kernels)

        # jit stuff
        #
        # self.update_solution = jit(self.update_solution)

    def __str__(self) -> str:
        string = '\n\n\n' + __class__.__name__ + ':\n'
        string = string + self.mesh.__str__() + '\n\n'
        string = string + '  Variables = '
        for n in range(len(self.variables)):
            string = string + self.variables[n] + ' '
        string = string + '\n\n'
        string = string + '  Kernels:\n'
        for kernel in self.kernels:
            string = string + kernel.__str__()
        string = string + '\n'
        string = string + '  Boundary Conditions:\n'
        for boundary_condition in self.boundary_conditions:
            string = string + boundary_condition.__str__()
        string = string + '\n'
        string = string + self.solver.__str__() + '\n'
        string = string + self.post_processor.__str__() + '\n'
        return string

    def _setup_mesh(self) -> MeshBaseClass:
        return mesh_factory(self.mesh_input_settings, self.number_of_dimensions)

    def _setup_variables(self) -> list:
        variables = list()
        for variable in self.variables_input_settings:
            variables.append(variable)
        return variables

    def _setup_kernels(self) -> list:
        kernels = list()
        for kernel in self.kernels_input_settings:
            kernels.append(kernel_factory(kernel, self.number_of_dimensions))
        return kernels

    def _setup_boundary_conditions(self) -> list:
        boundary_conditions = list()
        for boundary_condition in self.input_settings['boundary_conditions']:
            boundary_conditions.append(boundary_condition_factory(boundary_condition))

        for n in range(len(boundary_conditions)):
            boundary_conditions[n].populate_node_set_nodes_from_mesh(self.mesh)

        return boundary_conditions

    def _setup_solver(self) -> SolverBaseClass:
        return solver_factory(self.solver_input_settings,
                              self.variables,
                              self.kernels,
                              self.boundary_conditions)

    def _setup_post_processor(self) -> PostProcessorBaseClass:
        self.post_processor_input_settings['mesh_file'] = self.mesh.genesis_file
        return post_processor_factory(self.post_processor_input_settings)

    # public access
    #
    def run_analysis(self) -> None:
        u = jnp.zeros(self.mesh.number_of_nodes)
        self.post_processor.write_time_step(0, 0.0)
        self.post_processor.write_nodal_values('u', 0, np.array(u))

        residual_norm, delta_u_norm = 1.0, 1.0
        n = 0
        print('{0:16}{1:16}{2:16}'.format('Iteration', '|R|', '|dU|'))
        while residual_norm > 1e-8 or delta_u_norm > 1e-8:
            u, residual_norm, delta_u_norm = self.solver.update_solution(self.mesh.nodal_coordinates,
                                                                         self.mesh.element_connectivity,
                                                                         u)
            print('{0:8}\t{1:.8e}\t{2:.8e}'.format(n, residual_norm, delta_u_norm))
            n = n + 1

        self.post_processor.write_time_step(1, 1.0)
        self.post_processor.write_nodal_values('u', 1, np.array(u))
        self.post_processor.exo.close()






