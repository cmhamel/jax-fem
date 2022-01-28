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
        self.residual_methods, \
        self.tangent_methods = self._setup_residual_and_tangent_methods()
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

    def _setup_residual_and_tangent_methods(self):
        """
        Try to set up a method to pre figure out what the residual methods are
        for each variable based on the kernels supplied
        :return:
        """
        variable_kernels = []
        kernel_residual_methods = []
        bc_residual_methods = []
        bc_tangent_methods = []
        for variable in self.variables:
            temp_variable_kernels = []
            temp_kernel_residual_methods = []
            temp_bc_residual_methods = []
            temp_bc_tangent_methods = []
            for kernel in self.kernels:
                print(kernel)
                if kernel.variable == variable:
                    temp_variable_kernels.append(kernel)
                    residual_method = kernel.calculate_residual
                    temp_kernel_residual_methods.append(residual_method)

            for bc in self.boundary_conditions:
                if bc.variable == variable:
                    temp_bc_residual_methods.append(bc.modify_residual_vector_to_satisfy_boundary_conditions)
                    temp_bc_tangent_methods.append(bc.modify_tangent_matrix_to_satisfy_boundary_conditions)

            variable_kernels.append(temp_variable_kernels)
            kernel_residual_methods.append(temp_kernel_residual_methods)
            bc_residual_methods.append(temp_bc_residual_methods)
            bc_tangent_methods.append(temp_bc_tangent_methods)

        residual_methods = []
        for n in range(len(kernel_residual_methods)):
            temp_kernels = kernel_residual_methods[n]
            temp_bc_residual_methods = bc_residual_methods[n]

            # this is the dumb case that this variable isn't active in any kernels
            # add a 'None' just to make the lists the same sizes
            #
            if len(temp_kernels) < 1:
                residual_methods.append(None)
                continue

            # make lambdas
            #
            residual_method = lambda x, y, z: temp_kernels[0](x, y, z)
            if len(temp_kernels) > 1:
                for i in range(1, len(temp_kernels)):
                    residual_method = lambda x, y, z, k=residual_method, j=i: \
                        k(x, y, z) + temp_kernels[j](x, y, z)

            # apply boundary conditions to residual
            #
            for bc_residual_method in temp_bc_residual_methods:
                residual_method = lambda x, y, z, k=residual_method, temp_bc_residual_method=bc_residual_method: \
                    temp_bc_residual_method(k(x, y, z))

            residual_methods.append(jit(residual_method))

        # make tangent methods
        #
        tangent_methods = []
        for n in range(len(residual_methods)):
            if residual_methods[n] is not None:
                temp_bc_tangent_methods = bc_tangent_methods[n]
                tangent_method = jacfwd(residual_methods[n], argnums=2)

                for bc_tangent_method in temp_bc_tangent_methods:
                    tangent_method = lambda x, y, z, k=tangent_method, temp_bc_tangent_method=bc_tangent_method: \
                        temp_bc_tangent_method(k(x, y, z))

                tangent_methods.append(jit(tangent_method))
            else:
                tangent_methods.append(None)

        return residual_methods, tangent_methods

    def _setup_solver(self) -> SolverBaseClass:
        return solver_factory(self.solver_input_settings,
                              self.variables,
                              self.kernels,
                              self.boundary_conditions,
                              self.residual_methods,
                              self.tangent_methods)

    def _setup_post_processor(self) -> PostProcessorBaseClass:
        self.post_processor_input_settings['mesh_file'] = self.mesh.genesis_file
        return post_processor_factory(self.post_processor_input_settings)

    # public access
    #
    def run_analysis(self) -> None:
        u = jnp.zeros(len(self.variables) * self.mesh.number_of_nodes)
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






