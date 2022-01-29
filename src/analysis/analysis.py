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
        print('Setting up mesh...')
        self.mesh = self._setup_mesh()
        print('Setting up variables...')
        self.variables = self._setup_variables()
        print('Setting up kernels...')
        self.kernels = self._setup_kernels()
        print('Setting up boundary conditions...')
        self.boundary_conditions = self._setup_boundary_conditions()
        print('Setting up boundary condition methods...')
        self.bc_solution_methods, \
        self.bc_residual_methods, \
        self.bc_tangent_methods = self._setup_boundary_condition_methods()
        print('Setting up element level residual methods...')
        self.element_level_residual_methods = self._setup_element_level_residual_methods()
        print('Setting up residual methods...')
        self.residual_methods = self._setup_residual_methods()
        self.tangent_methods_diagonal = None
        self.tangent_methods_off_diagonal = None
        # assert False
        print('Setting up solver...')
        self.solver = self._setup_solver()
        print('Setting up post processor...')
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

    def _setup_boundary_condition_methods(self) -> tuple:

        # first organize the methods by variable
        #
        variable_bc_solution_methods = []
        variable_bc_residual_methods = []
        variable_bc_tangent_methods = []
        for variable in self.variables:

            # methods for enforcing bcs at the global level
            #
            temp_bc_solution_methods = []
            temp_bc_residual_methods = []
            temp_bc_tangent_methods = []

            for bc in self.boundary_conditions:
                if bc.variable == variable:
                    temp_bc_solution_methods.append(bc.modify_solution_vector_to_satisfy_boundary_conditions)
                    temp_bc_residual_methods.append(bc.modify_residual_vector_to_satisfy_boundary_conditions)
                    temp_bc_tangent_methods.append(bc.modify_tangent_matrix_to_satisfy_boundary_conditions)

            variable_bc_solution_methods.append(temp_bc_solution_methods)
            variable_bc_residual_methods.append(temp_bc_residual_methods)
            variable_bc_tangent_methods.append(temp_bc_tangent_methods)

        # now build lambdas for each variable
        #
        solution_methods = []
        residual_methods = []
        tangent_methods = []
        for n in range(len(self.variables)):

            # make solution methods
            #
            if len(variable_bc_solution_methods[n]) < 1:
                solution_methods.append(lambda u: u)
            else:
                temp_solution_method = lambda u, m=n, j=0: variable_bc_solution_methods[m][j](u)
                if len(variable_bc_solution_methods[n]) > 1:
                    for i in range(1, len(variable_bc_solution_methods[n])):
                        temp_solution_method = lambda u, m=n, j=i, f=temp_solution_method: \
                            variable_bc_solution_methods[m][j](f(u))

                solution_methods.append(temp_solution_method)

            # now make residual methods
            #
            if len(variable_bc_residual_methods[n]) < 1:
                residual_methods.append(lambda u: u)
            else:
                temp_residual_method = lambda u, m=n, j=0: variable_bc_residual_methods[m][j](u)
                if len(variable_bc_residual_methods[n]) > 1:
                    for i in range(1, len(variable_bc_residual_methods[n])):
                        temp_residual_method = lambda u, m=n, j=i, f=temp_residual_method: \
                            variable_bc_residual_methods[m][j](f(u))

                residual_methods.append(temp_residual_method)

            # now make tangent methods
            #
            if len(variable_bc_tangent_methods[n]) < 1:
                tangent_methods.append(lambda u: u)
            else:
                temp_tangent_method = lambda u, m=n, j=0: variable_bc_tangent_methods[m][j](u)
                if len(variable_bc_tangent_methods[n]) > 1:
                    for i in range(1, len(variable_bc_tangent_methods[n])):
                        temp_tangent_method = lambda u, m=n, j=i, f=temp_tangent_method: \
                            variable_bc_tangent_methods[m][j](f(u))

                tangent_methods.append(temp_tangent_method)

        return solution_methods, residual_methods, tangent_methods

    def _setup_element_level_residual_methods(self) -> tuple:

        # loop over variables to gather kernels
        #
        variable_kernels = []
        variable_element_level_residual_methods = []
        # TODO: add off diagonal tangent terms
        for variable in self.variables:
            temp_variable_kernels = []
            temp_element_level_residual_methods = []
            temp_element_level_tangent_diagonal_methods = []

            for kernel in self.kernels:
                if kernel.variable == variable:
                    temp_variable_kernels.append(kernel)
                    temp_element_level_residual_methods.append(kernel.calculate_element_level_residual)

            # add to global arrays
            #
            variable_kernels.append(temp_variable_kernels)
            variable_element_level_residual_methods.append(temp_element_level_residual_methods)

        # TODO: compose residual and tangent methods here
        #
        residual_methods = []
        for n in range(len(self.variables)):

            # make lambdas
            #
            element_level_residual_method = lambda x, y, m=n, j=0: variable_element_level_residual_methods[m][j](x, y)

            if len(variable_element_level_residual_methods[n]) > 1:
                print('here')
                for i in range(1, len(variable_element_level_residual_methods[n])):
                    element_level_residual_method = lambda x, y, m=n, j=i, r=element_level_residual_method: \
                        r(x, y) + variable_element_level_residual_methods[m][j](x, y)

            residual_methods.append(element_level_residual_method)

        return residual_methods

    def assemble_residual(self,
                          element_level_residual_method,
                          nodal_coordinates,
                          connectivity,
                          u):

        element_coordinates = nodal_coordinates[connectivity]
        element_us = u[connectivity]
        residual = jnp.zeros_like(u)

        def scan_body(residual_temp, inputs):
            element_level_coordinates, element_level_connectivity, element_level_us = inputs
            element_level_residual = element_level_residual_method(element_level_coordinates,
                                                                   element_level_us)
            residual_temp = jax.ops.index_add(residual_temp,
                                              jax.ops.index[element_level_connectivity],
                                              element_level_residual)
            return residual_temp, inputs

        residual, _ = jax.lax.scan(scan_body,
                                   residual,
                                   (element_coordinates, connectivity, element_us))

        return residual

    def _setup_residual_methods(self):
        residual_methods = []
        for n in range(len(self.variables)):

            residual_method = lambda x, y, z, m=n: \
                self.assemble_residual(self.element_level_residual_methods[m], x, y, z)
            residual_method = lambda x, y, z, m=n, r=residual_method: \
                self.bc_residual_methods[m](r(x, y, z))

            # TODO: try to bootstrap the bc update residual methods on top
            #
            residual_methods.append(residual_method)

        return residual_methods

    def _setup_residual_and_tangent_methods_old(self):
        """
        Try to set up a method to pre figure out what the residual methods are
        for each variable based on the kernels supplied
        :return:
        """

        # TODO: change everything to do with kernels such that you just define
        # TODO: an element level residual method, then the element level total residual
        # TODO: constructed for all the kernels on a variable will be composed here
        # TODO: and then fed into a general purpose assembly method such that
        # TODO: we only lax.scan over the set of elements once for each variable
        # TODO: the same should be done for the tangents so they're autodiffed at the element
        # TODO: level and lax.scanned only once
        #
        # TODO: even better would be to only define a quadrature level method
        # TODO: but that would complicate things requiring all kernels for a variable
        # TODO: to have the same quadrature rule

        variable_kernels = []
        kernel_residual_methods = []
        bc_residual_methods = []
        bc_tangent_methods_diagonal = []
        for variable in self.variables:
            temp_variable_kernels = []
            temp_kernel_residual_methods = []
            temp_bc_residual_methods = []
            temp_bc_tangent_methods_diagonal = []
            for kernel in self.kernels:
                print(kernel)
                if kernel.variable == variable:
                    temp_variable_kernels.append(kernel)
                    residual_method = kernel.calculate_residual
                    temp_kernel_residual_methods.append(residual_method)

            for bc in self.boundary_conditions:
                if bc.variable == variable:
                    temp_bc_residual_methods.append(bc.modify_residual_vector_to_satisfy_boundary_conditions)
                    temp_bc_tangent_methods_diagonal.append(bc.modify_tangent_matrix_to_satisfy_boundary_conditions)

            variable_kernels.append(temp_variable_kernels)
            kernel_residual_methods.append(temp_kernel_residual_methods)
            bc_residual_methods.append(temp_bc_residual_methods)
            bc_tangent_methods_diagonal.append(temp_bc_tangent_methods_diagonal)

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
        tangent_methods_diagonal = []
        for n in range(len(residual_methods)):
            if residual_methods[n] is not None:
                temp_bc_tangent_methods_diagonal = bc_tangent_methods_diagonal[n]
                tangent_method = jacfwd(residual_methods[n], argnums=2)

                for bc_tangent_method in temp_bc_tangent_methods_diagonal:
                    tangent_method = lambda x, y, z, k=tangent_method, temp_bc_tangent_method=bc_tangent_method: \
                        temp_bc_tangent_method(k(x, y, z))

                tangent_methods_diagonal.append(jit(tangent_method))
            else:
                tangent_methods_diagonal.append(None)

        # TODO: fill out the off diagonal tangents
        #
        return residual_methods, tangent_methods_diagonal, None

    def _setup_solver(self) -> SolverBaseClass:
        return solver_factory(self.solver_input_settings,
                              self.variables,
                              self.kernels,
                              self.boundary_conditions,
                              self.bc_solution_methods,
                              self.residual_methods,
                              self.tangent_methods_diagonal,
                              self.tangent_methods_off_diagonal)

    def _setup_post_processor(self) -> PostProcessorBaseClass:
        self.post_processor_input_settings['mesh_file'] = self.mesh.genesis_file
        return post_processor_factory(self.post_processor_input_settings)

    # public access
    #
    def run_analysis(self) -> None:
        u = jnp.zeros((self.mesh.number_of_nodes, len(self.variables)))

        # post processing setup TODO: move this to somewhere else
        #
        self.post_processor.write_time_step(0, 0.0)
        for n, variable in enumerate(self.variables):
            self.post_processor.write_nodal_values(variable, 0, np.array(u[:, n]))

        residual_norm, delta_u_norm = 1.0, 1.0
        n = 0
        print('\n\n\n{0:16}{1:16}{2:16}'.format('Iteration', '|R|', '|dU|'))
        while residual_norm > 1e-8 or delta_u_norm > 1e-8:
        # while residual_norm > 1e-8:
            # u, residual_norm, delta_u_norm = self.solver.update_solution(self.mesh.nodal_coordinates,
            #                                                              self.mesh.element_connectivity,
            #                                                              u)
            u, residual_norm, residual_norms, delta_u_norm, delta_u_norms = \
                self.solver.update_solution(self.mesh.nodal_coordinates, self.mesh.element_connectivity, u)
            print('{0:8}\t{1:.8e}\t{2:.8e}'.format(n, residual_norm, delta_u_norm))
            for m in range(len(residual_norms)):
                print('{0:8}\t  |R_{1:}| = {2:.8e}\t  |d{3:}| = {4:.8e}'.format('',
                                                                                self.variables[m],
                                                                                residual_norms[m],
                                                                                self.variables[m],
                                                                                delta_u_norms[m]))
            # print(residual_norms)
            n = n + 1

        self.post_processor.write_time_step(1, 1.0)
        for n, variable in enumerate(self.variables):
            self.post_processor.write_nodal_values(variable, 1, np.array(u[:, n]))
        self.post_processor.exo.close()

    def run_analysis_old_and_working_for_single_variable(self) -> None:
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






