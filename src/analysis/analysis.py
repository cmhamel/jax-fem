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

        # setup
        #
        self.mesh = self._setup_mesh()
        self.variables = self._setup_variables()
        self.kernels = self._setup_kernels()
        self.boundary_conditions = self._setup_boundary_conditions()

        # other stuff
        #
        self.number_of_variables = len(self.variables)
        self.number_of_kernels = len(self.kernels)

        # jit stuff
        #
        self.update_solution = jit(self.update_solution)

    def __str__(self) -> str:
        string = 'Analysis:\n'
        string = string + self.mesh.__str__() + '\n\n'
        string = string + 'Variables = '
        for n in range(len(self.variables)):
            string = string + self.variables[n] + ' '
        string = string + '\n\n'
        string = string + 'Kernels:\n'
        for kernel in self.kernels:
            string = string + kernel.__str__()
        string = string + 'Boundary Conditions:\n'
        for boundary_condition in self.boundary_conditions:
            string = string + boundary_condition.__str__()
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
            kernels.append(kernel_factory(kernel))
        return kernels

    def _setup_boundary_conditions(self) -> list:
        boundary_conditions = list()
        for boundary_condition in self.input_settings['boundary_conditions']:
            boundary_conditions.append(boundary_condition_factory(boundary_condition))

        for n in range(len(boundary_conditions)):
            boundary_conditions[n].populate_node_set_nodes_from_mesh(self.mesh)

        return boundary_conditions

    def calculate_total_residual(self, u):

        # TODO: need to update for coupled problems
        #
        # calculate residual
        #
        residual = jnp.zeros_like(u)
        for kernel in self.kernels:
            residual = residual + kernel.calculate_residual(self.mesh.nodal_coordinates,
                                                            self.mesh.element_connectivity,
                                                            u)

        # modify residual to satisfy bcs
        #
        for bc in self.boundary_conditions:
            residual = bc.modify_residual_vector_to_satisfy_boundary_conditions(residual)

        return residual

    def update_solution(self, u: jnp.ndarray) -> tuple:
        # store by number of variables
        #
        # u = jnp.zeros((self.mesh.number_of_nodes, self.number_of_variables))
        # u = jnp.zeros(self.mesh.number_of_nodes)

        # TODO: need to update based on the variable on the dirichlet BC
        #
        for bc in self.boundary_conditions:
            u = bc.modify_solution_vector_to_satisfy_boundary_conditions(u)

        residual = self.calculate_total_residual(u)
        tangent = jacfwd(self.calculate_total_residual)(u)

        for bc in self.boundary_conditions:
            tangent = bc.modify_tangent_matrix_to_satisfy_boundary_conditions(tangent)

        delta_u, _ = jax.scipy.sparse.linalg.gmres(tangent, residual)
        u = u - delta_u

        residual_norm = jnp.linalg.norm(residual)
        delta_u_norm = jnp.linalg.norm(delta_u)

        return u, residual_norm, delta_u_norm

    def run_analysis(self) -> None:
        u = jnp.zeros(self.mesh.number_of_nodes)

        residual_norm, delta_u_norm = 1.0, 1.0
        n = 0
        print('{0:16}{1:16}{2:16}'.format('Iteration', '|R|', '|dU|'))
        while residual_norm > 1e-8 or delta_u_norm > 1e-8:
            u, residual_norm, delta_u_norm = self.update_solution(u)
            print('{0:8}\t{1:.8e}\t{2:.8e}'.format(n, residual_norm, delta_u_norm))
            n = n + 1

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(self.mesh.nodal_coordinates, u)
        plt.show()






