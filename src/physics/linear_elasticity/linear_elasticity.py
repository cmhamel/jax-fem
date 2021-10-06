import jax
import jax.numpy as jnp
from jax import jit
from util import general_tardigrade_error
from elements import QuadElement
from solvers import NewtonRaphsonSolver
from ..physics import Physics
from .boundary_conditions import DisplacementBoundaryCondition


class LinearElasticity(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(LinearElasticity, self).__init__(n_dimensions, physics_input)
        print(self)
        self.n_dof_per_node = 2
        print('\n\n\n')
        print('Linear elasticity:')
        # need to make dof connectivity matrix, TODO need to do this for each block eventually but currently
        #                                       TODO everything is operating on one block
        #
        self.dof_connectivity = self.genesis_mesh.make_multiple_dof_connectivity(self.n_dof_per_node)

        # read in blocks
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):
            # set up elements
            #
            self.element_objects.append(
                QuadElement(quadrature_order=block['cell_interpolation']['quadrature_order'],
                            shape_function_order=block['cell_interpolation']['shape_function_order']))
            print('Block number = %s' % str(n + 1))

        self.displacement_bcs = []
        if 'displacement' in self.boundary_conditions_input_block.keys():
            for n, bc in enumerate(self.boundary_conditions_input_block['displacement']):
                self.displacement_bcs.append(DisplacementBoundaryCondition(bc, self.n_dof_per_node,
                                                                           self.genesis_mesh, n))
                print(self.displacement_bcs[n])

        # set up total BC arrays
        #
        self.displacement_bc_nodes = jnp.array((), dtype=jnp.int32)
        self.displacement_bc_values = jnp.array((), dtype=jnp.float64)
        for bc in self.displacement_bcs:
            self.displacement_bc_nodes = jnp.hstack((self.displacement_bc_nodes, bc.bc_nodes))
            self.displacement_bc_values = jnp.hstack((self.displacement_bc_values, bc.values))

        print(self.displacement_bc_nodes)
        print(type(self.displacement_bc_nodes))

        # set up force boundary conditions
        #
        # TODO

        # read properties
        #
        self.youngs_modulus = self.physics_input['properties']['youngs_modulus']
        self.poissons_ratio = self.physics_input['properties']['poissons_ratio']
        print('Properties:')
        print('\tYoung\'s modulus = {0:.4e} (MPa)'.format(self.youngs_modulus))
        print('\tPoisson\'s ratio = {0:.4e} (Unitless)'.format(self.poissons_ratio))

        self.D = (self.youngs_modulus / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))) * \
                 jnp.array([[1.0 - self.poissons_ratio, self.poissons_ratio, 0.0],
                            [self.poissons_ratio, 1.0 - self.poissons_ratio, 0.0],
                            [0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poissons_ratio)]], dtype=jnp.float64)
        print('D = ')
        print(self.D)

        # jit the assembly operations and element calculators
        #
        self.jit_calculate_element_level_stiffness_matrix = jit(self.calculate_element_level_stiffness_matrix)
        self.jit_calculate_element_level_force_vector = jit(self.calculate_element_level_force_vector)

        self.jit_assemble_stiffness_matrix = jit(self.assemble_stiffness_matrix)
        self.jit_assemble_force_vector = jit(self.assemble_force_vector)

        self.K = self.jit_assemble_stiffness_matrix()
        # print(self.K)
        self.solve()

    def solve(self):
        stiffness_matrix = self.K
        penalty = 1.0e6 * jnp.trace(stiffness_matrix) / stiffness_matrix.shape[0]
        stiffness_matrix = jax.ops.index_update(stiffness_matrix,
                                                jax.ops.index[self.displacement_bc_nodes, self.displacement_bc_nodes],
                                                penalty)

    def assemble_stiffness_matrix(self):
        stiffness_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node,
                                      self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node),
                                     dtype=jnp.float64)
        coordinates = self.genesis_mesh.nodal_coordinates[self.genesis_mesh.connectivity]
        dof_connectivity = self.dof_connectivity

        def element_calculation(e, input):
            stiffness_matrix_temp = input
            K_e = self.jit_calculate_element_level_stiffness_matrix(coordinates[e])
            indices = jnp.ix_(dof_connectivity[e], dof_connectivity[e])
            stiffness_matrix_temp = jax.ops.index_add(stiffness_matrix_temp, jax.ops.index[indices], K_e)
            return stiffness_matrix_temp

        stiffness_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                             stiffness_matrix)

        # modify to satisfy BCs
        # #
        # penalty = 1.0e6 * jnp.trace(stiffness_matrix) / stiffness_matrix.shape[0]
        # stiffness_matrix = jax.ops.index_update(stiffness_matrix,
        #                                         jax.ops.index[self.displacement_bc_nodes, self.displacement_bc_nodes],
        #                                         penalty)

        return stiffness_matrix

    def assemble_force_vector(self):
        pass

    def calculate_element_level_stiffness_matrix(self, input):
        n_dof_per_element = self.genesis_mesh.n_nodes_per_element[0] * self.n_dof_per_node
        K_e = jnp.zeros((n_dof_per_element, n_dof_per_element), dtype=jnp.float64)
        coords = input
        B_element = self.element_objects[0].calculate_B_matrix(coords)
        JxW_element = self.element_objects[0].calculate_JxW(coords)
        D = self.D

        def quadrature_calculation(q, K_element):
            B = B_element[q, :, :]
            JxW = JxW_element[q, 0]
            K_q = JxW * jnp.matmul(jnp.matmul(B.T, D), B)
            K_element = jax.ops.index_add(K_element, jax.ops.index[:, :], K_q)
            return K_element

        K_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, K_e)

        return K_e

    def calculate_element_level_force_vector(self):
        force_vector = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node, dtype=jnp.float64)
        coordinates = self.genesis_mesh.nodal_coordinates[self.genesis_mesh.connectivity]
        dof_connectivity = self.dof_connectivity

        def element_calculation(e, input):
            pass
