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

        # self.D = (self.youngs_modulus / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))) * \
        #          jnp.array([[1.0 - self.poissons_ratio, self.poissons_ratio, 0.0],
        #                     [self.poissons_ratio, 1.0 - self.poissons_ratio, 0.0],
        #                     [0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poissons_ratio)]], dtype=jnp.float64)
        self.D = (self.youngs_modulus / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))) * \
                 jnp.array([[1.0, self.poissons_ratio, 0.0],
                            [self.poissons_ratio, 1.0, 0.0],
                            [0.0, 0.0, 0.5 * (1.0 - self.poissons_ratio)]], dtype=jnp.float64)
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
        force_vector = jnp.zeros(self.K.shape[0], dtype=jnp.float64)
        stiffness_matrix = self.K

        force_for_reaction = force_vector[self.displacement_bc_nodes]
        stiffness_for_reaction = stiffness_matrix[self.displacement_bc_nodes, self.displacement_bc_nodes]

        # penalty method used for enforcing displacement boundary conditions
        #
        penalty = 1.0e12 * jnp.trace(stiffness_matrix) / stiffness_matrix.shape[0]
        # penalty = 1.0
        force_vector = jax.ops.index_update(force_vector, jax.ops.index[self.displacement_bc_nodes],
                                            penalty * self.displacement_bc_values)
        stiffness_matrix = jax.ops.index_update(stiffness_matrix,
                                                jax.ops.index[self.displacement_bc_nodes, self.displacement_bc_nodes],
                                                penalty)

        # solve for the displacement vector
        #
        displacement_vector = jnp.linalg.solve(stiffness_matrix, force_vector)
        u_x = displacement_vector[::2]
        u_y = displacement_vector[1::2]
        u = jnp.vstack((u_x, u_y)).T
        print(u)

        # now get the reaction force
        #
        reaction = jnp.zeros_like(force_vector)
        stiffness_matrix = jax.ops.index_update(stiffness_matrix,
                                                jax.ops.index[self.displacement_bc_nodes, self.displacement_bc_nodes],
                                                stiffness_for_reaction)
        reaction_temp = jnp.matmul(stiffness_matrix[self.displacement_bc_nodes, :], displacement_vector) - \
                        force_for_reaction
        reaction = jax.ops.index_update(reaction, jax.ops.index[self.displacement_bc_nodes], reaction_temp)
        reaction_x = reaction[::2]
        reaction_y = reaction[1::2]
        reaction = jnp.vstack((reaction_x, reaction_y)).T

        self.post_processor.exo.put_time(1, 0.0)
        self.post_processor.write_nodal_vector_variable('disp', 1, u)
        self.post_processor.write_nodal_vector_variable('reaction', 1, reaction)

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

        return stiffness_matrix

    def assemble_force_vector(self):
        force_vector = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node, dtype=jnp.float64)
        coordinates = self.genesis_mesh.nodal_coordinates[self.genesis_mesh.connectivity]
        dof_connectivity = self.dof_connectivity

        def element_calculation(e, input):
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
        pass
