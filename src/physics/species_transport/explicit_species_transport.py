import jax
import jax.numpy as jnp
from jax import jit
from elements import LineElement
from elements import QuadElement
from solvers import FirstOrderExplicit
from ..physics import Physics
from ..initial_conditions import InitialCondition
from ..boundary_conditions import DirichletBoundaryCondition
from ..boundary_conditions import NeumannBoundaryCondition
from ..constitutive_models import FicksLaw


class ExplicitSpeciesTransport(Physics):
    """
    Will solve
    Mc_dot + R(c) = 0
    c_dot = -M^-1 R(c)
    c_n+1 = c_n - delta_t * M^-1 R(c)
    """
    def __init__(self, n_dimensions, physics_input_block):
        super(ExplicitSpeciesTransport, self).__init__(n_dimensions, physics_input_block)

        print(self)

        # set number of degress of freedom per node
        #
        self.n_species = self.physics_input['number_of_species']
        self.n_dof_per_node = self.n_species
        print(self.n_dof_per_node)

        # get some boundary conditions
        #
        self.dirichlet_bcs = []
        self.dirichlet_bcs_nodes = []
        self.dirichlet_bcs_values = []
        for key in self.boundary_conditions_input_block.keys():
            bc_type = self.boundary_conditions_input_block[key]
            for i, bc in enumerate(bc_type):
                self.dirichlet_bcs.append(
                    DirichletBoundaryCondition(dirichlet_bc_input_block=bc,
                                               node_set_name=bc['node_set'],
                                               node_set_nodes=self.genesis_mesh.node_set_nodes[i]))
                self.dirichlet_bcs_nodes.append(self.dirichlet_bcs[i].node_set_nodes)
                self.dirichlet_bcs_values.append(self.dirichlet_bcs[i].values)

        self.dirichlet_bcs_nodes = jnp.array(self.dirichlet_bcs_nodes)
        self.dirichlet_bcs_values = jnp.array(self.dirichlet_bcs_values)

        # initialize the element type
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):

            # constitutive model
            #
            block_constitutive_models = []
            for s in range(self.n_species):
                block_constitutive_models.append(
                    FicksLaw(self.blocks_input_block[n]['constitutive_model'][s]))

            self.constitutive_models.append(block_constitutive_models)

            # set up elements
            #
            if self.n_dimensions == 1:
                self.element_objects.append(
                    LineElement(quadrature_order=block['cell_interpolation']['quadrature_order'],
                                shape_function_order=block['cell_interpolation']['shape_function_order']))
            elif self.n_dimensions == 2:
                self.element_objects.append(
                    QuadElement(quadrature_order=block['cell_interpolation']['quadrature_order'],
                                shape_function_order=block['cell_interpolation']['shape_function_order']))
            else:
                try:
                    assert False
                except AssertionError:
                    raise Exception('Unsupported number of dimensions in PoissonEquation')

            # print details about the blocks
            #
            print('Block number = %s' % str(n + 1))
            for s in range(self.n_species):
                print(self.constitutive_models[n][s])

        # set up the initial conditions
        #
        node_list = jnp.arange(0, self.genesis_mesh.nodal_coordinates.shape[0])
        initial_conditions = []
        for n, ic in enumerate(self.initial_conditions_input_block):
            initial_conditions.append(InitialCondition(ic_type=ic['type'],
                                                       block_nodes=node_list,
                                                       value=ic['value']))

        self.c_old = initial_conditions[0].values
        self.post_process_2d(1, 0.0)

        # self.mass_matrix_inv = jnp.linalg.inv(self.assemble_mass_matrix())

        # set up solver
        #
        self.solver = FirstOrderExplicit({},
                                         len(self.genesis_mesh.nodal_coordinates),
                                         1,
                                         self.genesis_mesh.connectivity,
                                         self.enforce_bcs_on_u,
                                         self.enforce_bcs_on_residual,
                                         self.enforce_bcs_on_tangent,
                                         # self.assemble_linear_system,
                                         self.assemble_stiffness_matrix,
                                         self.assemble_mass_matrix,
                                         property=self.constitutive_models[0][0].D)

        # initialize mass matrix
        #
        # self.mass_matrix = self.assemble_mass_matrix()
        # self.mass_matrix_inv = jnp.linalg.inv(self.assemble_mass_matrix())
        self.jit_solve = jit(self.solver.solve)

    @staticmethod
    def enforce_bcs_on_u(i, input):
        u_temp, bcs_nodes, bcs_values = input
        bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
        u_temp = jax.ops.index_update(u_temp, jax.ops.index[bc_nodes], bc_values)
        return u_temp, bcs_nodes, bcs_values

    # force residual to be 0 on dofs with dirchlet bcs
    #
    @staticmethod
    def enforce_bcs_on_residual(i, input):
        residual_temp, bcs_nodes = input
        bc_nodes = bcs_nodes[i]
        residual_temp = jax.ops.index_update(residual_temp, jax.ops.index[bc_nodes], 0.0)
        return residual_temp, bcs_nodes

    # enforce dirichlet BCs in the tangent matrix
    #
    @staticmethod
    def enforce_bcs_on_tangent(i, input):
        tangent_temp, bcs_nodes, bcs_values = input
        bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
        tangent_temp = jax.ops.index_update(tangent_temp, jax.ops.index[bc_nodes, bc_nodes], 1.0)
        return tangent_temp, bcs_nodes, bcs_values

    def calculate_element_level_mass_matrix(self, coords):
        M_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0], self.genesis_mesh.n_nodes_per_element[0]),
                        dtype=jnp.float64)
        JxW_element = self.element_objects[0].calculate_JxW(coords)

        def quadrature_calculation(q, M_element):
            N_xi = self.element_objects[0].N_xi[q, :, :]
            JxW = JxW_element[q, 0]
            M_q = JxW * jnp.matmul(N_xi, N_xi.transpose())
            M_element = jax.ops.index_add(M_element, jax.ops.index[:, :], M_q)
            return M_element

        M_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, M_e)

        return M_e

    def assemble_mass_matrix(self):
        mass_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node,
                                 self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node),
                                dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        jit_calculate_element_level_mass_matrix = jit(self.calculate_element_level_mass_matrix)

        def element_calculation(e, input):
            mass_matrix_temp = input
            M_e = jit_calculate_element_level_mass_matrix(coordinates[e])
            indices = jnp.ix_(connectivity[e], connectivity[e])
            mass_matrix_temp = jax.ops.index_add(mass_matrix_temp, jax.ops.index[indices], M_e)
            return mass_matrix_temp

        mass_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, mass_matrix)

        return mass_matrix

    def calculate_element_level_stiffness_matrix(self, coords):
        K_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0], self.genesis_mesh.n_nodes_per_element[0]),
                        dtype=jnp.float64)
        grad_N_X_element = self.element_objects[0].map_shape_function_gradients(coords)
        JxW_element = self.element_objects[0].calculate_JxW(coords)
        def quadrature_calculation(q, K_element):
            # N_xi = self.element_objects[0].N_xi[q, :, :]
            grad_N_X = grad_N_X_element[q, :, :]
            JxW = JxW_element[q, 0]
            D = self.constitutive_models[0][0].D
            K_q = JxW * D * jnp.matmul(grad_N_X, grad_N_X.transpose())
            K_element = jax.ops.index_add(K_element, jax.ops.index[:, :], K_q)
            return K_element

        K_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, K_e)

        return K_e

    def assemble_stiffness_matrix(self):
        mass_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node,
                                 self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node),
                                dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        jit_calculate_element_level_mass_matrix = jit(self.calculate_element_level_mass_matrix)

        def element_calculation(e, input):
            mass_matrix_temp = input
            M_e = jit_calculate_element_level_mass_matrix(coordinates[e])
            indices = jnp.ix_(connectivity[e], connectivity[e])
            mass_matrix_temp = jax.ops.index_add(mass_matrix_temp, jax.ops.index[indices], M_e)
            return mass_matrix_temp

        mass_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, mass_matrix)

        return mass_matrix

    def calculate_element_level_residual(self, nodal_fields):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        # coords, c_nodal, c_nodal_old = nodal_fields
        coords, c_nodal_old = nodal_fields
        R_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0]), dtype=jnp.float64)

        grad_N_X_element = self.element_objects[0].map_shape_function_gradients(coords)
        JxW_element = self.element_objects[0].calculate_JxW(coords)

        def quadrature_calculation(q, R_element):
            """
            :param q: the index of the current quadrature point
            :param R_element: the element level residual
            :return: the element level residual for this quadrature point only
            """
            grad_N_X = grad_N_X_element[q, :, :]
            JxW = JxW_element[q, 0]
            grad_c_q = jnp.matmul(grad_N_X.T, c_nodal_old).reshape((-1, 1))
            j_q = self.constitutive_models[0][0].calculate_species_flux(grad_c_q)
            R_q = JxW * jnp.matmul(grad_N_X, j_q)
            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def assemble_linear_system(self, u_old):

        # set up residual and grab connectivity for convenience
        #
        residual = jnp.zeros_like(u_old)
        connectivity = self.genesis_mesh.connectivity

        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        c_element_wise_old = u_old[connectivity]

        # jit the element level residual calculator
        #
        jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            # R_e = jit_calculate_element_level_residual((coordinates[e],
            #                                             c_element_wise[e], c_element_wise_old[e]))
            R_e = jit_calculate_element_level_residual((coordinates[e], c_element_wise_old[e]))
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        # adjust residual to satisfy dirichlet conditions
        #
        # residual, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
        #                                 self.enforce_bcs_on_residual, (residual, self.dirichlet_bcs_nodes))

        return residual

    def post_process_1d(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.show()

    def post_process_2d(self, time_step, time):
        self.post_processor.exo.put_time(time_step, time)
        self.post_processor.write_nodal_scalar_variable('c', time_step, jnp.asarray(self.c_old))
