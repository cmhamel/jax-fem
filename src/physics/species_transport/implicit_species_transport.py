import jax
import jax.numpy as jnp
from jax import jit, jacfwd
from elements import LineElement
from elements import QuadElement
# from time_control import TimeControl
# from solvers import NewtonRaphsonSolver
# from solvers import NewtonRaphsonTransientSolver
from solvers import FirstOrderImplicit
from solvers import Solver
from ..physics import Physics
from ..initial_conditions import InitialCondition
from ..boundary_conditions import DirichletBoundaryCondition
from ..boundary_conditions import NeumannBoundaryCondition
from ..constitutive_models import FicksLaw


class ImplicitSpeciesTransport(Physics):
    """
    Need to solve the following equation
    dcdt = \nabla(D\nabla c) + r(c)
    M\dot{c} = Kc + R(c)
    For an implicit solve do the following
    M(c_tau - c_t) / delta_t = Kc_tau + r(c_tau)
    MC_tau = Mc_t + delta_t * Kc_tau + delta_t * r(c_tau)
    forget r(c_tau) for now
    Mc_tau = Mc_t - delta_t * Kc_tau
    (M + delta_t * K)c_tau = Mc_t
    c_tau = Mc_t / (M + delta_t * K)
    """
    def __init__(self, n_dimensions, mesh_input, time_control):
        super(ImplicitSpeciesTransport, self).__init__(n_dimensions, mesh_input)
        print(self)

        # set number of degress of freedom per node
        #
        self.n_species = self.physics_input['number_of_species']
        self.n_dof_per_node = self.n_species
        self.time_control = time_control
        print(self.n_dof_per_node)
        # assert False

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
                                               node_set_nodes=self.genesis_mesh.node_set_nodes[i],
                                               time_end=self.time_control.time_end))
                self.dirichlet_bcs_nodes.append(self.dirichlet_bcs[i].node_set_nodes)
                self.dirichlet_bcs_values.append(self.dirichlet_bcs[i].values)

        self.dirichlet_bcs_nodes = jnp.array(self.dirichlet_bcs_nodes)
        self.dirichlet_bcs_values = jnp.array(self.dirichlet_bcs_values)

        # TODO add Neumann BC support
        #

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

        # enforce the dirichlet conditions upon the initial condition for both plotting
        # and for up front enforcement
        #
        self.c_old, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes),
                                             self.enforce_bcs_on_u,
                                             (self.c_old, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
        # self.u_old = self.c_old
        self.post_process_2d()

        # set up a dummy solver just for the print methods
        #
        self.dummy_solver = Solver()

        # jit some stuff
        #
        self.jit_assemble_residual = jit(self.assemble_residual)
        self.jit_assemble_tangent = jit(jacfwd(self.jit_assemble_residual))
        # self.jit_assemble_mass_matrix = jit(self.assemble_mass_matrix)
        # self.jit_assemble_stiffness_matrix = jit(self.assemble_stiffness_matrix)
        self.jit_gmres = jit(jax.scipy.sparse.linalg.gmres)

    def solve(self):

        self.dummy_solver.print_solver_heading(self.time_control.time_step_number)

        self.c_old, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes),
                                             self.enforce_bcs_on_u,
                                             (self.c_old, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

        # update dirichlet bcs
        #
        for n in range(len(self.dirichlet_bcs)):
            temp_values = self.dirichlet_bcs[n]. \
                update_bc_values(time=self.time_control.t)
            # print(temp_values)
            self.dirichlet_bcs_values = \
                jax.ops.index_update(self.dirichlet_bcs_values, jax.ops.index[n, :], temp_values)

        # print(self.dirichlet_bcs_values)


        c = jnp.ones_like(self.c_old)
        # apply dirichlet bcs here just in case
        #
        c, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes),
                                    self.enforce_bcs_on_u,
                                    (c, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

        n = 0
        while n <= 50:
            R = self.jit_assemble_residual(c)
            # tangent = self.jit_assemble_tangent(c)
            tangent = jacfwd(self.jit_assemble_residual)(c)
            # M = self.jit_assemble_mass_matrix()
            # K = self.jit_assemble_stiffness_matrix()
            # A = ((1 / self.time_control.time_increment) * M + K)
            # f = jnp.matmul(M, self.c_old) / self.time_control.time_increment

            # modify A and f matrix to satisfy BCs
            #
            # R, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes), self.enforce_bcs_on_residual,
            #                          (R, self.dirichlet_bcs_nodes))
            # f, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes), self.enforce_bcs_on_u,
            #                             (f, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
            c, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes),
                                        self.enforce_bcs_on_u,
                                        (c, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
            # A, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes), self.enforce_bcs_on_tangent,
            #                             (A, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
            # tangent, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes), self.enforce_bcs_on_tangent,
            #                                   (tangent, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

            # residual = f - jnp.matmul(A, self.c_old)
            # residual_norm = jnp.linalg.norm(residual)
            residual_norm = jnp.linalg.norm(R)

            if residual_norm < 1e-08:
                print('Converged on residual:  |R| = {0:.8e}'.format(residual_norm.ravel()[0]))
                c = self.c_old
                delta_c_norm = jnp.zeros(1)
                self.c_old = c
                return
            else:
                # c, _ = self.jit_gmres(A, f)
                # delta_c, _ = self.jit_gmres(A, R)
                delta_c, _ = self.jit_gmres(tangent, R)

                # delta_c = c - self.c_old
                c = c - delta_c
                print(type(c))
                # delta_c_norm = self.norm(delta_c)
                delta_c_norm = jnp.linalg.norm(delta_c)
                if delta_c_norm < 1e-08:
                    print('Converged on increment:  |du| = {0:.8e}'.format(delta_c_norm.ravel()[0]))
                    self.c_old = c
                    return

            # print solver messages
            #
            self.dummy_solver.print_solver_state(n, residual_norm.ravel(), delta_c_norm.ravel())
            n = n + 1

        assert False

        # return c

    @staticmethod
    def enforce_bcs_on_u(i, input):
        u_temp, bcs_nodes, bcs_values = input
        bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
        u_temp = jax.ops.index_update(u_temp, jax.ops.index[bc_nodes], bc_values)
        return u_temp, bcs_nodes, bcs_values

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
        tangent_temp = jax.ops.index_update(tangent_temp, jax.ops.index[bc_nodes, bc_nodes], 1.0e6)
        return tangent_temp, bcs_nodes, bcs_values

    # residual
    #
    def calculate_element_level_residual(self, nodal_fields):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, c_nodal, c_nodal_old = nodal_fields
        R_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0]), dtype=jnp.float64)

        def quadrature_calculation(q, R_element):
            """
            :param q: the index of the current quadrature point
            :param R_element: the element level residual
            :return: the element level residual for this quadrature point only
            """
            N_xi = self.element_objects[0].N_xi[q, :, :]
            grad_N_X = self.element_objects[0].map_shape_function_gradients(coords)[q, :, :]
            JxW = self.element_objects[0].calculate_JxW(coords)[q, 0]

            c_q = jnp.matmul(c_nodal, N_xi)
            c_q_old = jnp.matmul(c_nodal_old, N_xi)
            c_dot_q = (c_q - c_q_old) / self.time_control.time_increment
            grad_c_q = jnp.matmul(grad_N_X.T, c_nodal).reshape((-1, 1))

            # q_q = self.constitutive_models[0][0].calculate_species_flux(grad_c_q)
            D = self.constitutive_models[0][0].D
            R_q = JxW * c_dot_q * N_xi + \
                  JxW * jnp.matmul(grad_N_X, D * grad_c_q)
            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def assemble_residual(self, u):

        # set up residual and grab connectivity for convenience
        #
        residual = jnp.zeros_like(u)
        connectivity = self.genesis_mesh.connectivity

        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        c_element_wise = u[connectivity]
        c_element_wise_old = self.c_old[connectivity]

        # jit the element level residual calculator
        #
        jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            R_e = jit_calculate_element_level_residual((coordinates[e],
                                                        c_element_wise[e], c_element_wise_old[e]))
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        # adjust residual to satisfy dirichlet conditions
        #
        residual, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
                                        self.enforce_bcs_on_residual, (residual, self.dirichlet_bcs_nodes))

        return residual

    # mass matrix i.e. capacitance matrix
    #
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

    # stiffness matrix, i.e. conductivity matrix or diffusivity matrix
    #
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
        stiffness_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node,
                                      self.genesis_mesh.nodal_coordinates.shape[0] * self.n_dof_per_node),
                                     dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        jit_calculate_element_level_stiffness_matrix = jit(self.calculate_element_level_mass_matrix)

        def element_calculation(e, input):
            stiffness_matrix_temp = input
            K_e = jit_calculate_element_level_stiffness_matrix(coordinates[e])
            indices = jnp.ix_(connectivity[e], connectivity[e])
            stiffness_matrix_temp = jax.ops.index_add(stiffness_matrix_temp, jax.ops.index[indices], K_e)
            return stiffness_matrix_temp

        stiffness_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                             stiffness_matrix)

        return stiffness_matrix

    def post_process_2d(self):
        self.post_processor.exo.put_time(self.time_control.time_step_number,
                                         self.time_control.t)
        self.post_processor.write_nodal_scalar_variable('c', self.time_control.time_step_number, jnp.asarray(self.c_old))

