import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from jax import random
from elements import QuadElement
from time_control import TimeControl
from physics import Physics


class CahnHilliard(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(CahnHilliard, self).__init__(n_dimensions, physics_input)
        print(self)
        self.n_dof_per_node = 2
        self.M_prop = 1.0
        self.lambda_prop = 0.01
        print('Number of degrees of freedom per node = %s' % self.n_dof_per_node)

        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):
            self.element_objects.append(
                QuadElement(quadrature_order=block['cell_interpolation']['quadrature_order'],
                            shape_function_order=block['cell_interpolation']['shape_function_order']))
            print('Block number = %s' % str(n + 1))# initial conditions

        # make a nodelist in case ya need it
        #
        node_list = jnp.arange(0, self.genesis_mesh.nodal_coordinates.shape[0])

        self.time_control_block = self.physics_input['time_control']
        self.time_control = TimeControl(self.time_control_block)

        # jit everything
        #
        # self.jit_dfdc = jit(jacfwd(self.f))
        self.jit_calculate_element_level_mass_matrix = jit(self.calculate_element_level_mass_matrix)
        self.jit_calculate_element_level_stiffness_matrix = jit(self.calculate_element_level_stiffness_matrix)
        self.jit_calculate_element_level_right_hand_side = jit(self.calculate_element_level_right_hand_side)
        self.jit_assemble_mass_matrix = jit(self.assemble_mass_matrix)
        self.jit_assemble_stiffness_matrix = jit(self.assemble_stiffness_matrix)
        self.jit_assemble_right_hand_side = jit(self.assemble_right_hand_side)

        # set initial conditions, mu = 0 at t = 0 and c = some random shit at t = 0
        #
        # self.c_old = 0.63 + 0.02 * (0.5 - random.uniform(random.PRNGKey(128), shape=(node_list.shape[0], ), minval=0,
        #                                                  maxval=1))
        self.c_old = random.uniform(random.PRNGKey(1028), shape=(node_list.shape[0], ), minval=0, maxval=1)
        self.mu_old = jnp.zeros(len(node_list), dtype=jnp.float64)

        # calculate mass matrix up front
        #
        self.M = self.assemble_mass_matrix()
        self.M_lumped = jnp.sum(self.M, axis=1)
        self.max_eigenvalue = jnp.max(self.M_lumped)

        # finally solve
        #
        self.solve()

    def f(self, c):
        return 100.0 * c**2 * (1.0 - c)**2

    def dfdc(self, c):
        # return 100.0 * (2.0 * c * (1.0 - c)**2 - 2.0 * c**2 * (1.0 - c))
        return c**3 - c

    def solve(self):
        self.post_process(1, 0.0, self.c_old, self.mu_old)
        self.time_control.increment_time()
        c_old = self.c_old
        mu_old = self.mu_old
        print('{0:8}\t\t{1:8}\t\t{2:8}\t\t{3:8}'.
              format('Increment', 'Time', '|dc|', '|dmu|'))
        while self.time_control.t <= self.time_control.time_end:
            K_c = self.jit_assemble_stiffness_matrix(mu_old, self.M_prop)
            K_mu = self.jit_assemble_stiffness_matrix(c_old, self.lambda_prop)
            R_mu = self.jit_assemble_right_hand_side(c_old)

            c = c_old - self.time_control.time_increment * jnp.matmul(K_c, mu_old) / self.M_lumped
            mu = (R_mu - jnp.matmul(K_mu, c_old)) / self.M_lumped

            if self.time_control.time_step_number % 100 == 0:
                print('{0:8}\t\t{1:8}\t\t{2:8}\t\t{3:8}'.
                      format('Increment', 'Time', '|dc|', '|dmu|'))
            if self.time_control.time_step_number % 10 == 0:
                delta_c_error = jnp.linalg.norm(c - c_old)
                delta_mu_error = jnp.linalg.norm(mu - mu_old)
                print('{0:8}\t\t{1:.8e}\t\t{2:.8e}\t\t{3:.8e}'.
                      format(self.time_control.time_step_number, self.time_control.t,
                             delta_c_error.ravel()[0],
                             delta_mu_error.ravel()[0]))

            if self.time_control.time_step_number % 50 == 0:
                self.post_process(self.time_control.time_step_number, self.time_control.t, c, mu)

            c_old = c
            mu_old = mu
            self.time_control.increment_time()

    def assemble_mass_matrix(self):
        mass_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0],
                                 self.genesis_mesh.nodal_coordinates.shape[0]),
                                dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        def element_calculation(e, input):
            mass_matrix_temp = input
            M_e = self.jit_calculate_element_level_mass_matrix(coordinates[e])
            indices = jnp.ix_(connectivity[e], connectivity[e])
            mass_matrix_temp = jax.ops.index_add(mass_matrix_temp, jax.ops.index[indices], M_e)
            return mass_matrix_temp

        mass_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                        mass_matrix)

        return mass_matrix

    def assemble_stiffness_matrix(self, u_old, k_prop):
        stiffness_matrix = jnp.zeros((self.genesis_mesh.nodal_coordinates.shape[0],
                                      self.genesis_mesh.nodal_coordinates.shape[0]),
                                     dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        def element_calculation(e, input):
            stiffness_matrix_temp = input
            K_e = self.jit_calculate_element_level_stiffness_matrix((coordinates[e], u_old, k_prop))
            indices = jnp.ix_(connectivity[e], connectivity[e])
            stiffness_matrix_temp = jax.ops.index_add(stiffness_matrix_temp, jax.ops.index[indices], K_e)
            return stiffness_matrix_temp

        stiffness_matrix = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                             stiffness_matrix)

        return stiffness_matrix

    def assemble_right_hand_side(self, u_old):
        # set up residual and grab connectivity for convenience
        #
        connectivity = self.genesis_mesh.connectivity
        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]

        R = jnp.zeros_like(u_old)

        u_element_wise_old = u_old[connectivity]

        def element_calculation(e, input):
            rhs_temp = input
            R_e = self.jit_calculate_element_level_right_hand_side((coordinates[e], u_element_wise_old[e]))
            rhs_temp = jax.ops.index_add(rhs_temp, jax.ops.index[connectivity[e]], R_e)
            return rhs_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, R)

        return residual

    def calculate_element_level_mass_matrix(self, coords):
        M_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0],
                         self.genesis_mesh.n_nodes_per_element[0]),
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

    def calculate_element_level_stiffness_matrix(self, inputs):
        coords, u_old, k_prop = inputs
        K_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0],
                         self.genesis_mesh.n_nodes_per_element[0]),
                        dtype=jnp.float64)
        grad_N_X_element = self.element_objects[0].map_shape_function_gradients(coords)
        JxW_element = self.element_objects[0].calculate_JxW(coords)

        def quadrature_calculation(q, K_element):
            grad_N_X = grad_N_X_element[q, :, :]
            JxW = JxW_element[q, 0]
            K_q = JxW * k_prop * jnp.matmul(grad_N_X, grad_N_X.T)
            K_element = jax.ops.index_add(K_element, jax.ops.index[:, :], K_q)
            return K_element

        K_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, K_e)

        return K_e

    def calculate_element_level_right_hand_side(self, inputs):
        coords, u_nodal_old = inputs
        R_e = jnp.zeros(self.genesis_mesh.n_nodes_per_element[0], dtype=jnp.float64)
        JxW_element = self.element_objects[0].calculate_JxW(coords)

        def quadrature_calculation(q, R_element):
            N_xi = self.element_objects[0].N_xi[q, :, :]
            JxW = JxW_element[q, 0]

            # u_q_old = N_xi * u_nodal_old
            u_q_old = jnp.matmul(u_nodal_old, N_xi)
            # dfdc_q = self.jit_dfdc(u_q_old)
            dfdc_q = self.dfdc(u_q_old)
            R_q = JxW * dfdc_q * N_xi

            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def post_process(self, time_step, time, c, mu):
        self.post_processor.exo.put_time(time_step, time)
        self.post_processor.write_nodal_scalar_variable('c', time_step, jnp.asarray(c))
        self.post_processor.write_nodal_scalar_variable('mu', time_step, jnp.asarray(mu))