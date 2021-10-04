import jax
import jax.numpy as jnp
from jax import jit
from jax import jacfwd
from .application import Application
from time_control import TimeControl
from elements import LineElement, QuadElement
from physics import InitialCondition
from physics import Physics
from physics import RadiativeTransfer
from physics.constitutive_models import FicksLaw
from physics import ExplicitSpeciesTransport
from physics import ImplicitSpeciesTransport


class PhotoChemistry(Application):
    """
    Application to do a staggered solve on a photochemistry equations

    Light is solved for implicitly every n_I steps to update the light field
    for potential changes in light absorption

    Species transport will be solved for explicitly
    """
    def __init__(self, n_dimension, physics_input_blocks):
        super(PhotoChemistry, self).__init__(n_dimension, physics_input_blocks)
        self.n_species = 1

        # unpack input deck block
        #
        self.time_control_block = self.physics_input_blocks['time_control']
        self.radiative_transfer_block = self.physics_input_blocks['radiative_transfer']
        self.chemistry_block = self.physics_input_blocks['chemistry']

        # make objects which build the app physics and time control
        #
        self.time_control = TimeControl(self.time_control_block)
        self.radiative_transfer = RadiativeTransfer(self.n_dimensions, self.radiative_transfer_block)
        self.chemistry = Physics(self.n_dimensions, self.chemistry_block)

        # initialize the element type
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.chemistry.blocks_input_block):

            # constitutive model
            #
            block_constitutive_models = []
            for s in range(self.n_species):
                block_constitutive_models.append(
                    FicksLaw(self.chemistry.blocks_input_block[n]['constitutive_model'][s]))

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

        # initial conditions
        #
        node_list = jnp.arange(0, self.chemistry.genesis_mesh.nodal_coordinates.shape[0])
        self.C_I_ic = InitialCondition(ic_type=self.chemistry.initial_conditions_input_block['C_I']['type'],
                                       block_nodes=node_list,
                                       value=self.chemistry.initial_conditions_input_block['C_I']['value'])
        self.C_R_ic = InitialCondition(ic_type=self.chemistry.initial_conditions_input_block['C_R']['type'],
                                       block_nodes=node_list,
                                       value=self.chemistry.initial_conditions_input_block['C_R']['value'])
        self.C_M_ic = InitialCondition(ic_type=self.chemistry.initial_conditions_input_block['C_M']['type'],
                                       block_nodes=node_list,
                                       value=self.chemistry.initial_conditions_input_block['C_M']['value'])

        self.C_I_old = self.C_I_ic.values
        self.C_R_old = self.C_R_ic.values
        self.C_M_old = self.C_M_ic.values

        # chemistry properties
        #
        self.properties = self.physics_input_blocks['chemistry']['properties']

        # photochemistry stuff
        #
        self.jit_calculate_element_level_mass_matrix = jit(self.calculate_element_level_mass_matrix)
        self.jit_calculate_element_level_right_hand_side = jit(self.calculate_element_level_right_hand_side)

        self.jit_assemble_mass_matrix = jit(self.assemble_mass_matrix)
        self.jit_assemble_right_hand_side = jit(self.assemble_right_hand_side)

        # pre-calculate mass matrix since it will never change for this problem
        #
        self.M = self.jit_assemble_mass_matrix()
        self.M_lumped = jnp.sum(self.M, axis=1)
        self.max_eigenvalue = jnp.max(self.M_lumped)

        # TODO set up an auto time step calculator
        #
        # self.time_control.time_increment = self.species_transport.solver.max_eigenvalue / \
        #                                    (4.0 * self.species_transport.constitutive_models[0][0].D)

        # run the simulation
        #
        self.solve()

    def __str__(self):
        print('Photochemistry app')
        print('Number of dimensions = %s' % self.n_dimensions)

    def solve(self):

        # write initial time first
        #
        self.radiative_transfer.solve(self.time_control.time_step_number,
                                      self.time_control.t)
        self.post_process(1, 0.0, self.radiative_transfer.I, self.C_I_old, self.C_R_old, self.C_M_old)
        self.time_control.increment_time()
        I_old = self.radiative_transfer.I
        C_I_old = self.C_I_old
        C_R_old = self.C_R_old
        C_M_old = self.C_M_old
        print('{0:8}\t\t{1:8}\t\t{2:8}\t\t{3:8}\t\t{4:8}\t\t{5:8}'.
              format('Increment', 'Time', '|dC_I|', '|dC_R|', '|dC_M|', '|dp|'))
        while self.time_control.t < self.time_control.time_end:
            # print('Time = %s' % self.time_control.t)
            R_C_I, R_C_R, R_C_M = self.jit_assemble_right_hand_side(I_old, C_I_old, C_R_old, C_M_old)
            C_I = C_I_old + self.time_control.time_increment * R_C_I / self.M_lumped
            C_R = C_R_old + self.time_control.time_increment * R_C_R / self.M_lumped
            C_M = C_M_old + self.time_control.time_increment * R_C_M / self.M_lumped

            # error logging and post-processing
            #
            if self.time_control.time_step_number % 3000 == 0:
                print('{0:8}\t\t{1:8}\t\t{2:8}\t\t{3:8}\t\t{4:8}'.
                      format('Increment', 'Time', '|dC_I|', '|dC_R|', '|dC_M|'))
            if self.time_control.time_step_number % 100 == 0:
                p = 1.0 - C_M / self.C_M_ic.value
                p_old = 1.0 - C_M_old / self.C_M_ic.value
                delta_C_I_error = jnp.linalg.norm(C_I - C_I_old)
                delta_C_R_error = jnp.linalg.norm(C_R - C_R_old)
                delta_C_M_error = jnp.linalg.norm(C_M - C_M_old)
                delta_p_error = jnp.linalg.norm(p - p_old)
                print('{0:8}\t\t{1:.8e}\t\t{2:.8e}\t\t{3:.8e}\t\t{4:.8e}\t\t{5:.8e}'.
                      format(self.time_control.time_step_number, self.time_control.t,
                             delta_C_I_error.ravel()[0],
                             delta_C_R_error.ravel()[0],
                             delta_C_M_error.ravel()[0],
                             delta_p_error.ravel()[0]))

            if self.time_control.time_step_number % 50000 == 0:
                self.post_process(self.time_control.time_step_number, self.time_control.t,
                                  I_old, C_I_old, C_R_old, C_M_old)

            C_I_old = jax.ops.index_update(C_I_old, jax.ops.index[:], C_I)
            C_R_old = jax.ops.index_update(C_R_old, jax.ops.index[:], C_R)
            C_M_old = jax.ops.index_update(C_M_old, jax.ops.index[:], C_M)
            self.time_control.increment_time()
            # assert False

    def calculate_element_level_mass_matrix(self, coords):
        M_e = jnp.zeros((self.chemistry.genesis_mesh.n_nodes_per_element[0],
                         self.chemistry.genesis_mesh.n_nodes_per_element[0]),
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

    def calculate_element_level_stiffness_matrix(self):
        pass

    def calculate_element_level_right_hand_side(self, input):
        # extract the input
        #
        coords, I_nodal_old, C_I_nodal_old, C_R_nodal_old, C_M_nodal_old = input

        # initialize element level residuals
        #
        R_C_I_e = jnp.zeros(self.chemistry.genesis_mesh.n_nodes_per_element[0], dtype=jnp.float64)
        R_C_R_e = jnp.zeros(self.chemistry.genesis_mesh.n_nodes_per_element[0], dtype=jnp.float64)
        R_C_M_e = jnp.zeros(self.chemistry.genesis_mesh.n_nodes_per_element[0], dtype=jnp.float64)

        # initial condition
        #
        C_M_0 = self.C_M_ic.value

        # calculate shape function stuff
        #
        JxW_element = self.element_objects[0].calculate_JxW(coords)

        # get properties TODO eventually update these
        #
        beta = self.properties['beta']
        m = self.properties['m']
        #
        k_p_0 = self.properties['k_p_0']
        k_p_D_0 = self.properties['k_p_D_0']
        c = self.properties['c']
        #
        k_t_SD = self.properties['k_t_SD']
        k_t_TD_0 = self.properties['k_t_TD_0']
        C_RD = self.properties['C_RD']


        # calculate at the quadrature point
        #
        def quadrature_calculation(q, R_element):
            R_C_I_element, R_C_R_element, R_C_M_element = R_element

            N_xi = self.element_objects[0].N_xi[q, :, :]
            JxW = JxW_element[q, 0]

            # get fields at quadrature points
            #
            I_q_old = jnp.matmul(I_nodal_old, N_xi)
            C_I_q_old = jnp.matmul(C_I_nodal_old, N_xi)
            C_R_q_old = jnp.matmul(C_R_nodal_old, N_xi)
            C_M_q_old = jnp.matmul(C_M_nodal_old, N_xi)
            p_q_old = (1.0 - C_M_q_old / C_M_0)

            # update material properties from old step
            #
            k_p = k_p_0 / (1.0 + (k_p_0 / k_p_D_0) * jnp.exp(c * p_q_old))
            k_term = 1.0 / ((1.0 / k_t_SD) + (jnp.exp(c * p_q_old) / k_t_TD_0)) + \
                     (C_RD * (1.0 - p_q_old) * k_p)

            R_C_I_q = -JxW * beta * I_q_old * C_I_q_old * N_xi
            R_C_R_q = JxW * m * beta * I_q_old * C_I_q_old * N_xi - 2.0 * k_term * C_R_q_old * C_R_q_old * N_xi
            R_C_M_q = -JxW * k_p * C_R_q_old * C_M_q_old * N_xi

            R_C_I_element = jax.ops.index_add(R_C_I_element, jax.ops.index[:], R_C_I_q[:, 0])
            R_C_R_element = jax.ops.index_add(R_C_R_element, jax.ops.index[:], R_C_R_q[:, 0])
            R_C_M_element = jax.ops.index_add(R_C_M_element, jax.ops.index[:], R_C_M_q[:, 0])

            return R_C_I_element, R_C_R_element, R_C_M_element

        R_C_I_e, R_C_R_e, R_C_M_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points,
                                                      quadrature_calculation,
                                                      (R_C_I_e, R_C_R_e, R_C_M_e))

        return R_C_I_e, R_C_R_e, R_C_M_e

    def assemble_mass_matrix(self):
        mass_matrix = jnp.zeros((self.chemistry.genesis_mesh.nodal_coordinates.shape[0] * self.n_species,
                                 self.chemistry.genesis_mesh.nodal_coordinates.shape[0] * self.n_species),
                                dtype=jnp.float64)
        connectivity = self.chemistry.genesis_mesh.connectivity
        coordinates = self.chemistry.genesis_mesh.nodal_coordinates[connectivity]

        # jit the element level mass matrix calculator
        #
        def element_calculation(e, input):
            mass_matrix_temp = input
            M_e = self.jit_calculate_element_level_mass_matrix(coordinates[e])
            indices = jnp.ix_(connectivity[e], connectivity[e])
            mass_matrix_temp = jax.ops.index_add(mass_matrix_temp, jax.ops.index[indices], M_e)
            return mass_matrix_temp

        mass_matrix = jax.lax.fori_loop(0, self.chemistry.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                        mass_matrix)

        return mass_matrix

    def assemble_right_hand_side(self, I_old, C_I_old, C_R_old, C_M_old):
        # set up residual and grab connectivity for convenience
        #
        connectivity = self.chemistry.genesis_mesh.connectivity
        coordinates = self.chemistry.genesis_mesh.nodal_coordinates[connectivity]

        R_C_I = jnp.zeros_like(C_I_old)
        R_C_R = jnp.zeros_like(C_R_old)
        R_C_M = jnp.zeros_like(C_M_old)

        I_element_wise_old = I_old[connectivity]
        C_I_element_wise_old = C_I_old[connectivity]
        C_R_element_wise_old = C_R_old[connectivity]
        C_M_element_wise_old = C_M_old[connectivity]

        def element_calculation(e, input):
            rhs_C_I_temp, rhs_C_R_temp, rhs_C_M_temp = input
            R_C_I_e, R_C_R_e, R_C_M_e = self.jit_calculate_element_level_right_hand_side((coordinates[e],
                                                                                          I_element_wise_old[e],
                                                                                          C_I_element_wise_old[e],
                                                                                          C_R_element_wise_old[e],
                                                                                          C_M_element_wise_old[e]))
            rhs_C_I_temp = jax.ops.index_add(rhs_C_I_temp, jax.ops.index[connectivity[e]], R_C_I_e)
            rhs_C_R_temp = jax.ops.index_add(rhs_C_R_temp, jax.ops.index[connectivity[e]], R_C_R_e)
            rhs_C_M_temp = jax.ops.index_add(rhs_C_M_temp, jax.ops.index[connectivity[e]], R_C_M_e)
            return rhs_C_I_temp, rhs_C_R_temp, rhs_C_M_temp

        residual = jax.lax.fori_loop(0, self.chemistry.genesis_mesh.n_elements_in_blocks[0], element_calculation,
                                     (R_C_I, R_C_R, R_C_M))

        return residual

    def post_process(self, time_step, time, I, C_I, C_R, C_M):
        # write time
        #
        self.radiative_transfer.post_processor.exo.put_time(time_step, time)
        self.chemistry.post_processor.exo.put_time(time_step, time)

        # write nodal variables
        #
        self.radiative_transfer.post_processor.write_nodal_scalar_variable('I', time_step, jnp.asarray(I))
        self.chemistry.post_processor.write_nodal_scalar_variable('I', time_step, jnp.asarray(I))
        self.chemistry.post_processor.write_nodal_scalar_variable('C_I', time_step, jnp.asarray(C_I))
        self.chemistry.post_processor.write_nodal_scalar_variable('C_R', time_step, jnp.asarray(C_R))
        self.chemistry.post_processor.write_nodal_scalar_variable('C_M', time_step, jnp.asarray(C_M))
        self.chemistry.post_processor.write_nodal_scalar_variable('p', time_step,
                                                                  jnp.asarray(1.0 - C_M / self.C_M_ic.value))
