import jax
import jax.numpy as jnp
from jax import jit, jacfwd
from elements import LineElement
from elements import QuadElement
from solvers import Solver
from solvers import NewtonRaphsonSolver
# from solvers import NewtonRaphsonTransientSolver
from ..physics import Physics
from ..initial_conditions import InitialCondition
from ..boundary_conditions import DirichletBoundaryCondition
from ..boundary_conditions import NeumannBoundaryCondition
from ..constitutive_models import FouriersLaw
from time_control import TimeControl


class TransientHeatConduction(Physics):
    def __init__(self, n_dimensions, mesh_input):
        super(TransientHeatConduction, self).__init__(n_dimensions, mesh_input)
        print(self)

        # set number of degress of freedom per node
        #
        self.n_dof_per_node = 1

        # set initial conditions
        #

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

        # TODO add Neumann BC support
        #

        # initialize the element type
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):

            # constitutive model
            #
            self.constitutive_models.append(
                FouriersLaw(self.blocks_input_block[n]['constitutive_model']))

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
            print(self.constitutive_models[n])

        # set up a solver
        #
        # self.solver = NewtonRaphsonSolver(self.solver_input_block,
        #                                   len(self.genesis_mesh.nodal_coordinates),
        #                                   1,
        #                                   self.genesis_mesh.connectivity,
        #                                   self.enforce_bcs_on_u,
        #                                   self.enforce_bcs_on_residual,
        #                                   self.enforce_bcs_on_tangent,
        #                                   self.assemble_residual)
        self.dummy_solver = Solver()
        # set up time control
        #
        self.time_control = TimeControl(self.physics_input['time_control'])

        # solve here for test, otherwise call an app for this
        #
        self.jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)
        self.jit_assemble_residual = jit(self.assemble_residual)
        self.jit_assemble_tangent = jit(jacfwd(self.assemble_residual))
        self.jit_linear_solver = jit(jax.scipy.sparse.linalg.gmres)

        # set up the newton solver
        #
        self.solver = NewtonRaphsonSolver(self.solver_input_block,
                                          len(self.genesis_mesh.nodal_coordinates),
                                          1,
                                          self.genesis_mesh.connectivity,
                                          self.jit_assemble_residual,
                                          self.jit_assemble_tangent,
                                          self.enforce_bcs_on_u,
                                          self.enforce_bcs_on_residual,
                                          self.enforce_bcs_on_tangent)

        # self.solve()
        self.solve_native()

    def solve(self):
        # make initial condition zero for now and write to exodus
        #
        print(self.time_control)
        u_old = jnp.zeros(len(self.genesis_mesh.nodal_coordinates[:, 0] * self.n_dof_per_node), dtype=jnp.float64)
        self.post_process_2d(self.time_control.time_step_number, self.time_control.t, u_old)
        self.time_control.increment_time()

        # begin loop over time
        #
        while self.time_control.t <= self.time_control.time_end:
            self.dummy_solver.print_solver_heading(self.time_control.time_step_number)

            self.time_control.increment_time()

    def solve_native(self):

        # make initial condition zero for now and write to exodus
        #
        print(self.time_control)
        u_old = jnp.zeros(len(self.genesis_mesh.nodal_coordinates[:, 0] * self.n_dof_per_node), dtype=jnp.float64)
        self.post_process_2d(self.time_control.time_step_number, self.time_control.t, u_old)
        self.time_control.increment_time()

        # begin loop over time
        #
        while self.time_control.t <= self.time_control.time_end:
            self.dummy_solver.print_solver_heading(self.time_control.time_step_number)
            # update dirichlet bcs objects
            #
            for n in range(len(self.dirichlet_bcs)):
                temp_values = self.dirichlet_bcs[n].update_bc_values(time=self.time_control.t)
                self.dirichlet_bcs_values = jax.ops.index_update(self.dirichlet_bcs_values, jax.ops.index[n, :],
                                                                 temp_values)

            # make an initial guess for newton iterations
            #
            u = jnp.zeros_like(u_old)

            # begin loop over newton iterations
            #
            n = 0
            while n <= self.solver_input_block['maximum_iterations']:
                # enforce bcs on u
                #
                try:
                    u, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes),
                                                self.enforce_bcs_on_u,
                                                (u, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
                except IndexError:
                    pass

                # assemble residual and tangent
                #
                residual = self.jit_assemble_residual(u, u_old,
                                                      self.time_control.t, self.time_control.time_increment)
                try:
                    residual, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
                                                    self.enforce_bcs_on_residual, (residual, self.dirichlet_bcs_nodes))
                except IndexError:
                    pass

                residual_error = jnp.linalg.norm(residual)

                # check error on residual
                #
                if residual_error < self.solver_input_block['residual_tolerance']:
                    print('Converged on residual: |R| = {0:.8e}'.format(residual_error.ravel()[0]))
                    break

                # if not converged on residual, calculate tangent and newton step
                #
                tangent = self.jit_assemble_tangent(u, u_old, self.time_control.t, self.time_control.time_increment)

                # enforce bcs on residual and tangent
                #
                try:
                    tangent, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs_nodes), self.enforce_bcs_on_tangent,
                                                      (tangent, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))
                except IndexError:
                    pass

                delta_u, _ = self.jit_linear_solver(tangent, -residual)
                u = jax.ops.index_add(u, jax.ops.index[:], delta_u)

                increment_error = jnp.linalg.norm(delta_u)

                if increment_error < self.solver_input_block['increment_tolerance']:
                    print('Converged on increment: |du| = {0:.8e}'.format(increment_error.ravel()[0]))
                    break

                self.dummy_solver.print_solver_state(n, residual_error.ravel(), increment_error.ravel())

                n = n + 1

            # now post-process
            #
            self.post_process_2d(self.time_control.time_step_number, self.time_control.t, u)

            u_dot = (u - u_old) / self.time_control.time_increment
            u_dot_norm = jnp.linalg.norm(self.time_control.time_increment * u_dot)
            print('|u_n+1 - u_n| = {0:.6e}'.format(u_dot_norm.ravel()[0]))

            # now update u_old
            #
            u_old = jax.ops.index_update(u_old, jax.ops.index[:], u)

            # increment time
            #
            self.time_control.increment_time()

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

    # def calculate_element_level_residual(self, nodal_fields):
    def calculate_element_level_residual(self, inputs):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, theta_nodal, theta_nodal_old, t, delta_t = inputs
        # coords, theta_nodal, theta_nodal_old = nodal_fields
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

            x_q = jnp.matmul(coords[:, 0], N_xi)
            y_q = jnp.matmul(coords[:, 1], N_xi)

            theta_q = jnp.matmul(theta_nodal, N_xi)
            theta_q_old = jnp.matmul(theta_nodal_old, N_xi)
            grad_theta_q = jnp.matmul(grad_N_X.T, theta_nodal).reshape((-1, 1))

            q_q = self.constitutive_models[0].calculate_heat_conduction(grad_theta_q)

            # source = jnp.sin(2.0 * jnp.pi * t / 1000.0) * \
            #          jnp.exp(-((x_q - 0.5) ** 2 + (y_q - 0.5) ** 2) / 25.0) + 1.0
            source = 0.0

            R_q = JxW * (theta_q * N_xi - theta_q_old * N_xi -
                         delta_t * jnp.matmul(grad_N_X, q_q) - source * N_xi)
            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    # def assemble_residual(self, u):
    def assemble_residual(self, u, u_old, t, delta_t):
        # set up residual and grab connectivity for convenience
        #
        # u, u_old, t, delta_t = inputs
        residual = jnp.zeros_like(u)
        connectivity = self.genesis_mesh.connectivity

        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        theta_element_wise = u[connectivity]
        theta_element_wise_old = u_old[connectivity]

        # jit the element level residual calculator
        #
        # TODO move this up to a top level operation
        # jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            R_e = self.jit_calculate_element_level_residual((coordinates[e],
                                                            theta_element_wise[e], theta_element_wise_old[e],
                                                            t, delta_t))
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)



        return residual

    def post_process_1d(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.plot(self.genesis_mesh.nodal_coordinates, -0.5 * self.genesis_mesh.nodal_coordinates ** 2 +
                 0.5 * self.genesis_mesh.nodal_coordinates,
                 linestyle='None', marker='o')
        plt.show()

    def post_process_2d(self, time_step, time, theta):
        self.post_processor.exo.put_time(time_step, time)
        self.post_processor.write_nodal_scalar_variable('theta', time_step, jnp.asarray(theta))
