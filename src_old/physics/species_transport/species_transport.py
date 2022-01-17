import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, jacfwd
from elements import LineElement
from elements import QuadElement
from solvers import NewtonRaphsonSolver
from ..physics import Physics
from ..initial_conditions import InitialCondition
from ..boundary_conditions import DirichletBoundaryCondition
from ..constitutive_models import FicksLaw
from time_control import TimeControl


class SpeciesTransport(Physics):
    def __init__(self, n_dimensions, mesh_input):
        super(SpeciesTransport, self).__init__(n_dimensions, mesh_input)
        print(self)

        # set number of degress of freedom per node
        #
        self.n_dof_per_node = self.physics_input['number_of_species']
        print('Number of species requested = {}'.format(self.n_dof_per_node))

        # need to modify the connectivity matrix for the number of dofs, keep around other connectivity matrix
        # for easily calculating element coordinates, we need to figure out how to do this with one connectivity matrix
        # efficiently
        #
        self.connectivity = jnp.zeros((self.genesis_mesh.connectivity.shape[0],
                                       self.genesis_mesh.connectivity.shape[1] * self.n_dof_per_node),
                                      dtype=jnp.int32)

        for e in range(self.genesis_mesh.connectivity.shape[0]):
            self.connectivity = jax.ops.index_update(self.connectivity, jax.ops.index[e, 0::self.n_dof_per_node],
                                                     self.n_dof_per_node * self.genesis_mesh.connectivity[e])
            for m in range(1, self.n_dof_per_node):
                self.connectivity = jax.ops.index_update(self.connectivity, jax.ops.index[e, m::self.n_dof_per_node],
                                                         self.connectivity[e, 0::self.n_dof_per_node] + m)
        #
        assert self.connectivity.shape == (self.genesis_mesh.connectivity.shape[0],
                                           self.n_dof_per_node * self.genesis_mesh.connectivity.shape[1])

        # set initial conditions
        #

        # get some boundary conditions
        #
        self.dirichlet_bcs = []
        self.dirichlet_bcs_nodes = []
        self.dirichlet_bcs_values = []
        self.dirichlet_bcs_species_association = []

        print(self.boundary_conditions_input_block)
        # import sys
        # sys.exit()
        for key in self.boundary_conditions_input_block.keys():
            bc_type = self.boundary_conditions_input_block[key]
            for i, bc in enumerate(bc_type):
                self.dirichlet_bcs.append(
                    DirichletBoundaryCondition(dirichlet_bc_input_block=bc,
                                               node_set_name=bc['node_set'],
                                               node_set_nodes=self.genesis_mesh.node_set_nodes[i]))
                # self.dirichlet_bcs_nodes.append(self.dirichlet_bcs[i].node_set_nodes)

                self.dirichlet_bcs_nodes.append(self.n_dof_per_node * self.dirichlet_bcs[i].node_set_nodes +
                                                bc['species'] - 1)
                self.dirichlet_bcs_values.append(self.dirichlet_bcs[i].values)
                self.dirichlet_bcs_species_association.append(bc['species'])

        self.dirichlet_bcs_nodes = jnp.array(self.dirichlet_bcs_nodes)
        self.dirichlet_bcs_values = jnp.array(self.dirichlet_bcs_values)

        # TODO add Neumann BC support
        #

        # initialize the element type
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):
            block_constitutive_models = []
            for m in range(self.n_dof_per_node):
                block_constitutive_models.append(FicksLaw(self.blocks_input_block[n]['constitutive_model'][m]))
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
            for m in range(self.n_dof_per_node):
                print('\tSpecies = %s' % m)
                print(self.constitutive_models[n][m])

        # set up time control
        #
        self.time_control = TimeControl(self.physics_input['time_control'])

        # jit a bunch of methods
        #
        self.jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)
        self.jit_assemble_residual = jit(self.assemble_residual)
        # self.jit_calculate_element_level_residual = self.calculate_element_level_residual
        # self.jit_assemble_residual = self.assemble_residual
        self.jit_assemble_tangent = jit(jacfwd(self.assemble_residual))
        self.jit_linear_solver = jit(jax.scipy.sparse.linalg.gmres)
        self.jit_enforce_bcs_on_u = jit(self.enforce_bcs_on_u)
        self.jit_enforce_bcs_on_residual = jit(self.enforce_bcs_on_residual)
        self.jit_enforce_bcs_on_tangent = jit(self.enforce_bcs_on_tangent)

        # set up the newton solver
        #
        self.solver = NewtonRaphsonSolver(self.solver_input_block,
                                          len(self.genesis_mesh.nodal_coordinates),
                                          self.n_dof_per_node,
                                          self.connectivity,
                                          self.jit_assemble_residual,
                                          self.jit_assemble_tangent,
                                          self.jit_enforce_bcs_on_u,
                                          self.jit_enforce_bcs_on_residual,
                                          self.jit_enforce_bcs_on_tangent)

        self.solve()

    def solve(self):
        # make initial condition zero for now and write to exodus
        #
        print(self.time_control)
        u_old = jnp.zeros(len(self.genesis_mesh.nodal_coordinates[:, 0]) * self.n_dof_per_node, dtype=jnp.float64)
        self.post_process_2d(self.time_control.time_step_number, self.time_control.t, u_old)
        self.time_control.increment_time()

        print('here')

        # begin loop over time
        #
        while self.time_control.t <= self.time_control.time_end:

            # update dirichlet bcs objects
            #
            for n in range(len(self.dirichlet_bcs)):
                temp_values = self.dirichlet_bcs[n].update_bc_values(time=self.time_control.t)
                self.dirichlet_bcs_values = jax.ops.index_update(self.dirichlet_bcs_values, jax.ops.index[n, :],
                                                                 temp_values)

            # call the newton solver
            #
            u = self.solver.solve(u_old,
                                  len(self.dirichlet_bcs), self.dirichlet_bcs_nodes, self.dirichlet_bcs_values,
                                  self.time_control.time_step_number, self.time_control.t,
                                  self.time_control.time_increment)

            # now post-process
            #
            self.post_process_2d(self.time_control.time_step_number, self.time_control.t, u)

            u_dot = (u - u_old) / self.time_control.time_increment
            u_dot_norm = jnp.linalg.norm(self.time_control.time_increment * u_dot)
            print('|u_n+1 - u_n| = {0:.6e}'.format(u_dot_norm.ravel()[0]))

            # now update u_old
            #
            u_old = jax.ops.index_update(u_old, jax.ops.index[:], u)

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

    def calculate_element_level_residual(self, inputs):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, c_nodal, c_nodal_old, t, delta_t = inputs
        R_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0] * self.n_dof_per_node), dtype=jnp.float64)

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

            for m in range(self.n_dof_per_node):
                c_nodal_m = c_nodal[::self.n_dof_per_node]
                c_nodal_old_m = c_nodal_old[::self.n_dof_per_node]
                c_q = jnp.matmul(c_nodal_m, N_xi)
                c_q_old = jnp.matmul(c_nodal_old_m, N_xi)
                grad_c_q = jnp.matmul(grad_N_X.T, c_nodal_m).reshape((-1, 1))
                j_q = self.constitutive_models[0][m].calculate_species_flux(grad_c_q)
                source = 0.0
                R_q = JxW * (c_q * N_xi - c_q_old * N_xi -
                             delta_t * jnp.matmul(grad_N_X, j_q) - source * N_xi)
                # print(R_element)
                R_element = jax.ops.index_add(R_element, jax.ops.index[m::self.n_dof_per_node], R_q[:, 0])

            # c_q = jnp.matmul(c_nodal, N_xi)
            # c_q_old = jnp.matmul(c_nodal_old, N_xi)
            # grad_c_q = jnp.matmul(grad_N_X.T, c_nodal).reshape((-1, 1))

            # q_q = self.constitutive_models[0].calculate_heat_conduction(grad_c_q)

            # source = jnp.sin(2.0 * jnp.pi * t / 1000.0) * \
            #          jnp.exp(-((x_q - 0.5) ** 2 + (y_q - 0.5) ** 2) / 25.0) + 1.0
            # source = 0.0
            #
            # R_q = JxW * (c_q * N_xi - c_q_old * N_xi -
            #              delta_t * jnp.matmul(grad_N_X, q_q) - source * N_xi)
            # R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    # def assemble_residual(self, u):
    def assemble_residual(self, u, u_old, t, delta_t):
        # set up residual and grab connectivity for convenience
        #
        # u, u_old, t, delta_t = inputs
        residual = jnp.zeros_like(u)
        # connectivity = self.genesis_mesh.connectivity
        connectivity = self.connectivity
        # print(connectivity)

        # coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        coordinates = self.genesis_mesh.nodal_coordinates[self.genesis_mesh.connectivity]
        c_element_wise = u[connectivity]
        c_element_wise_old = u_old[connectivity]

        # jit the element level residual calculator
        #
        # TODO move this up to a top level operation
        # jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            # print(connectivity[e])
            R_e = self.jit_calculate_element_level_residual((coordinates[e],
                                                            c_element_wise[e], c_element_wise_old[e],
                                                            t, delta_t))
            # print(R_e)
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        # print(self.genesis_mesh.n_elements_in_blocks[0])
        # print(type(self.connectivity))
        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        return residual

    def post_process_1d(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.plot(self.genesis_mesh.nodal_coordinates, -0.5 * self.genesis_mesh.nodal_coordinates ** 2 +
                 0.5 * self.genesis_mesh.nodal_coordinates,
                 linestyle='None', marker='o')
        plt.show()

    def post_process_2d(self, time_step, time, c):
        self.post_processor.exo.put_time(time_step, time)
        for m in range(self.n_dof_per_node):
            self.post_processor.write_nodal_scalar_variable('c_' + str(m + 1), time_step,
                                                            jnp.asarray(c[m::self.n_dof_per_node]))
