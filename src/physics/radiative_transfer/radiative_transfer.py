import jax
import jax.numpy as jnp
from jax import jit
from elements import LineElement
from elements import QuadElement
from solvers import NewtonRaphsonSolver
# from solvers import NewtonRaphsonTransientSolver
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition


class RadiativeTransfer(Physics):
    def __init__(self, n_dimensions, physics_input_block):
        super(RadiativeTransfer, self).__init__(n_dimensions, physics_input_block)

        # set number of degress of freedom per node
        #
        self.n_dof_per_node = 1

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
                                               coordinates=self.genesis_mesh.nodal_coordinates[
                                                   self.genesis_mesh.node_set_nodes[i]]))
                self.dirichlet_bcs_nodes.append(self.dirichlet_bcs[i].node_set_nodes)
                self.dirichlet_bcs_values.append(self.dirichlet_bcs[i].values)

        self.dirichlet_bcs_nodes = jnp.array(self.dirichlet_bcs_nodes)
        self.dirichlet_bcs_values = jnp.array(self.dirichlet_bcs_values)

        # initialize the element type
        #
        self.element_objects = []
        # self.constitutive_models = []
        self.ds = []
        self.sigmas = []
        for n, block in enumerate(self.blocks_input_block):

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

            # get properties, the 0:2 is for 2d, need to fix this for more general problems
            #
            self.ds.append(jnp.array(block['radiative_transfer_properties']['d'][0:2]).reshape((-1, 1)))
            self.sigmas.append(block['radiative_transfer_properties']['sigma'])

            # print details about the blocks
            #
            print('Block number = %s' % str(n + 1))
            print('\td     = %s' % self.ds[n])
            print('\tsigma = %s' % self.sigmas[n])

            # set up a solver
            #
            self.solver = NewtonRaphsonSolver(self.solver_input_block,
                                              len(self.genesis_mesh.nodal_coordinates),
                                              1,
                                              self.genesis_mesh.connectivity,
                                              self.enforce_bcs_on_u,
                                              self.enforce_bcs_on_residual,
                                              self.enforce_bcs_on_tangent,
                                              self.assemble_linear_system)

            # make an initial guess on the solution
            #
            u_0 = jnp.zeros(len(self.genesis_mesh.nodal_coordinates[:, 0] * self.n_dof_per_node),
                            dtype=jnp.float64)

            self.u = self.solver.solve(0, u_0, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values)
            self.I = self.u
            self.post_process_2d(1, 0.0)

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


    def supg_factor(self, coords):

        epsilon = 1e-12

        h_xi = 0.5 * jnp.sqrt((coords[2, 0] + coords[1, 0] - coords[0, 0] - coords[3, 0])**2 +
                              (coords[2, 1] + coords[1, 1] - coords[0, 1] - coords[3, 1])**2)
        h_eta = 0.5 * jnp.sqrt((coords[2, 0] + coords[3, 0] - coords[0, 0] - coords[1, 0])**2 +
                               (coords[2, 1] + coords[3, 1] - coords[0, 1] - coords[1, 1])**2)

        e_xi = jnp.zeros((2, 1), dtype=jnp.float64)
        e_eta = jnp.zeros((2, 1), dtype=jnp.float64)


        e_xi = jax.ops.index_update(e_xi, jax.ops.index[0, 0],
                                    (coords[2, 0] + coords[1, 0] - coords[0, 0] - coords[3, 0]) / (2.0 * h_xi))
        e_xi = jax.ops.index_update(e_xi, jax.ops.index[1, 0],
                                    (coords[2, 1] + coords[1, 1] - coords[0, 1] - coords[3, 1]) / (2.0 * h_xi))
        #
        e_eta = jax.ops.index_update(e_eta, jax.ops.index[0, 0],
                                     (coords[2, 0] + coords[3, 0] - coords[0, 0] - coords[1, 0]) / (2.0 / h_eta))
        e_eta = jax.ops.index_update(e_eta, jax.ops.index[1, 0],
                                     (coords[2, 1] + coords[3, 1] - coords[0, 1] - coords[1, 1]) / (2.0 / h_eta))

        d_xi = self.ds[0][0, 0] * e_xi[0, 0] + self.ds[0][1, 0] * e_xi[1, 0]
        d_eta = self.ds[0][0, 0] * e_eta[1, 0] + self.ds[0][1, 0] * e_eta[1, 0]

        alpha_xi = (d_xi * h_xi) / (2.0 * epsilon)
        alpha_eta = (d_eta * h_eta) / (2.0 * epsilon)

        # if jnp.abs(alpha_xi) <= 0.1:
        #     xi_bar = (1.0 / 3.0) * alpha_xi - (1.0 / 45.0) * alpha_xi**3 + \
        #              (2.0 / 945.0) * alpha_xi**5 - (1.0 / 4725.0) * alpha_xi**7
        # else:
        xi_bar = 1 / jnp.tanh(alpha_xi) - 1.0 / alpha_xi

        # if jnp.abs(alpha_eta) <= 0.1:
        #     eta_bar = (1.0 / 3.0) * alpha_eta - (1.0 / 45.0) * alpha_eta**3 + \
        #              (2.0 / 945.0) * alpha_eta**5 - (1.0 / 4725.0) * alpha_eta**7
        # else:
        eta_bar = 1 / jnp.tanh(alpha_eta) - 1.0 / alpha_eta

        supg = (xi_bar * d_xi * h_xi + eta_bar * d_eta * h_eta) / 2.0

        return supg

    def calculate_element_level_residual(self, nodal_fields):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, I_nodal = nodal_fields
        R_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0]), dtype=jnp.float64)
        # supg = self.supg_factor(coords)
        # supg = 0.025 / 2
        supg = self.blocks_input_block[0]['radiative_transfer_properties']['h'] / 2

        def quadrature_calculation(q, R_element):
            """
            :param q: the index of the current quadrature point
            :param R_element: the element level residual
            :return: the element level residual for this quadrature point only
            """
            N_xi = self.element_objects[0].N_xi[q, :, :]
            grad_N_X = self.element_objects[0].map_shape_function_gradients(coords)[q, :, :]
            JxW = self.element_objects[0].calculate_JxW(coords)[q, 0]

            I_q = jnp.matmul(I_nodal, N_xi)
            grad_I_q = jnp.matmul(grad_N_X.T, I_nodal).reshape((-1, 1))
            d_dot_grad_I_q = self.ds[0][0, 0] * grad_I_q[0, 0] + self.ds[0][1, 0] * grad_I_q[1, 0]
            d_dot_grad_N_X = self.ds[0][0, 0] * grad_N_X[0, 0] + self.ds[0][1, 0] * grad_N_X[1, 0]
            residual = d_dot_grad_I_q + self.sigmas[0] * I_q
            R_q = JxW * N_xi * residual + \
                  JxW * supg * d_dot_grad_N_X * residual

            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def assemble_linear_system(self, u):

        # set up residual and grab connectivity for convenience
        #
        residual = jnp.zeros_like(u)
        connectivity = self.genesis_mesh.connectivity

        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        I_element_wise = u[connectivity]

        # jit the element level residual calculator
        #
        jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            R_e = jit_calculate_element_level_residual((coordinates[e],
                                                        I_element_wise[e]))
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        # adjust residual to satisfy dirichlet conditions
        #
        residual, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
                                        self.enforce_bcs_on_residual, (residual, self.dirichlet_bcs_nodes))

        return residual

    def post_process_2d(self, time_step, time):
        self.post_processor.exo.put_time(time_step, time)
        self.post_processor.write_nodal_scalar_variable('I', time_step, jnp.asarray(self.I))
