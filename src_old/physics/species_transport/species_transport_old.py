import jax
import jax.numpy as jnp
from jax import jit
from elements import LineElement
from elements import QuadElement
from solvers import NewtonRaphsonSolver
# from solvers import NewtonRaphsonTransientSolver
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition
from ..boundary_conditions import NeumannBoundaryCondition
from ..constitutive_models import FicksLaw


class SpeciesTransport(Physics):
    def __init__(self, n_dimensions, mesh_input):
        super(SpeciesTransport, self).__init__(n_dimensions, mesh_input)
        print(self)

        # set number of degress of freedom per node
        #
        self.n_species = self.physics_input['number_of_species']
        self.n_dof_per_node = self.n_species
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
                    DirichletBoundaryCondition(node_set_name=bc['node_set'],
                                               node_set_nodes=self.genesis_mesh.node_set_nodes[i],
                                               bc_type=bc['type'].lower(),
                                               value=bc['value']))
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
        self.u_old = jnp.zeros(len(self.genesis_mesh.nodal_coordinates[:, 0] * self.n_dof_per_node),
                               dtype=jnp.float64)

        self.time_control_block = self.physics_input['time_control']
        print(self.time_control_block)

        if self.time_control_block['type'].lower() == 'transient':
            self.t_0 = self.time_control_block['time_initial']
            self.t_end = self.time_control_block['time_end']
            self.delta_t = self.time_control_block['time_increment']

            for n in range(len(self.dirichlet_bcs)):
                self.dirichlet_bcs[n].time_end = self.t_end

            # loop over time
            #
            self.t = self.t_0
            time_step = 0
            while self.t < self.t_end:

                # update dirichlet bcs
                #
                for n in range(len(self.dirichlet_bcs)):
                    temp_values = self.dirichlet_bcs[n].update_bc_values(time=self.t)
                    self.dirichlet_bcs_values = jax.ops.index_update(self.dirichlet_bcs_values, jax.ops.index[n, :],
                                                                     temp_values)
                # now solve
                #
                self.u = self.solver.solve(time_step, u_0, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values)

                # update
                #
                u_0 = self.u
                self.u_old = u_0

                time_step = time_step + 1
                self.t = self.t + self.delta_t

                # post-process
                #
                if n_dimensions == 1:
                    self.post_process_1d()
                elif n_dimensions == 2:
                    self.post_process_2d(time_step, self.t)
                else:
                    try:
                        assert False
                    except AssertionError:
                        raise Exception('Unsupported number of dimensions to postprocess')

            self.post_processor.exo.close()

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
            c_dot_q = (c_q - c_q_old) / self.delta_t
            grad_c_q = jnp.matmul(grad_N_X.T, c_nodal).reshape((-1, 1))

            q_q = self.constitutive_models[0][0].calculate_species_flux(grad_c_q)

            R_q = JxW * c_dot_q * N_xi - \
                  JxW * jnp.matmul(grad_N_X, q_q)
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
        c_element_wise = u[connectivity]
        c_element_wise_old = self.u_old[connectivity]

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

    def post_process_1d(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.show()

    def post_process_2d(self, time_step, time):
        self.post_processor.exo.put_time(time_step, time)
        self.post_processor.write_nodal_scalar_variable('c', time_step, jnp.asarray(self.u))
