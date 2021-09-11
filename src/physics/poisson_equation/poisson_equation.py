import jax.scipy.sparse.linalg

from pre_processing import GenesisMesh
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition
from ..source import Source
from elements import LineElement, QuadElement
import jax.numpy as jnp
from jax import jacfwd, jit, partial, vmap
import jax


class PoissonEquation(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(PoissonEquation, self).__init__(n_dimensions, physics_input)
        print(self)

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
        for n, block in enumerate(self.blocks_input_block):
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

        # set up the source
        #
        # TODO make a source block dependent
        #
        source_input_block = self.physics_input['source']
        self.source = Source(self.genesis_mesh.nodal_coordinates.shape,
                             source_input_block['type'],
                             source_input_block['value'])
        # print(source_input_block)
        print(self.source.values)

        # solve
        #
        self.solver_input_block = self.physics_input['solver']
        self.u = self.solve()

        if n_dimensions == 1:
            self.post_process_1d()
        elif n_dimensions == 2:
            self.post_process_2d()
            self.post_processor.exo.close()
        else:
            assert False

    def calculate_element_level_residual(self, nodal_fields):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, u_nodal, source = nodal_fields
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

            s_q = jnp.matmul(N_xi.T, source)
            grad_u_q = jnp.matmul(grad_N_X.T, u_nodal).reshape((-1, 1))

            # R_q = JxW * (s_q * N_xi - grad_u_q * grad_N_X)
            R_q = JxW * (s_q * N_xi - jnp.matmul(grad_N_X, grad_u_q))

            R_element = jax.ops.index_add(R_element, jax.ops.index[:], R_q[:, 0])

            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def assemble_linear_system(self, u):

        # set up residual and grab connectivity for convenience
        #
        residual = jnp.zeros(self.genesis_mesh.nodal_coordinates[:, 0].shape, dtype=jnp.float64)
        connectivity = self.genesis_mesh.connectivity

        coordinates = self.genesis_mesh.nodal_coordinates[connectivity]
        u_element_wise = u[connectivity]
        source_element_wise = jnp.ones_like(u_element_wise)

        # jit the element level residual calculator
        #
        jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        def element_calculation(e, input):
            residual_temp = input
            R_e = jit_calculate_element_level_residual((coordinates[e], u_element_wise[e], source_element_wise[e]))
            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[connectivity[e]], R_e)
            return residual_temp

        residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        # adjust residual to satisfy dirichlet conditions
        #
        def enforce_bcs_on_residual(i, input):
            residual_temp, bcs_nodes = input
            bc_nodes = bcs_nodes[i]
            residual_temp = jax.ops.index_update(residual_temp, jax.ops.index[bc_nodes], 0.0)
            return residual_temp, bcs_nodes

        residual, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
                                        enforce_bcs_on_residual, (residual, self.dirichlet_bcs_nodes))

        return residual

    def solve(self):

        # jit the relevant methods
        #
        jit_assemble_linear_system = jit(self.assemble_linear_system)    # TODO problem with jit here
                                                                         # TODO some nodal arrays are changing shape
                                                                         # TODO every other iteration
        jit_tangent = jit(jacfwd(self.assemble_linear_system))         # TODO problem is actually when tangent
                                                                         # TODO is jitted
        jit_gmres = jit(jax.scipy.sparse.linalg.gmres)

        # initialize the solution vector
        #
        u_solve = jnp.zeros(self.genesis_mesh.nodal_coordinates[:, 0].shape, dtype=jnp.float64)
        residual_solve = jnp.zeros_like(u_solve)
        delta_u_solve = jnp.zeros_like(u_solve)

        # begin loop over non-linear iterations
        #
        # n = 0
        # while n <= self.solver_input_block['maximum_iterations']:

        def solver_function(values):

            residual, delta_u, u = values

            # force u to satisfy dirichlet conditions on the bcs
            #
            def enforce_bcs_on_u(i, input):
                u_temp, bcs_nodes, bcs_values = input
                bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
                u_temp = jax.ops.index_update(u_temp, jax.ops.index[bc_nodes], bc_values)
                return u_temp, bcs_nodes, bcs_values

            u, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs),
                                        enforce_bcs_on_u, (u, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

            # assemble the residual
            #
            residual = jit_assemble_linear_system(u)  # TODO once the jit is fixed above uncomment this

            # assert False
            # calculate the tangent matrix using auto-differentiation
            #
            tangent = jit_tangent(residual)

            # enforce dirichlet BCs in the tangent matrix
            #
            def enforce_bcs_on_tangent(i, input):
                tangent_temp, bcs_nodes, bcs_values = input
                bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
                tangent_temp = jax.ops.index_update(tangent_temp, jax.ops.index[bc_nodes, bc_nodes], 1.0)
                return tangent_temp, bcs_nodes, bcs_values

            tangent, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs), enforce_bcs_on_tangent,
                                              (tangent, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

            # solve for solution increment
            #
            delta_u, _ = jit_gmres(tangent, residual)

            # update the solution increment, note where the minus sign is
            #
            u = jax.ops.index_add(u, jax.ops.index[:], -delta_u)

            return residual, delta_u, u

        # begin solver loop
        #
        n = 0
        self.print_solver_heading()
        while n <= self.solver_input_block['maximum_iterations']:
            output = solver_function((residual_solve, delta_u_solve, u_solve))
            n = n + 1
            residual_solve, delta_u_solve, u_solve = output

            residual_error, increment_error = jnp.linalg.norm(residual_solve), jnp.linalg.norm(delta_u_solve)

            self.print_solver_state(n, residual_error.ravel(), increment_error.ravel())

            if residual_error < self.solver_input_block['residual_tolerance']:
                print('Converged on residual: |R| = {0:.8e}'.format(residual_error.ravel()[0]))
                break

            if increment_error < self.solver_input_block['increment_tolerance']:
                print('Converged on increment: |du| = {0:.8e}'.format(increment_error.ravel()[0]))
                break

        return u_solve

    def print_solver_heading(self):
        print('-----------------------------------------------------')
        print('--- Time step %s' % 0)
        print('-----------------------------------------------------')
        print('Iteration\t\t|R|\t\t|du|')

    def print_solver_state(self, increment, residual_error, increment_error):
        print('\t{0:4}\t\t{1:.8e}\tt{2:.8e}'.format(increment, residual_error[0], increment_error[0]))

    def post_process_1d(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.plot(self.genesis_mesh.nodal_coordinates, -0.5 * self.genesis_mesh.nodal_coordinates**2 +
                 0.5 * self.genesis_mesh.nodal_coordinates,
                 linestyle='None', marker='o')
        plt.show()

    def post_process_2d(self):
        self.post_processor.exo.put_time(1, 0.0)
        self.post_processor.write_nodal_scalar_variable('u', 1, jnp.asarray(self.u))
