import jax.scipy.sparse.linalg

from pre_processing import GenesisMesh
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition
from ..source import Source
from elements import LineElement
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

        # solve
        #
        self.solver_input_block = self.physics_input['solver']
        self.u = self.solve()
        self.post_process()

    def calculate_element_level_residual(self, nodal_fields):
        """
        :param nodal_fields: relevant nodal fields
        :return: the integrated element level residual vector
        """
        coords, u_nodal, source = nodal_fields
        R_e = jnp.zeros((self.genesis_mesh.n_nodes_per_element[0], 1), dtype=jnp.float64)

        def quadrature_calculation(q, R_element):
            """
            :param q: the index of the current quadrature point
            :param R_element: the element level residual
            :return: the element level residual for this quadrature point only
            """
            N_xi = self.element_objects[0].N_xi[q, :, :]
            grad_N_X = self.element_objects[0].map_shape_function_gradients(coords)[q, :, :]
            J = self.element_objects[0].calculate_deriminant_of_jacobian_map(coords)
            w = self.element_objects[0].w[q, 0]

            s_q = jnp.matmul(N_xi.T, source)
            grad_u_q = jnp.matmul(grad_N_X.T, u_nodal)
            R_q = J * w * (s_q * N_xi - grad_u_q * grad_N_X)
            R_element = jax.ops.index_add(R_element, jax.ops.index[:, :], R_q)
            return R_element

        R_e = jax.lax.fori_loop(0, self.element_objects[0].n_quadrature_points, quadrature_calculation, R_e)

        return R_e

    def assemble_linear_system(self, u):

        # set up residual and grab connectivity for convenience
        #
        residual = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape, dtype=jnp.float64)
        coordinates = self.genesis_mesh.nodal_coordinates
        connectivity = self.genesis_mesh.element_connectivities[0]
        n_nodes_per_el = self.genesis_mesh.n_nodes_per_element[0]

        # jit the element level residual calculator
        #
        jit_calculate_element_level_residual = jit(self.calculate_element_level_residual)

        for e in range(self.genesis_mesh.n_elements_in_blocks[0]):
            coords = coordinates[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]
            u_nodal = u[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]
            source = self.source.values[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]

            R_e = jit_calculate_element_level_residual((coords, u_nodal, source))

            residual = jax.ops.index_add(residual, jax.ops.index[connectivity[n_nodes_per_el * e:
                                                                              n_nodes_per_el * (e + 1)]], R_e)


        # def element_calculation(e, R):
        #     coords = coordinates[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]
        #     u_nodal = u[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]
        #     source = self.source.values[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]]
        #
        #     R_e = jit_calculate_element_level_residual((coords, u_nodal, source))
        #     R = jax.ops.index_update(R, jax.ops.index[connectivity[n_nodes_per_el * e:n_nodes_per_el * (e + 1)]], R_e)
        #
        #     return R
        #
        # residual = jax.lax.fori_loop(0, self.genesis_mesh.n_elements_in_blocks[0], element_calculation, residual)

        # adjust residual to satisfy dirichlet conditions
        #
        for i, bc in enumerate(self.dirichlet_bcs):
            residual = jax.ops.index_update(residual, jax.ops.index[bc.node_set_nodes, 0], 0)

        return residual[:, 0]
    

    def solve(self):

        # jit the relevant methods
        #
        jit_assemble_linear_system = jit(self.assemble_linear_system)
        jit_tangent = jit(jacfwd(self.assemble_linear_system))
        jit_gmres = jit(jax.scipy.sparse.linalg.gmres)

        # initialize the solution vector
        #
        u_solve = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape, dtype=jnp.float64)
        residual_solve = jnp.zeros_like(u_solve)
        delta_u_solve = jnp.zeros_like(u_solve)
        # u = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape, dtype=jnp.float64)

        # begin loop over non-linear iterations
        #
        # n = 0
        # while n <= self.solver_input_block['maximum_iterations']:

        def solver_function(values):

            # n, u = values
            # u = values
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
            residual = jit_assemble_linear_system(u)

            # calculate the tangent matrix using auto-differentiation
            #
            tangent = jit_tangent(residual)

            # enforce dirichlet BCs in the tangent matrix
            #
            def enforce_bcs_on_tangent(i, input):
                tangent_temp, bcs_nodes, bcs_values = input
                bc_nodes, bc_values = bcs_nodes[i], bcs_values[i]
                tangent_temp = jax.ops.index_update(tangent_temp, jax.ops.index[bc_nodes], bc_values)
                return tangent_temp, bcs_nodes, bcs_values

            tangent, _, _ = jax.lax.fori_loop(0, len(self.dirichlet_bcs), enforce_bcs_on_tangent,
                                              (tangent, self.dirichlet_bcs_nodes, self.dirichlet_bcs_values))

            # solve for solution increment
            #
            delta_u, _ = jit_gmres(tangent, residual)

            # update the solution increment, note where the minus sign is
            #
            u = jax.ops.index_add(u, jax.ops.index[:, 0], -delta_u)

            # print some stuff about the solution status
            #
            # print('|R| = %s' % jnp.linalg.norm(residual))
            # print(r'$|\Delta U|$ = %s' % jnp.linalg.norm(delta_u))
            # # print(n)
            # # n = n + 1
            #
            # # print(residual.shape)
            # # print(delta_u.shape)
            # # print(u.shape)
            # # assert False
            #
            # # norm_R, norm_delta_u = jnp.norm(residual), jnp.norm(delta_u)
            #
            # print(jnp.linalg.norm(residual).ravel())
            # print(jnp.linalg.norm(delta_u).ravel())
            print('here')
            return residual.reshape((-1, 1)), delta_u.reshape((-1, 1)), u
            # return n, u

        print('========================================')
        print('=== Iteration %s' % 0)
        print('========================================')
        
        # condition_function = lambda output: output[0] <= self.solver_input_block['maximum_iterations']
        condition_function = lambda output: jnp.linalg.norm(output[0]) < 1e-12
        print('here')
        # residual_solve, delta_u_solve, u_solve = jax.lax.while_loop(condition_function, solver_function,
        #                                                             (residual_solve, delta_u_solve, u_solve))
        print('solution found')
        n = 0
        while n <= self.solver_input_block['maximum_iterations']:
            output = solver_function((residual_solve, delta_u_solve, u_solve))
            n = n + 1
            residual_solve, delta_u_solve, u_solve = output

            print(jnp.linalg.norm(residual_solve).ravel())
            print(jnp.linalg.norm(delta_u_solve).ravel())

            print(condition_function(output))

        # print(residual_solve)
        # print(delta_u_solve)
        # print(u_solve)

        return u_solve

    def post_process(self):
        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, self.u)
        plt.plot(self.genesis_mesh.nodal_coordinates, -0.5 * self.genesis_mesh.nodal_coordinates**2 +
                 0.5 * self.genesis_mesh.nodal_coordinates,
                 linestyle='None', marker='o')
        plt.show()
