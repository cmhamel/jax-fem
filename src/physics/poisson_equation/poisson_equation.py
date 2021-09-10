import jax.scipy.sparse.linalg

from pre_processing import GenesisMesh
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition
from ..source import Source
from elements import LineElement
import jax.numpy as jnp
from jax import jacfwd, jit, partial


class PoissonEquation(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(PoissonEquation, self).__init__(n_dimensions, physics_input)
        print(self)

        # get some boundary conditions
        #
        self.dirichlet_bcs = []
        for key in self.boundary_conditions_input_block.keys():
            bc_type = self.boundary_conditions_input_block[key]
            for i, bc in enumerate(bc_type):
                self.dirichlet_bcs.append(
                    DirichletBoundaryCondition(node_set_name=bc['node_set'],
                                               node_set_nodes=self.genesis_mesh.node_set_nodes[i],
                                               bc_type=bc['type'].lower(),
                                               value=bc['value']))

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
        self.solve()

    # @partial(jit, static_argnums=(0,))
    def assemble_linear_system(self, u):
        residual = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape, dtype=jnp.float64)
        connectivity = self.genesis_mesh.element_connectivities[0]

        # make support for multiple blocks
        #
        n_nodes_per_element = self.genesis_mesh.n_nodes_per_element[0]
        for e in range(self.genesis_mesh.n_elements_in_blocks[0]):

            # get coordinates and other nodal stuff
            #
            coords = \
                self.genesis_mesh.nodal_coordinates[self.genesis_mesh.element_connectivities[0]
                [e * n_nodes_per_element:(e + 1) * n_nodes_per_element]]
            u_nodal = \
                u[self.genesis_mesh.element_connectivities[0][e * n_nodes_per_element:(e + 1) * n_nodes_per_element]]
            source = \
                self.source.values[self.genesis_mesh.element_connectivities[0]
                [e * n_nodes_per_element:(e + 1) * n_nodes_per_element]]

            # get element level residual
            #
            R_e = jnp.zeros((n_nodes_per_element, 1))
            for q in range(self.element_objects[0].n_quadrature_points):
                N_xi = self.element_objects[0].N_xi[q, :, :]
                grad_N_X = self.element_objects[0].map_shape_function_gradients(coords)[q, :, :]
                J = self.element_objects[0].calculate_deriminant_of_jacobian_map(coords)
                w = self.element_objects[0].w[q, 0]
                s_e = 1.0

                grad_u_e = jnp.matmul(grad_N_X.T, u_nodal)
                R_e = R_e + J * w * (s_e * N_xi - grad_u_e * grad_N_X)

            # add this to the global residual
            #
            residual = residual.at[connectivity[e * n_nodes_per_element:(e + 1) * n_nodes_per_element]].\
                set(residual[connectivity[e * n_nodes_per_element:(e + 1) * n_nodes_per_element]] + R_e)

            # adjust residual to satisfy dirichlet conditions
            #
            for i, bc in enumerate(self.dirichlet_bcs):
                residual = residual.at[bc.node_set_nodes, 0].set(0)

        return residual[:, 0]

    def solve(self):
        u = jnp.zeros(self.genesis_mesh.nodal_coordinates.shape, dtype=jnp.float64)

        n = 0
        while n <= self.solver_input_block['maximum_iterations']:
            # force u to satisfy dirichlet conditions on the bcs
            #
            for i, bc in enumerate(self.dirichlet_bcs):
                u = u.at[bc.node_set_nodes, 0].set(bc.values)

            # residual = self.assemble_linear_system(u)
            # print(residual)
            residual = self.assemble_linear_system(u).reshape((-1, 1))
            tangent = jacfwd(self.assemble_linear_system)(u[:, 0])

            for i, bc in enumerate(self.dirichlet_bcs):
                tangent = tangent.at[bc.node_set_nodes, bc.node_set_nodes].set(1.0)

            # delta_u = -jnp.matmul(jnp.linalg.inv(tangent), residual)
            delta_u, _ = jax.scipy.sparse.linalg.gmres(tangent, residual)
            # u = u + delta_u
            u = u - delta_u
            print('========================================')
            print('=== Iteration %s' % n)
            print('========================================')
            print('')
            print('|R| = %s' % jnp.linalg.norm(residual))
            print(r'$|\Delta U|$ = %s' % jnp.linalg.norm(delta_u))

            n = n + 1

        import matplotlib.pyplot as plt
        plt.plot(self.genesis_mesh.nodal_coordinates, u)
        plt.plot(self.genesis_mesh.nodal_coordinates, -0.5 * self.genesis_mesh.nodal_coordinates**2 +
                 0.5 * self.genesis_mesh.nodal_coordinates,
                 linestyle='None', marker='o')
        plt.show()
