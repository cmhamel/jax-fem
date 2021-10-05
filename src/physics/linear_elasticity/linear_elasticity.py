import jax
import jax.numpy as jnp
from jax import jit
from util import general_tardigrade_error
from elements import QuadElement
from solvers import NewtonRaphsonSolver
from ..physics import Physics
from .boundary_conditions import DisplacementBoundaryCondition


class LinearElasticity(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(LinearElasticity, self).__init__(n_dimensions, physics_input)
        print(self)
        self.n_dof_per_node = 2

        # need to make dof connectivity matrix, TODO need to do this for each block eventually but currently
        #                                       TODO everything is operating on one block
        #
        self.dof_connectivity = self.genesis_mesh.make_multiple_dof_connectivity(self.n_dof_per_node)

        # read in blocks
        #
        self.element_objects = []
        self.constitutive_models = []
        for n, block in enumerate(self.blocks_input_block):
            # set up elements
            #
            self.element_objects.append(
                QuadElement(quadrature_order=block['cell_interpolation']['quadrature_order'],
                            shape_function_order=block['cell_interpolation']['shape_function_order']))
            print('Block number = %s' % str(n + 1))

        self.displacement_bcs = []
        if 'displacement' in self.boundary_conditions_input_block.keys():
            for n, bc in enumerate(self.boundary_conditions_input_block['displacement']):
                self.displacement_bcs.append(DisplacementBoundaryCondition(bc, self.n_dof_per_node,
                                                                           self.genesis_mesh, n))
                print(self.displacement_bcs[n])

        # set up force boundary conditions
        #
