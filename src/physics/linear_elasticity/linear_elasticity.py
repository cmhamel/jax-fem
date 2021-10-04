import jax
import jax.numpy as jnp
from jax import jit
from elements import QuadElement
from solvers import NewtonRaphsonSolver
from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition


class LinearElasticity(Physics):
    def __init__(self, n_dimensions, physics_input):
        super(LinearElasticity, self).__init__(n_dimensions, physics_input)

        self.n_dof_per_node = 2

        # need to make dof connectivity matrix
        #
        self.dof_connectivity = self.genesis_mesh.make_multiple_dof_connectivity(self.n_dof_per_node)