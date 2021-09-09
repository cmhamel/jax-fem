from ..physics import Physics
from ..boundary_conditions import DirichletBoundaryCondition
from ..boundary_conditions import NeumannBoundaryCondition


class SteadyStateHeatConduction(Physics):
    def __init__(self, n_dimensions, mesh_input):
        super(SteadyStateHeatConduction, self).__init__(n_dimensions, mesh_input)

        print(self.n_dimensions)
        print(self.genesis_file)
        print(self.nodal_coordinates)