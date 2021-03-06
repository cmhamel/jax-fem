from .linear_solver_factory import linear_solver_factory


class SolverBaseClass:
    def __init__(self,
                 solver_input_settings: dict,
                 variables: list,
                 kernels: list,
                 boundary_conditions: list,
                 bc_update_solution_methods: list,
                 residual_methods: list,
                 tangent_methods_diagonal: list,
                 tangent_methods_off_diagonal=None) -> None:
        self.solver_input_settings = solver_input_settings
        self.linear_solver_input_settings = self.solver_input_settings['linear_solver']
        self.linear_solver = linear_solver_factory(self.linear_solver_input_settings)

        self.variables = variables
        self.kernels = kernels
        self.boundary_conditions = boundary_conditions
        self.bc_update_solution_methods = bc_update_solution_methods
        self.residual_methods = residual_methods
        self.tangent_methods_diagonal = tangent_methods_diagonal
        self.tangent_methods_off_diagonal = tangent_methods_off_diagonal
