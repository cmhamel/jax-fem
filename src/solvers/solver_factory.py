from .solver_base_class import SolverBaseClass


def solver_factory(solver_input_settings: dict,
                   variables: list,
                   kernels: list,
                   boundary_conditions: list,
                   bc_update_solution_methods: list,
                   residual_methods: list,
                   tangent_methods_diagonal: list,
                   tangent_methods_off_diagonal=None) -> SolverBaseClass:
    if solver_input_settings['type'].lower() == 'newton_solver':
        from .newton_solver import NewtonSolver
        return NewtonSolver(solver_input_settings,
                            variables, kernels, boundary_conditions, bc_update_solution_methods,
                            residual_methods, tangent_methods_diagonal, tangent_methods_off_diagonal)
    else:
        assert False, 'Unsupported solver type!'
