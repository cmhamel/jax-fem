from .solver_base_class import SolverBaseClass


def solver_factory(solver_input_settings: dict,
                   variables: list,
                   kernels: list,
                   boundary_conditions: list,
                   residual_methods: list,
                   tangent_methods: list) -> SolverBaseClass:
    if solver_input_settings['type'].lower() == 'newton_solver':
        from .newton_solver import NewtonSolver
        return NewtonSolver(solver_input_settings,
                            variables, kernels, boundary_conditions,
                            residual_methods, tangent_methods)
    else:
        assert False, 'Unsupported solver type!'
