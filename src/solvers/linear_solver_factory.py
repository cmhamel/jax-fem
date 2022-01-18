def linear_solver_factory(linear_solver_input_settings: dict):
    if linear_solver_input_settings['type'].lower() == 'gmres':
        from jax.scipy.sparse.linalg import gmres
        return gmres
    else:
        assert False, 'Unsupported linear solver!'
