from .boundary_condition_base_class import BoundaryConditionBaseClass


def boundary_condition_factory(boundary_condition_input_settings: dict) -> BoundaryConditionBaseClass:
    if boundary_condition_input_settings['type'].lower() == 'constant_dirichlet':
        from .constant_dirichlet_boundary_condition import ConstantDirichletBoundaryCondition
        return ConstantDirichletBoundaryCondition(boundary_condition_input_settings)
    else:
        assert False, 'Unsupported boundary condition!'
