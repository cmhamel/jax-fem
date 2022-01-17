from .mesh_base_class import MeshBaseClass


def mesh_factory(mesh_input_settings: dict, number_of_dimensions: int) -> MeshBaseClass:
    if 'genesis_mesh' in mesh_input_settings:
        from .genesis_mesh import GenesisMesh
        return GenesisMesh(mesh_input_settings, number_of_dimensions)
    else:
        assert False, 'Unsupported mesh type'
