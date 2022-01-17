from .kernel_base_class import KernelBaseClass


def kernel_factory(kernel_input_settings: dict) -> KernelBaseClass:
    kernel_type = kernel_input_settings['type']
    if kernel_type.lower() == 'diffusion':
        from .diffusion import Diffusion
        return Diffusion(kernel_input_settings)
    elif kernel_type.lower() == 'constant_scalar_source':
        from .constant_scalar_source import ConstantScalarSource
        return ConstantScalarSource(kernel_input_settings)
    else:
        assert False, 'Unsupported Kernel'
