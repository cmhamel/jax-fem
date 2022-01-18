import jax.numpy as jnp
from jax import jacfwd
from elements import element_factory
from elements import ElementBaseClass


class KernelBaseClass:
    def __init__(self, kernel_input_settings: dict, number_of_dimensions: int) -> None:
        self.kernel_input_settings = kernel_input_settings
        self.number_of_dimensions = number_of_dimensions
        self.coupled = False  # used to track which kernels are coupled

        # self.blocks = self.kernel_input_settings/

        # TODO: need to add in hooks to element types
        #
        self.element = self._setup_element()

    def _setup_element(self) -> ElementBaseClass:
        if self.number_of_dimensions == 1:
            element_type = 'line_element'
        elif self.number_of_dimensions == 2:
            element_type = 'quad_element'
        else:
            assert False, 'Unsupported number of dimensions in Kernel'

        element_input_settings = {'element_type': element_type,
                                  'quadrature_order': self.kernel_input_settings['quadrature_order'],
                                  'shape_function_order': self.kernel_input_settings['shape_function_order']}
        return element_factory(element_input_settings)

    def calculate_residual(self,
                           nodal_coordinates: jnp.ndarray,
                           connectivity: jnp.ndarray,
                           u: jnp.ndarray) -> jnp.ndarray:
        assert False, 'This needs to be overridden in your derived class!'

    # def _calculate_element_level_residual(self,
    #                                       element_level_coordinates: jnp.ndarray,
    #                                       element_level_connectivity: jnp.ndarray,
    #                                       element_level_u: jnp.ndarray) -> jnp.ndarray:
    #     assert False, 'This needs to be overridden in your derived class!'
