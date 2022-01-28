import jax.numpy as jnp
import jax.ops

from .kernel_base_class import KernelBaseClass


class Diffusion(KernelBaseClass):
    def __init__(self, kernel_input_settings: dict, number_of_dimensions: int) -> None:
        super(Diffusion, self).__init__(kernel_input_settings, number_of_dimensions)
        self.variable = kernel_input_settings['variable']
        self.D = kernel_input_settings['D']

    def __str__(self) -> str:
        string = '    ' + __class__.__name__ + ':\n'
        string = string + '      Variable = %s\n' % self.variable
        string = string + '      D        = %s\n\n' % self.D
        return string

    def _calculate_element_level_residual(self, residual_temp: jnp.ndarray, inputs: tuple) -> tuple:
        element_level_coordinates, element_level_connectivity, element_level_us = inputs
        element_level_us = element_level_us.reshape((-1, 1))
        element_level_residual = jnp.zeros_like(element_level_us)

        for q in range(self.element.number_of_quadrature_points):
            JxW = self.element.calculate_JxW(element_level_coordinates, q)
            grad_N_X = self.element.map_shape_function_gradients(element_level_coordinates, q)
            grad_u_q = jnp.matmul(grad_N_X.T, element_level_us)
            element_level_residual = element_level_residual - \
                                     JxW * self.D * jnp.matmul(grad_N_X, grad_u_q)

        residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[element_level_connectivity],
                                          element_level_residual[:, 0])

        return residual_temp, inputs

    def calculate_residual(self,
                           nodal_coordinates: jnp.ndarray,
                           connectivity: jnp.ndarray,
                           u: jnp.ndarray) -> jnp.ndarray:

        # elementify the stuff
        #
        element_coordinates = nodal_coordinates[connectivity]
        element_us = u[connectivity]

        # initialize residual to zeros and scan over elements
        #
        residual = jnp.zeros_like(u)
        residual, _ = jax.lax.scan(self._calculate_element_level_residual,
                                   residual, (element_coordinates, connectivity, element_us))
        return residual
