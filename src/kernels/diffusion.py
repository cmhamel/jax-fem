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

    def calculate_element_level_residual(self,
                                         element_level_coordinates: jnp.ndarray,
                                         element_level_u: jnp.ndarray) -> jnp.ndarray:

        element_level_residual = jnp.zeros_like(element_level_u)

        def loop_body(q: int, temp_element_level_residual: jnp.ndarray) -> jnp.ndarray:
            JxW = self.element.calculate_JxW(element_level_coordinates, q)
            grad_N_X = self.element.map_shape_function_gradients(element_level_coordinates, q)
            grad_u_q = jnp.matmul(grad_N_X.T, element_level_u)
            temp_element_level_residual = temp_element_level_residual + \
                                          JxW * self.D * jnp.matmul(grad_N_X, grad_u_q)
            return temp_element_level_residual

        element_level_residual = jax.lax.fori_loop(0, self.element.number_of_quadrature_points,
                                                   loop_body, element_level_residual)

        return element_level_residual
