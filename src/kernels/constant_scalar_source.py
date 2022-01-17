import jax.numpy as jnp
import jax.ops

from .kernel_base_class import KernelBaseClass


class ConstantScalarSource(KernelBaseClass):
    def __init__(self, kernel_input_settings: dict) -> None:
        super(ConstantScalarSource, self).__init__(kernel_input_settings)
        self.variable = self.kernel_input_settings['variable']
        self.constant = self.kernel_input_settings['constant']

    def __str__(self) -> str:
        string = 'ConstantScalarSource(KernelBaseClass):\n'
        string = string + '\tVariable = %s\n' % self.variable
        string = string + '\tConstant = %s\n' % self.constant
        return string

    def calculate_residual(self,
                           nodal_coordinates: jnp.ndarray,
                           connectivity: jnp.ndarray,
                           u: jnp.ndarray) -> jnp.ndarray:

        # elementify the stuff
        #
        element_coordinates = nodal_coordinates[connectivity]
        element_us = u[connectivity]
        residual = jnp.zeros_like(u)

        def calculate_element_level_residual(residual_temp: jnp.ndarray, inputs: tuple) -> tuple:
            element_level_coordinates, element_level_connectivity, element_level_us = inputs
            element_level_us = element_level_us.reshape((-1, 1))
            element_level_residual = jnp.zeros_like(element_level_us)

            for q in range(self.element.number_of_quadrature_points):
                JxW = self.element.calculate_JxW(element_level_coordinates, q)
                N_xi = self.element.N_xi[q]
                element_level_residual = element_level_residual + \
                                         JxW * self.constant * N_xi

            residual_temp = jax.ops.index_add(residual_temp, jax.ops.index[element_level_connectivity],
                                              element_level_residual[:, 0])
            return residual_temp, inputs

        residual, _ = jax.lax.scan(calculate_element_level_residual, residual, (element_coordinates,
                                                                                connectivity,
                                                                                element_us))
        return residual
