import jax
import jax.numpy as jnp
from .element_base_class import ElementBaseClass


class HexElement(ElementBaseClass):
    def __init__(self, element_input_settings: dict) -> None:
        super(HexElement, self).__init__(element_input_settings)

    def _setup_number_of_quadrature_points(self) -> int:
        if self.quadrature_order == 1:
            return 1
        elif self.quadrature_order == 2:
            return 8
        else:
            assert False, 'Unsupported'

    def _setup_number_of_shape_functions(self) -> int:
        if self.shape_function_order == 1:
            return 8
        else:
            assert False, 'Unsupported'

    def _calculate_quadrature(self) -> (jnp.ndarray, jnp.ndarray):
        if self.quadrature_order == 1 and self.shape_function_order == 1:
            xi = jnp.array([[0.0, 0.0, 0.0]])
            w = jnp.array([8.0])
            return xi, w
        elif self.quadrature_order == 2 and self.shape_function_order == 1:
            xi = jnp.array([[-1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                            [1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                            [-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                            [1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0)],
                            [-1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)],
                            [1.0 / jnp.sqrt(3.0), -1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)],
                            [-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)],
                            [1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)]])
            w = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            return xi, w
        else:
            assert False, 'Unsupported quadrature order/shape function order combination'

    def _calculate_shape_function_values(self) -> jnp.ndarray:
        N_xi = jnp.zeros((self.number_of_quadrature_points, self.number_of_shape_functions, 1))
        if self.shape_function_order == 1:
            for q in range(self.number_of_quadrature_points):
                N_xi = \
                jax.ops.index_update(N_xi,
                jax.ops.index[q, :, :],
                0.125 *
                jnp.array([[(1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 1]) * (1.0 - self.xi[q, 2])],
                           [(1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 1]) * (1.0 - self.xi[q, 2])],
                           [(1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 1]) * (1.0 - self.xi[q, 2])],
                           [(1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 1]) * (1.0 - self.xi[q, 2])],
                           [(1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 1]) * (1.0 + self.xi[q, 2])],
                           [(1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 1]) * (1.0 + self.xi[q, 2])],
                           [(1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 1]) * (1.0 + self.xi[q, 2])],
                           [(1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 1]) * (1.0 + self.xi[q, 2])]]
                         ))
            return N_xi
        else:
            assert False, 'Unsupported shape function order'

    def _calculate_shape_function_gradients(self) -> jnp.ndarray:
        grad_N_xi = jnp.zeros((self.number_of_quadrature_points, self.number_of_shape_functions, 3))
        if self.shape_function_order == 1:
            for q in range(self.number_of_quadrature_points):
                grad_N_xi = \
                jax.ops.index_update(grad_N_xi, jax.ops.index[q, :, :],
                0.125 * jnp.array([[-(1.0 - self.xi[q, 1]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 1])],  # grad_N_xi_1
                                   [(1.0 - self.xi[q, 1]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 1])],  # grad_N_xi_2
                                   [(1.0 + self.xi[q, 1]) * (1.0 - self.xi[q, 2]),
                                    (1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 1])],  # grad_N_xi_3
                                   [-(1.0 + self.xi[q, 1]) * (1.0 - self.xi[q, 2]),
                                    (1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 2]),
                                    -(1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 1])],  # grad_N_xi_4
                                   [-(1.0 - self.xi[q, 1]) * (1.0 + self.xi[q, 2]),
                                    -(1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 2]),
                                    (1.0 - self.xi[q, 0]) * (1.0 - self.xi[q, 1])],  # grad_N_xi_5
                                   [(1.0 - self.xi[q, 1]) * (1.0 + self.xi[q, 2]),
                                    -(1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 2]),
                                    (1.0 + self.xi[q, 0]) * (1.0 - self.xi[q, 1])],  # grad_N_xi_6
                                   [(1.0 + self.xi[q, 1]) * (1.0 + self.xi[q, 2]),
                                    (1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 2]),
                                    (1.0 + self.xi[q, 0]) * (1.0 + self.xi[q, 1])],  # grad_N_xi_7
                                   [-(1.0 + self.xi[q, 1]) * (1.0 + self.xi[q, 2]),
                                    (1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 2]),
                                    (1.0 - self.xi[q, 0]) * (1.0 + self.xi[q, 1])]   # grad_N_xi_8
                ]))
            return grad_N_xi
        else:
            assert False, 'Unsupported shape function order'
