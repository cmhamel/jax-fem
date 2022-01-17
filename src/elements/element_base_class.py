import jax.numpy as jnp


class ElementBaseClass:
    def __init__(self, element_input_settings: dict) -> None:
        self.quadrature_order = element_input_settings['quadrature_order']
        self.shape_function_order = element_input_settings['shape_function_order']

        assert self.quadrature_order > 0, 'Quadrature order must be greater then zero'
        assert self.shape_function_order > 0, 'Shape function order must be greater then zero'

        # method calls to get pre-calculated quantities
        #
        self.number_of_quadrature_points = self._setup_number_of_quadrature_points()
        self.number_of_shape_functions = self._setup_number_of_shape_functions()
        self.xi, self.w = self._calculate_quadrature()
        self.N_xi = self._calculate_shape_function_values()
        self.grad_N_xi = self._calculate_shape_function_gradients()

    def __str__(self) -> str:
        string = 'ElementBaseClass:\n'
        string = string + 'Quadrature Order            = %s\n' % self.quadrature_order
        string = string + 'Number of Quadrature Points = %s\n' % self.number_of_quadrature_points
        string = string + 'Shape Function Order        = %s\n' % self.shape_function_order
        string = string + 'Number of Shape Functions   = %s\n' % self.number_of_shape_functions
        string = string + '\n'
        return string

    # private methods to be overridden
    # the idea, is that these will only be called once and stored
    # the mapping methods will then utilize these pre-calculated
    # quantities and element coordinates to get mapped
    # quantities
    #
    def _setup_number_of_quadrature_points(self) -> int:
        assert False, 'You need to override this method in your element!'

    def _setup_number_of_shape_functions(self) -> int:
        assert False, 'You need to override this method in your element!'

    def _calculate_quadrature(self) -> (jnp.ndarray, jnp.ndarray):
        assert False, 'You need to override this method in your element!'

    def _calculate_shape_function_values(self) -> jnp.ndarray:
        assert False, 'You need to override this method in your element!'

    def _calculate_shape_function_gradients(self) -> jnp.ndarray:
        assert False, 'You need to override this method in your element!'

    # static methods to be called externally
    #
    def calculate_jacobian_map(self, nodal_coordinates: jnp.ndarray, q: int) -> jnp.ndarray:
        """
        Calculate the jacobian map between reference element and current
        :param nodal_coordinates: nodal coordinates of the elements
        :param q: quadrature point for indexing
        :return: jacobian map at quadrature point q
        """
        J_q = jnp.matmul(self.grad_N_xi[q, :, :].T, nodal_coordinates)
        return J_q

    def calculate_JxW(self, nodal_coordinates: jnp.ndarray, q: int) -> jnp.ndarray:
        """
        calculate the term detJ x quadrature_weight[q]
        :param nodal_coordinates: nodal coordinates of the elements
        :param q: quadrature point for indexing
        :return: JxW at quadrature point q
        """
        J_q = self.calculate_jacobian_map(nodal_coordinates, q)
        det_J_q = jnp.linalg.det(J_q)
        return self.w[q] * det_J_q

    def map_shape_function_gradients(self, nodal_coordinates: jnp.ndarray, q: int) -> jnp.ndarray:
        """
        calculate grad_N_X at quadrature point q
        :param nodal_coordinates: nodal coordinates of the elements
        :param q: quadrature point for indexing
        :return: grad_N_X at quadrature point q
        """
        J_q = self.calculate_jacobian_map(nodal_coordinates, q)
        J_q_inv = jnp.linalg.inv(J_q)
        grad_N_X = jnp.matmul(J_q_inv, self.grad_N_xi[q, :, :].T)
        return grad_N_X

