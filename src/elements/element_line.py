from .element_base_class import ElementBaseClass
import jax.numpy as jnp
from jax import jit
from jax import partial
import jax


class LineElement(ElementBaseClass):
    """
    Line element for 1D code development and educational purposes
    """
    def __init__(self, quadrature_order, shape_function_order):
        super(LineElement, self).__init__()

        self.quadrature_order = quadrature_order
        self.shape_function_order = shape_function_order

        self.n_quadrature_points = self.quadrature_order
        self.n_nodes = shape_function_order + 1

        assert self.n_quadrature_points > 0
        assert self.n_nodes > 1

        # initialize the arrays that are the same no matter the element distortion
        #
        self.xi, self.w = self._calculate_quadrature()
        self.N_xi = self._calculate_shape_function_values()
        self.grad_N_xi = self._calculate_shape_function_gradients()

    @partial(jit, static_argnums=(0,))
    def _calculate_quadrature(self):
        xi = jnp.zeros((self.n_quadrature_points, 1), dtype=jnp.float64)
        w = jnp.zeros((self.n_quadrature_points, 1), dtype=jnp.float64)
        if self.quadrature_order == 1:
            w = jax.ops.index_update(w, jax.ops.index[0, 0], 2.0)
        elif self.quadrature_order == 2:
            xi = jax.ops.index_update(xi, jax.ops.index[0, 0], -jnp.sqrt(1.0 / 3.0))
            xi = jax.ops.index_update(xi, jax.ops.index[1, 0], jnp.sqrt(1.0 / 3.0))
            w = jax.ops.index_update(w, jax.ops.index[0, 0], 1.0)
            w = jax.ops.index_update(w, jax.ops.index[1, 0], 1.0)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported quadrature order in LineElement')

        return xi, w

    @partial(jit, static_argnums=(0,))
    def _calculate_shape_function_values(self):
        N_xi = jnp.zeros((self.n_quadrature_points, self.n_nodes, 1), dtype=jnp.float64)
        for q in range(self.n_quadrature_points):
            if self.shape_function_order == 1:
                N_xi = jax.ops.index_update(N_xi, jax.ops.index[q, 0, 0], 0.5 * (1.0 - self.xi[q, 0]))
                N_xi = jax.ops.index_update(N_xi, jax.ops.index[q, 1, 0], 0.5 * (1.0 + self.xi[q, 0]))
            else:
                # don't need to check this in shape function gradients since it's already
                # checked here
                #
                try:
                    assert False
                except AssertionError:
                    raise Exception('Unsupported shape function order in LineElement')

        return N_xi

    @partial(jit, static_argnums=(0,))
    def _calculate_shape_function_gradients(self):
        grad_N_xi = jnp.zeros((self.n_quadrature_points, self.n_nodes, 1), dtype=jnp.float64)
        for q in range(self.n_quadrature_points):
            if self.shape_function_order == 1:
                grad_N_xi = jax.ops.index_update(grad_N_xi, jax.ops.index[q, 0, 0], -0.5)
                grad_N_xi = jax.ops.index_update(grad_N_xi, jax.ops.index[q, 1, 0], 0.5)
        return grad_N_xi

    @partial(jit, static_argnums=(0,))
    def calculate_jacobian_map(self, nodal_coordinates):
        element_length = jnp.abs(nodal_coordinates[0, 0] - nodal_coordinates[-1, 0])
        return element_length / 2.0

    @partial(jit, static_argnums=(0,))
    def calculate_deriminant_of_jacobian_map(self, nodal_coordinates):
        element_length = jnp.abs(nodal_coordinates[0, 0] - nodal_coordinates[-1, 0])
        return element_length / 2.0

    @partial(jit, static_argnums=(0,))
    def map_shape_function_gradients(self, nodal_coordinates):
        element_length = jnp.abs(nodal_coordinates[0, 0] - nodal_coordinates[-1, 0])
        # det_J = self.calculate_deriminant_of_jacobian_map(nodal_coordinates)

        grad_N_X = jnp.zeros((self.n_quadrature_points, self.n_nodes, 1), dtype=jnp.float64)
        for q in range(self.n_quadrature_points):
            if self.shape_function_order == 1:
                grad_N_X = jax.ops.index_update(grad_N_X, jax.ops.index[q, 0, 0], -1.0 / element_length)
                grad_N_X = jax.ops.index_update(grad_N_X, jax.ops.index[q, 1, 0], 1.0 / element_length)

        return grad_N_X
