import numpy as np
import jax.numpy as jnp
from .quadrature_base_class import QuadratureBaseClass


class GaussQuadratureQuad:
    def __init__(self, quadrature_order):
        self.quadrature_order = quadrature_order
        self.n_cell_quadrature_points = self.quadrature_order**2
        self.xi = jnp.zeros((self.n_cell_quadrature_points, 2), dtype=jnp.float64)
        self.w = jnp.zeros((self.n_cell_quadrature_points, 1), dtype=jnp.float64)

        if self.quadrature_order == 1:
            self.w = self.w.at[0, 0].set(4.0)
        elif self.quadrature_order == 2:
            sqrt_third = np.sqrt(1.0 / 3.0)
            self.xi = self.xi.at[0, 0].set(-sqrt_third)
            self.xi = self.xi.at[0, 1].set(-sqrt_third)
            self.xi = self.xi.at[1, 0].set(sqrt_third)
            self.xi = self.xi.at[1, 1].set(-sqrt_third)
            self.xi = self.xi.at[2, 0].set(sqrt_third)
            self.xi = self.xi.at[2, 1].set(sqrt_third)
            self.xi = self.xi.at[3, 0].set(-sqrt_third)
            self.xi = self.xi.at[3, 1].set(sqrt_third)

            self.w = self.w.at[0, 0].set(1.0)
            self.w = self.w.at[1, 0].set(1.0)
            self.w = self.w.at[2, 0].set(1.0)
            self.w = self.w.at[3, 0].set(1.0)
        else:
            assert False, 'Unsupported quadrature order'

    def __str__(self):
        s = 'First order Gauss quadrature'
        s = s + 'Number of cell quadrature points = %s' % self.n_cell_quadrature_points
        s = s + 'xi                               = %s ' % self.xi
        s = s + 'w                                = %s ' % self.w

        return s