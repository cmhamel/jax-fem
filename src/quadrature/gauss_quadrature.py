import jax.numpy as jnp
from .quadrature_base_class import QuadratureBaseClass
from .gauss_quadrature_line import GaussQuadratureLine
from .gauss_quadrature_quad import GaussQuadratureQuad


class GaussQuadrature(QuadratureBaseClass):
    def __init__(self, n_dimensions, quadrature_order, element_type):
        super(GaussQuadrature, self).__init__(n_dimensions)

        self.quadrature_order = quadrature_order
        self.element_type = element_type.lower()

        assert self.quadrature_order > 0
        assert self.element_type in ['line', 'tri', 'quad', 'hex', 'tet']

        if self.element_type == 'line':
            self.quadrature = GaussQuadratureLine(self.quadrature_order)
        elif self.element_type == 'tri':
            pass
        elif self.element_type == 'quad':
            self.quadrature = GaussQuadratureQuad(self.quadrature_order)
        elif self.element_type == 'hex':
            pass
        elif self.element_type == 'tet':
            pass
        else:
            assert False, 'Unsupported element type given to GaussQuadrature'


