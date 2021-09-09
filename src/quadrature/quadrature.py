from .gauss_quadrature import GaussQuadrature


class Quadrature:
    def __init__(self, n_dimensions, quadrature_block, element_type):
        self.n_dimensions = n_dimensions
        self.quadrature_block = quadrature_block

        self.quadrature_type = quadrature_block['quadrature_type'].lower()
        self.quadrature_order = quadrature_block['quadrature_order']
        self.element_type = element_type

        if self.quadrature_type == 'gauss_quadrature':
            if self.element_type == 'line':
                self.quadrature = GaussQuadrature(self.n_dimensions,
                                                  self.quadrature_order,
                                                  self.element_type)
            elif self.element_type == 'quad':
                assert False
            else:
                assert False, 'Unsupported element type in Quadrature'
        else:
            assert False, 'Unsupported shape functions'

    def __del__(self):
        pass
