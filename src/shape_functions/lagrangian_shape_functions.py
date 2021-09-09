from .shape_functions_base_class import ShapeFunctionsBaseClass


class LagrangianShapeFunctions(ShapeFunctionsBaseClass):
    def __init__(self, n_dimensions, shape_function_order, element_type):
        super(LagrangianShapeFunctions, self).__init__(n_dimensions)
        self.shape_function_order = shape_function_order
        self.element_type = element_type
