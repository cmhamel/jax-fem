from .element_base_class import ElementBaseClass


def element_factory(element_input_settings: dict) -> ElementBaseClass:
    element_type = element_input_settings['element_type']
    if element_type == 'line_element':
        from .line_element import LineElement
        return LineElement(element_input_settings)
    elif element_type == 'quad_element':
        from .quad_element import QuadElement
        return QuadElement(element_input_settings)
    elif element_type == 'hex_element':
        from .hex_element import HexElement
        return HexElement(element_input_settings)
    else:
        assert False
