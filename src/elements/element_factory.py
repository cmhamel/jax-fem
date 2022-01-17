from .element_base_class import ElementBaseClass


def element_factory(element_input_settings: dict) -> ElementBaseClass:
    element_type = element_input_settings['element_type']
    if element_type == 'line_element':
        from .line_element import LineElement
        return LineElement(element_input_settings)
    else:
        assert False
