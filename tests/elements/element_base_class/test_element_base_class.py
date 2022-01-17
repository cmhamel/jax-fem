import pytest
from src.elements.element_base_class import ElementBaseClass


def test_failed_quadrature_order():
    with pytest.raises(AssertionError):
        ElementBaseClass(0, 1)


def test_failed_shape_function_order():
    with pytest.raises(AssertionError):
        ElementBaseClass(1, 0)


def test_assertion_error_in_init_method():
    with pytest.raises(AssertionError):
        ElementBaseClass(1, 1)





