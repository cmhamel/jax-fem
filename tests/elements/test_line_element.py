import pytest
import jax.numpy as jnp
from src.elements import LineElement
from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture
def line_element_order_q1_p1():
    return LineElement(quadrature_order=1, shape_function_order=1)


@pytest.fixture
def line_element_order_q2_p1():
    return LineElement(quadrature_order=2, shape_function_order=1)


@pytest.mark.xfail(raises=AssertionError)
def test_bad_quadrature_input():
    e = LineElement(quadrature_order=-1, shape_function_order=1)


@pytest.mark.xfail(raises=Exception)
def test_unsupported_quadrature():
    e = LineElement(quadrature_order=10, shape_function_order=1)


@pytest.mark.xfail(raises=AssertionError)
def test_bad_shape_function_input():
    e = LineElement(quadrature_order=1, shape_function_order=-1)


@pytest.mark.xfail(raises=Exception)
def test_unsupported_shape_function():
    e = LineElement(quadrature_order=1, shape_function_order=10)


def test_quadrature(line_element_order_q1_p1,
                    line_element_order_q2_p1):

    # q1 p1 element
    #
    assert line_element_order_q1_p1.xi.shape == (1, 1)
    assert line_element_order_q1_p1.w.shape == (1, 1)
    assert jnp.allclose(line_element_order_q1_p1.xi[0, 0], 0.0)
    assert jnp.allclose(line_element_order_q1_p1.w[0, 0], 2.0)

    # q2 p1 element
    #
    assert line_element_order_q2_p1.xi.shape == (2, 1)
    assert line_element_order_q2_p1.w.shape == (2, 1)
    assert jnp.allclose(line_element_order_q2_p1.xi[0, 0], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(line_element_order_q2_p1.xi[1, 0], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(line_element_order_q2_p1.w[0, 0], 1.0)
    assert jnp.allclose(line_element_order_q2_p1.w[1, 0], 1.0)


def test_shape_function_values(line_element_order_q1_p1,
                               line_element_order_q2_p1):

    # p1 q1 element
    #
    N_xi = line_element_order_q1_p1.N_xi
    assert N_xi.shape == (1, 2, 1)
    assert jnp.allclose(N_xi[0, 0, 0], 0.5)
    assert jnp.allclose(N_xi[0, 1, 0], 0.5)

    # p1 q2 element
    #
    N_xi = line_element_order_q2_p1.N_xi
    assert N_xi.shape == (2, 2, 1)
    assert jnp.allclose(N_xi[0, 0, 0], 0.5 * (1.0 + jnp.sqrt(1.0 / 3.0)))
    assert jnp.allclose(N_xi[0, 1, 0], 0.5 * (1.0 - jnp.sqrt(1.0 / 3.0)))
    assert jnp.allclose(N_xi[1, 0, 0], 0.5 * (1.0 - jnp.sqrt(1.0 / 3.0)))
    assert jnp.allclose(N_xi[1, 1, 0], 0.5 * (1.0 + jnp.sqrt(1.0 / 3.0)))


def test_shape_function_gradients(line_element_order_q1_p1,
                                  line_element_order_q2_p1):

    # p1 q1 element
    #
    grad_N_xi = line_element_order_q1_p1.grad_N_xi
    assert grad_N_xi.shape == (1, 2, 1)
    assert jnp.allclose(grad_N_xi[0, 0, 0], -0.5)
    assert jnp.allclose(grad_N_xi[0, 1, 0], 0.5)

    # p1 q2 element
    #
    grad_N_xi = line_element_order_q2_p1.grad_N_xi
    assert grad_N_xi.shape == (2, 2, 1)
    assert jnp.allclose(grad_N_xi[0, 0, 0], -0.5)
    assert jnp.allclose(grad_N_xi[0, 1, 0], 0.5)
    assert jnp.allclose(grad_N_xi[1, 0, 0], -0.5)
    assert jnp.allclose(grad_N_xi[1, 1, 0], 0.5)
