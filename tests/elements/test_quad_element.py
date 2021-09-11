import pytest
import jax.numpy as jnp
from src.elements import QuadElement
from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture
def quad_element_order_q1_p1():
    return QuadElement(quadrature_order=1, shape_function_order=1)


@pytest.fixture
def quad_element_order_q2_p1():
    return QuadElement(quadrature_order=2, shape_function_order=1)


@pytest.mark.xfail(raises=AssertionError)
def test_bad_quadrature_input():
    e = QuadElement(quadrature_order=-1, shape_function_order=1)


@pytest.mark.xfail(raises=Exception)
def test_unsupported_quadrature():
    e = QuadElement(quadrature_order=10, shape_function_order=1)


@pytest.mark.xfail(raises=AssertionError)
def test_bad_shape_function_input():
    e = QuadElement(quadrature_order=1, shape_function_order=-1)


@pytest.mark.xfail(raises=Exception)
def test_unsupported_shape_function():
    e = QuadElement(quadrature_order=1, shape_function_order=10)


def test_quadrature(quad_element_order_q1_p1,
                    quad_element_order_q2_p1):

    # q1 p1 element
    #
    assert quad_element_order_q1_p1.xi.shape == (1, 2)
    assert quad_element_order_q1_p1.w.shape == (1, 1)
    assert jnp.allclose(quad_element_order_q1_p1.xi[0, 0], 0.0)
    assert jnp.allclose(quad_element_order_q1_p1.xi[0, 1], 0.0)
    assert jnp.allclose(quad_element_order_q1_p1.w[0, 0], 4.0)

    # q2 p1 element
    #
    assert quad_element_order_q2_p1.xi.shape == (4, 2)
    assert quad_element_order_q2_p1.w.shape == (4, 1)
    assert jnp.allclose(quad_element_order_q2_p1.xi[0, 0], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[0, 1], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[1, 0], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[1, 1], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[2, 0], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[2, 1], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[3, 0], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.xi[3, 1], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(quad_element_order_q2_p1.w[0, 0], 1.0)
    assert jnp.allclose(quad_element_order_q2_p1.w[1, 0], 1.0)
    assert jnp.allclose(quad_element_order_q2_p1.w[2, 0], 1.0)
    assert jnp.allclose(quad_element_order_q2_p1.w[3, 0], 1.0)


def test_shape_function_values(quad_element_order_q1_p1,
                               quad_element_order_q2_p1):

    # p1 q1 element
    #
    N_xi = quad_element_order_q1_p1.N_xi
    xi, w = quad_element_order_q1_p1.xi, quad_element_order_q1_p1.w
    assert N_xi.shape == (1, 4, 1)
    assert jnp.allclose(N_xi[0, 0, 0], 0.25 * (1.0 - xi[0, 0]) * (1.0 - xi[0, 1]))
    assert jnp.allclose(N_xi[0, 1, 0], 0.25 * (1.0 + xi[0, 0]) * (1.0 - xi[0, 1]))
    assert jnp.allclose(N_xi[0, 2, 0], 0.25 * (1.0 + xi[0, 0]) * (1.0 + xi[0, 1]))
    assert jnp.allclose(N_xi[0, 3, 0], 0.25 * (1.0 - xi[0, 0]) * (1.0 + xi[0, 1]))

    # p1 q2 element
    #
    N_xi = quad_element_order_q2_p1.N_xi
    xi, w = quad_element_order_q2_p1.xi, quad_element_order_q2_p1.w
    assert N_xi.shape == (4, 4, 1)
    print(N_xi[0, 0, 0])
    print(0.25 * (1.0 - xi[0, 0]) * (1.0 - xi[0, 1]))
    assert jnp.allclose(N_xi[0, 0, 0], 0.25 * (1.0 - xi[0, 0]) * (1.0 - xi[0, 1]))
    assert jnp.allclose(N_xi[0, 1, 0], 0.25 * (1.0 + xi[0, 0]) * (1.0 - xi[0, 1]))
    assert jnp.allclose(N_xi[0, 2, 0], 0.25 * (1.0 + xi[0, 0]) * (1.0 + xi[0, 1]))
    assert jnp.allclose(N_xi[0, 3, 0], 0.25 * (1.0 - xi[0, 0]) * (1.0 + xi[0, 1]))
    #
    assert jnp.allclose(N_xi[1, 0, 0], 0.25 * (1.0 - xi[1, 0]) * (1.0 - xi[1, 1]))
    assert jnp.allclose(N_xi[1, 1, 0], 0.25 * (1.0 + xi[1, 0]) * (1.0 - xi[1, 1]))
    assert jnp.allclose(N_xi[1, 2, 0], 0.25 * (1.0 + xi[1, 0]) * (1.0 + xi[1, 1]))
    assert jnp.allclose(N_xi[1, 3, 0], 0.25 * (1.0 - xi[1, 0]) * (1.0 + xi[1, 1]))
    #
    assert jnp.allclose(N_xi[2, 0, 0], 0.25 * (1.0 - xi[2, 0]) * (1.0 - xi[2, 1]))
    assert jnp.allclose(N_xi[2, 1, 0], 0.25 * (1.0 + xi[2, 0]) * (1.0 - xi[2, 1]))
    assert jnp.allclose(N_xi[2, 2, 0], 0.25 * (1.0 + xi[2, 0]) * (1.0 + xi[2, 1]))
    assert jnp.allclose(N_xi[2, 3, 0], 0.25 * (1.0 - xi[2, 0]) * (1.0 + xi[2, 1]))
    #
    assert jnp.allclose(N_xi[3, 0, 0], 0.25 * (1.0 - xi[3, 0]) * (1.0 - xi[3, 1]))
    assert jnp.allclose(N_xi[3, 1, 0], 0.25 * (1.0 + xi[3, 0]) * (1.0 - xi[3, 1]))
    assert jnp.allclose(N_xi[3, 2, 0], 0.25 * (1.0 + xi[3, 0]) * (1.0 + xi[3, 1]))
    assert jnp.allclose(N_xi[3, 3, 0], 0.25 * (1.0 - xi[3, 0]) * (1.0 + xi[3, 1]))


def test_shape_function_gradients(quad_element_order_q1_p1,
                                  quad_element_order_q2_p1):

    # p1 q1 element
    #
    grad_N_xi = quad_element_order_q1_p1.grad_N_xi
    assert grad_N_xi.shape == (1, 4, 2)
    assert jnp.allclose(grad_N_xi[0, 0, 0], -0.25)
    assert jnp.allclose(grad_N_xi[0, 0, 1], -0.25)
    assert jnp.allclose(grad_N_xi[0, 1, 0], 0.25)
    assert jnp.allclose(grad_N_xi[0, 1, 1], -0.25)
    assert jnp.allclose(grad_N_xi[0, 2, 0], 0.25)
    assert jnp.allclose(grad_N_xi[0, 2, 1], 0.25)
    assert jnp.allclose(grad_N_xi[0, 3, 0], -0.25)
    assert jnp.allclose(grad_N_xi[0, 3, 1], 0.25)

    # p1 q2 element
    #
    xi, w = quad_element_order_q2_p1.xi, quad_element_order_q2_p1.w
    grad_N_xi = quad_element_order_q2_p1.grad_N_xi
    assert grad_N_xi.shape == (4, 4, 2)
    assert jnp.allclose(grad_N_xi[0, 0, 0], -0.25 * (1.0 - xi[0, 1]))
    assert jnp.allclose(grad_N_xi[0, 0, 1], -0.25 * (1.0 - xi[0, 0]))
    assert jnp.allclose(grad_N_xi[0, 1, 0], 0.25 * (1.0 - xi[0, 1]))
    assert jnp.allclose(grad_N_xi[0, 1, 1], -0.25 * (1.0 + xi[0, 0]))
    assert jnp.allclose(grad_N_xi[0, 2, 0], 0.25 * (1.0 + xi[0, 1]))
    assert jnp.allclose(grad_N_xi[0, 2, 1], 0.25 * (1.0 + xi[0, 0]))
    assert jnp.allclose(grad_N_xi[0, 3, 0], -0.25 * (1.0 + xi[0, 1]))
    assert jnp.allclose(grad_N_xi[0, 3, 1], 0.25 * (1.0 - xi[0, 0]))
    #
    assert jnp.allclose(grad_N_xi[1, 0, 0], -0.25 * (1.0 - xi[1, 1]))
    assert jnp.allclose(grad_N_xi[1, 0, 1], -0.25 * (1.0 - xi[1, 0]))
    assert jnp.allclose(grad_N_xi[1, 1, 0], 0.25 * (1.0 - xi[1, 1]))
    assert jnp.allclose(grad_N_xi[1, 1, 1], -0.25 * (1.0 + xi[1, 0]))
    assert jnp.allclose(grad_N_xi[1, 2, 0], 0.25 * (1.0 + xi[1, 1]))
    assert jnp.allclose(grad_N_xi[1, 2, 1], 0.25 * (1.0 + xi[1, 0]))
    assert jnp.allclose(grad_N_xi[1, 3, 0], -0.25 * (1.0 + xi[1, 1]))
    assert jnp.allclose(grad_N_xi[1, 3, 1], 0.25 * (1.0 - xi[1, 0]))
    #
    assert jnp.allclose(grad_N_xi[2, 0, 0], -0.25 * (1.0 - xi[2, 1]))
    assert jnp.allclose(grad_N_xi[2, 0, 1], -0.25 * (1.0 - xi[2, 0]))
    assert jnp.allclose(grad_N_xi[2, 1, 0], 0.25 * (1.0 - xi[2, 1]))
    assert jnp.allclose(grad_N_xi[2, 1, 1], -0.25 * (1.0 + xi[2, 0]))
    assert jnp.allclose(grad_N_xi[2, 2, 0], 0.25 * (1.0 + xi[2, 1]))
    assert jnp.allclose(grad_N_xi[2, 2, 1], 0.25 * (1.0 + xi[2, 0]))
    assert jnp.allclose(grad_N_xi[2, 3, 0], -0.25 * (1.0 + xi[2, 1]))
    assert jnp.allclose(grad_N_xi[2, 3, 1], 0.25 * (1.0 - xi[2, 0]))
    #
    assert jnp.allclose(grad_N_xi[3, 0, 0], -0.25 * (1.0 - xi[3, 1]))
    assert jnp.allclose(grad_N_xi[3, 0, 1], -0.25 * (1.0 - xi[3, 0]))
    assert jnp.allclose(grad_N_xi[3, 1, 0], 0.25 * (1.0 - xi[3, 1]))
    assert jnp.allclose(grad_N_xi[3, 1, 1], -0.25 * (1.0 + xi[3, 0]))
    assert jnp.allclose(grad_N_xi[3, 2, 0], 0.25 * (1.0 + xi[3, 1]))
    assert jnp.allclose(grad_N_xi[3, 2, 1], 0.25 * (1.0 + xi[3, 0]))
    assert jnp.allclose(grad_N_xi[3, 3, 0], -0.25 * (1.0 + xi[3, 1]))
    assert jnp.allclose(grad_N_xi[3, 3, 1], 0.25 * (1.0 - xi[3, 0]))