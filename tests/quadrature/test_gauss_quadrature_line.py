import pytest
import jax.numpy as jnp
from src.quadrature.quadrature import Quadrature
from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture
def gauss_quadrature_order_1():
    block = {'quadrature_type': 'gauss_quadrature',
             'quadrature_order': 1}
    return Quadrature(n_dimensions=1, quadrature_block=block, element_type='line')


@pytest.fixture
def gauss_quadrature_order_2():
    block = {'quadrature_type': 'gauss_quadrature',
             'quadrature_order': 2}
    return Quadrature(n_dimensions=1, quadrature_block=block, element_type='line')


def test_gauss_quadrature_order_1_line_integral(gauss_quadrature_order_1):
    quadrature = gauss_quadrature_order_1.quadrature.quadrature
    xi = quadrature.xi
    w = quadrature.w

    assert xi.shape == (1, 1)
    assert w.shape == (1, 1)
    assert jnp.allclose(xi[0, 0], 0.0)
    assert jnp.allclose(w[0, 0], 2.0)

    integrand = jnp.zeros(1, dtype=jnp.float64)
    for q in range(gauss_quadrature_order_1.quadrature.quadrature.n_cell_quadrature_points):
        integrand = integrand + w[q, 0]

    assert jnp.allclose(integrand, 2.0 * jnp.ones(1, dtype=jnp.float64))


def test_gauss_quadrature_order_2_line_integral(gauss_quadrature_order_2):
    quadrature = gauss_quadrature_order_2.quadrature.quadrature
    xi = quadrature.xi
    w = quadrature.w

    assert xi.shape == (2, 1)
    assert w.shape == (2, 1)
    assert jnp.allclose(xi[0, 0], -jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(xi[1, 0], jnp.sqrt(1.0 / 3.0))
    assert jnp.allclose(w[0, 0], 1.0)
    assert jnp.allclose(w[1, 0], 1.0)

    integrand = jnp.zeros(1, dtype=jnp.float64)
    for q in range(gauss_quadrature_order_2.quadrature.quadrature.n_cell_quadrature_points):
        integrand = integrand + w[q, 0]

    assert jnp.allclose(integrand, 2.0 * jnp.ones(1, dtype=jnp.float64))
