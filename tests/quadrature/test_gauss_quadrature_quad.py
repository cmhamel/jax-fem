import pytest
import jax.numpy as jnp
from src.quadrature import Quadrature
from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture
def gauss_quadrature_order_1():
    block = {'quadrature_type': 'gauss_quadrature',
             'quadrature_order': 1}
    return Quadrature(n_dimensions=2, quadrature_block=block, element_type='line')


@pytest.fixture
def gauss_quadrature_order_2():
    block = {'quadrature_type': 'gauss_quadrature',
             'quadrature_order': 2}
    return Quadrature(n_dimensions=2, quadrature_block=block, element_type='line')

