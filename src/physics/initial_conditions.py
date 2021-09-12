import jax
import jax.numpy as jnp
import random
from jax import random


class InitialCondition:
    def __init__(self, ic_type, block_nodes,
                 value=None,
                 function=None,
                 spatial_function=None):
        self.ic_type = ic_type
        self.block_nodes = block_nodes
        self.value = value
        self.function = function
        self.spatial_function = None

        if ic_type == 'constant':
            assert self.value is not None
            self.values = self.value * jnp.zeros(self.block_nodes.shape[0], jnp.float64)
        elif ic_type == 'function':
            assert self.function is not None
            # self.values = jnp.zeros(self.block_nodes.shape[0], dtype=jnp.float64)
            #
            # def loop_function(n, temp_values):
            #     temp_values = jax.ops.index_update(temp_values, jax.ops.index[n],
            #                                        0.63 + 0.02 * (0.5 - random.random()))
            #     return temp_values
            # self.values = jax.lax.fori_loop(0, self.block_nodes.shape[0], loop_function, self.values)
            # self.values = random.uniform(random.PRNGKey(128),
            #                              shape=(self.block_nodes.shape[0], ),
            #                              minval=0, maxval=1)
            self.values = self.function()
        elif ic_type == 'spatial_function':
            assert False, 'not supported yet'


