import jax.numpy as jnp


class Source:
    def __init__(self, shape, source_type, value):
        if source_type == 'constant':
            self.values = value * jnp.ones(shape, dtype=jnp.float64)
        else:
            try:
                assert False
            except AssertionError:
                raise Exception('Unsupported source type in Source')
