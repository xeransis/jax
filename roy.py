from functools import partial
import jax.numpy as jnp
from jax import pmap, grad
from jax import lax

from jax.config import config
config.enable_omnistaging()

def f(x):
  print(x)
  return jnp.sin(x) ** 2 + jnp.cos(x) ** 2
out = pmap(f, axis_name='i')(jnp.arange(4))
print(out)


@partial(pmap, axis_name='i')
@partial(pmap, axis_name='j')
def g(x):
  print(x)
  return jnp.sin(x) ** 2 + jnp.cos(x) ** 2
x = jnp.arange(8).reshape((4, 2))
out = g(x)
print(out)


@partial(pmap, axis_name='i')
@partial(pmap, axis_name='j')
def h(x):
  print(x)
  return lax.psum(jnp.sin(x) ** 2 + jnp.cos(x) ** 2, 'i')
x = jnp.arange(8.).reshape((4, 2))
out = h(x)
print(out)

out = grad(lambda x: h(x).sum())(x)
print(out)
