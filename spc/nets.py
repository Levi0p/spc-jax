from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

init = nn.initializers.normal(0.02)


class MLP(nn.Module):
    """TD-MPC2 style MLP: [Linear -> LayerNorm -> Mish] x depth -> Linear."""
    hidden: Sequence[int]
    out: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, *xs, train=False):
        x = jnp.concatenate(xs, -1)
        for i, h in enumerate(self.hidden):
            x = nn.Dense(h, kernel_init=init, name=f'fc{i}')(x)
            if i == 0 and self.dropout > 0:
                x = nn.Dropout(self.dropout, deterministic=not train)(x)
            x = nn.LayerNorm(epsilon=1e-5, name=f'ln{i}')(x)
            x = jax.nn.mish(x)
        return nn.Dense(self.out, kernel_init=init, name=f'fc{len(self.hidden)}')(x)


class Ensemble(nn.Module):
    """`num` independent MLPs evaluated in parallel; output has a leading ensemble axis."""
    num: int
    hidden: Sequence[int]
    out: int

    @nn.compact
    def __call__(self, *xs):
        vmapped = nn.vmap(MLP, variable_axes={'params': 0}, split_rngs={'params': True},
                          in_axes=None, axis_size=self.num)
        return vmapped(self.hidden, self.out, name='mlp')(*xs)


def simnorm(x, dim):
    shp = x.shape
    return jax.nn.softmax(x.reshape(*shp[:-1], -1, dim), -1).reshape(shp)


class FSQ:
    """Finite scalar quantization, matching vector_quantize_pytorch.FSQ without projections."""

    def __init__(self, levels):
        self.levels = np.asarray(levels)
        self.channels = len(levels)
        self.codebook_size = int(np.prod(levels))
        self.basis = np.cumprod([1] + list(levels[:-1]))
        self.half_width = self.levels // 2
        idx = np.arange(self.codebook_size)[:, None]
        self.codebook = jnp.asarray(((idx // self.basis) % self.levels - self.half_width) / self.half_width,
                                    dtype=jnp.float32)

    def __call__(self, z):
        z = z.reshape(*z.shape[:-1], -1, self.channels)
        half_l = (self.levels - 1) * (1 + 1e-3) / 2
        offset = np.where(self.levels % 2 == 0, 0.5, 0.0)
        bounded = jnp.tanh(z + np.arctanh(offset / half_l)) * half_l - offset
        codes = (bounded + jax.lax.stop_gradient(jnp.round(bounded) - bounded)) / self.half_width
        indices = jnp.round(((codes * self.half_width + self.half_width) * self.basis).sum(-1)).astype(jnp.int32)
        return codes.reshape(*codes.shape[:-2], -1), indices


def lerp(target, online, tau):
    return jax.tree.map(lambda t, o: t + tau * (o - t), target, online)


def normal_log_prob(x, mu, std):
    return -((x - mu) ** 2) / (2 * std ** 2) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
