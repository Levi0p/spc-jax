from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .nets import MLP, lerp, normal_log_prob


class DroQ:
    """SAC with dropout + LayerNorm critics, used to collect the per-task offline datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        hidden = [cfg.hidden_dim] * cfg.hidden_depth
        self.actor = MLP(hidden, 2 * cfg.action_dim)
        self.q = MLP(hidden, 1, dropout=cfg.dropout)
        self.opt = dict(actor=optax.adam(cfg.actor_lr), qs=optax.adam(cfg.critic_lr), log_alpha=optax.adam(cfg.alpha_lr))
        self.target_entropy = float(cfg.target_entropy)

    def init(self, key):
        cfg = self.cfg
        k = jax.random.split(key, 5)
        obs, sa = jnp.zeros((1, cfg.obs_dim)), jnp.zeros((1, cfg.obs_dim + cfg.action_dim))
        params = dict(
            actor=self.actor.init(k[0], obs)['params'],
            qs=[self.q.init(k[1], sa)['params'], self.q.init(k[2], sa)['params']],
            log_alpha=jnp.log(cfg.init_alpha if cfg.tune_alpha else cfg.alpha),
        )
        target = [self.q.init(k[3], sa)['params'], self.q.init(k[4], sa)['params']]
        return dict(params=params, target=target, opt={name: o.init(params[name]) for name, o in self.opt.items()})

    def _q(self, qs_params, obs, a, key):
        k1, k2 = jax.random.split(key)
        return [self.q.apply({'params': p}, obs, a, train=True, rngs={'dropout': k}) for p, k in zip(qs_params, (k1, k2))]

    def _sample(self, actor_params, obs, key):
        mu, logstd = jnp.split(self.actor.apply({'params': actor_params}, obs), 2, -1)
        logstd = self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (jnp.tanh(logstd) + 1)
        std = jnp.exp(logstd)
        x = mu + std * jax.random.normal(key, mu.shape)
        log_prob = normal_log_prob(x, mu, std) - 2 * (jnp.log(2.0) - x - jax.nn.softplus(-2 * x))
        return jnp.tanh(x), jnp.tanh(mu), log_prob.sum(-1, keepdims=True)

    def _step(self, state, name, grads):
        updates, state['opt'][name] = self.opt[name].update(grads, state['opt'][name], state['params'][name])
        state['params'][name] = optax.apply_updates(state['params'][name], updates)

    @partial(jax.jit, static_argnums=0)
    def update(self, state, obs, actions, rewards, next_obs, terminals, key):
        cfg = self.cfg
        k = jax.random.split(key, 6)
        p = state['params']
        alpha = jnp.exp(p['log_alpha'])

        next_a, _, next_logp = self._sample(p['actor'], next_obs, k[0])
        next_q = jnp.minimum(*self._q(state['target'], next_obs, next_a, k[1])) - alpha * next_logp
        target_q = jax.lax.stop_gradient(rewards + cfg.discount * (1 - terminals) * next_q)

        def critic_loss(qs_params):
            q1, q2 = self._q(qs_params, obs, actions, k[2])
            return ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean(), (q1.mean(), q2.mean())

        (cl, (q1, q2)), g = jax.value_and_grad(critic_loss, has_aux=True)(p['qs'])
        self._step(state, 'qs', g)

        def actor_loss(actor_params):
            a, _, logp = self._sample(actor_params, obs, k[3])
            q = jnp.minimum(*self._q(state['params']['qs'], obs, a, k[4]))
            return -q.mean() + alpha * logp.mean(), logp

        (al, logp), g = jax.value_and_grad(actor_loss, has_aux=True)(p['actor'])
        self._step(state, 'actor', g)
        info = dict(actor_loss=al, critic_loss=cl, average_q1=q1, average_q2=q2, entropy=-logp.mean())

        if cfg.tune_alpha:
            _, _, logp = self._sample(state['params']['actor'], obs, k[5])
            logp = jax.lax.stop_gradient(logp)
            alpha_loss_fn = lambda log_alpha: (-jnp.exp(log_alpha) * (logp + self.target_entropy)).mean()
            alpha_loss, g = jax.value_and_grad(alpha_loss_fn)(p['log_alpha'])
            self._step(state, 'log_alpha', g)
            info.update(alpha_loss=alpha_loss, alpha=alpha)

        state['target'] = lerp(state['target'], state['params']['qs'], cfg.tau)
        return state, info

    @partial(jax.jit, static_argnums=(0, 4))
    def _act(self, actor_params, obs, key, deterministic):
        a, mean, _ = self._sample(actor_params, obs, key)
        return mean if deterministic else a

    def act(self, state, obs, key, deterministic=False):
        return np.asarray(self._act(state['params']['actor'], jnp.asarray(obs, jnp.float32), key, deterministic))
