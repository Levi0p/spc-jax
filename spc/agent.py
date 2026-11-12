from functools import partial

import jax
import jax.numpy as jnp
import optax

from .nets import MLP, Ensemble, lerp, normal_log_prob
from .world_model import WorldModel


class SPC:
    """Self-predictive contextual offline meta-RL agent: a context encoder trained through a
    contextual latent world model (+ InfoNCE), and an IQL actor-critic on the model's latents."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.wm = WorldModel(cfg)
        self.ctx_in = 2 * cfg.obs_dim + cfg.action_dim + 1 if cfg.use_next_obs_in_context else cfg.obs_dim + cfg.action_dim + 1
        self.ctx = MLP([cfg.context_hidden_dims] * cfg.context_hidden_depth, cfg.z_dim)
        self.actor = MLP([cfg.actor_hidden_dim] * cfg.actor_hidden_depth, 2 * cfg.action_dim)
        self.qs = Ensemble(cfg.num_critics, [cfg.q_hidden_dim] * cfg.q_hidden_depth, 1)
        self.v = MLP([cfg.q_hidden_dim] * cfg.q_hidden_depth, 1)
        self.opt = dict(
            model=optax.adamw(cfg.model_lr, weight_decay=0.01),
            ctx=optax.adam(cfg.context_enc_lr),
            actor=optax.adam(cfg.actor_lr),
            qs=optax.adam(cfg.critic_lr),
            v=optax.adam(cfg.critic_lr),
        )

    def init(self, key):
        cfg, L = self.cfg, self.wm.latent_dim
        k = jax.random.split(key, 5)
        lz = jnp.zeros((1, L + cfg.z_dim))
        params = dict(
            model=self.wm.init(k[0]),
            ctx=self.ctx.init(k[1], jnp.zeros((1, self.ctx_in)))['params'],
            actor=self.actor.init(k[2], lz)['params'],
            qs=self.qs.init(k[3], jnp.zeros((1, L + cfg.action_dim + cfg.z_dim)))['params'],
            v=self.v.init(k[4], lz)['params'],
        )
        return dict(
            params=params,
            target=dict(enc=params['model']['enc'], qs=params['qs']),
            opt={name: o.init(params[name]) for name, o in self.opt.items()},
            history=jnp.zeros((cfg.num_train_tasks, cfg.z_dim)),
        )

    def _step(self, state, name, grads):
        updates, state['opt'][name] = self.opt[name].update(grads, state['opt'][name], state['params'][name])
        state['params'][name] = optax.apply_updates(state['params'][name], updates)

    # context

    def encode_context(self, ctx_params, transitions):
        return jnp.tanh(self.ctx.apply({'params': ctx_params}, transitions))

    def infer_z(self, ctx_params, context):
        """context: (tasks, n, ctx_in) -> (tasks, z_dim), mean over transitions."""
        return self.encode_context(ctx_params, context).mean(1)

    def infonce(self, means, task_ids, history):
        cfg = self.cfg
        counts = jnp.zeros(cfg.num_train_tasks).at[task_ids].add(1.0)[:, None]
        sums = jnp.zeros_like(history).at[task_ids].add(means)
        current = jnp.where(counts > 0, sums / jnp.maximum(counts, 1.0), history)
        history = (1 - cfg.infonce_tau) * history + cfg.infonce_tau * jax.lax.stop_gradient(current)
        dist = jnp.sqrt(((current[None] - history[:, None]) ** 2).sum(-1) + 1e-6)
        logits = -jnp.concatenate([jnp.diag(dist)[:, None], dist], 1) / cfg.infonce_radius
        return -jax.nn.log_softmax(logits)[:, 0].mean(), history

    # policy

    def policy(self, actor_params, latent, z):
        mu, logstd = jnp.split(jnp.tanh(self.actor.apply({'params': actor_params}, latent, z)), 2, -1)
        logstd = self.cfg.log_std_min + 0.5 * (self.cfg.log_std_max - self.cfg.log_std_min) * (logstd + 1)
        return mu, jnp.exp(logstd)

    def _pi(self, actor_params, latent, z, key, deterministic=False):
        mu, std = self.policy(actor_params, latent, z)
        a = mu if deterministic else mu + std * jax.random.normal(key, mu.shape)
        return jnp.clip(a, -1, 1)

    @partial(jax.jit, static_argnums=(0, 6, 7))
    def act(self, params, obs, z, key, prev_mean=None, deterministic=False, t0=True):
        latent, _ = self.wm.encode(params['model']['enc'], obs)
        if not self.cfg.mpc:
            return self._pi(params['actor'], latent, z, key, deterministic), prev_mean
        keys = jax.random.split(key, latent.shape[0])
        plan = partial(self._plan, params, t0=t0, deterministic=deterministic)
        a, mean = jax.vmap(plan)(latent, z, prev_mean, keys)
        if self.cfg.mppi_use_mean:
            a = mean[:, 0]
        return jnp.clip(a, -1, 1), mean

    # training

    @partial(jax.jit, static_argnums=(0, 6))
    def update(self, state, batch, context, task_ids, key, update_model):
        """batch: tuple of (tasks, batch, seq_len, dim); context: (tasks, n, ctx_in);
        task_ids: indices into the training tasks, used for the InfoNCE history."""
        cfg = self.cfg
        T, B, S = batch[0].shape[:3]
        obss, actions, rewards, next_obss, dones = (x.transpose(2, 0, 1, 3).reshape(S, T * B, -1) for x in batch)
        info = {}

        def task_z(ctx_params):
            return jnp.repeat(self.infer_z(ctx_params, context), B, axis=0)

        if update_model:
            def loss_fn(model_params, ctx_params):
                z = task_z(ctx_params)
                zm = z if cfg.allow_z_model else jax.lax.stop_gradient(z)
                tc, rew = self.wm.loss(model_params, state['target']['enc'], obss, actions, rewards, next_obss, dones, zm, key)
                model_loss = cfg.consistency_coef * tc + cfg.reward_coef * rew
                nce, history = self.infonce(z[::B], task_ids, state['history']) if cfg.use_infonce else (0.0, state['history'])
                return model_loss + cfg.infonce_weight * nce, (z, tc, rew, model_loss, nce, history)

            (_, (z, tc, rew, model_loss, nce, history)), (g_model, g_ctx) = jax.value_and_grad(loss_fn, (0, 1), has_aux=True)(
                state['params']['model'], state['params']['ctx'])
            self._step(state, 'model', g_model)
            state['target']['enc'] = lerp(state['target']['enc'], state['params']['model']['enc'], cfg.tau)
            self._step(state, 'ctx', g_ctx)
            state['history'] = history
            info.update(tc_loss=tc, reward_loss=rew, total_loss=model_loss, infonce_loss=nce)
        else:
            z = task_z(state['params']['ctx'])

        z = jnp.broadcast_to(z, (S, T * B, cfg.z_dim)).reshape(S * T * B, -1)
        obss, actions, rewards, next_obss, dones = (x.reshape(S * T * B, -1) for x in (obss, actions, rewards, next_obss, dones))
        enc = state['params']['model']['enc']
        latents = jax.lax.stop_gradient(self.wm.encode(enc, obss)[0])
        next_latents = jax.lax.stop_gradient(self.wm.encode(enc, next_obss)[0])
        info.update(self._update_actor_critic(state, latents, actions, rewards, next_latents, dones, z))
        state['target']['qs'] = lerp(state['target']['qs'], state['params']['qs'], cfg.tau)
        return state, info

    def _update_actor_critic(self, state, latents, actions, rewards, next_latents, dones, z):
        cfg, p = self.cfg, state['params']
        q = self.qs.apply({'params': state['target']['qs']}, latents, actions, z).min(0)

        def v_loss(v_params):
            diff = q - self.v.apply({'params': v_params}, latents, z)
            return (jnp.where(diff > 0, cfg.expectile, 1 - cfg.expectile) * diff ** 2).mean()

        v_old = self.v.apply({'params': p['v']}, latents, z)
        exp_a = jnp.minimum(jnp.exp((q - v_old) * cfg.temperature), 100.0)
        vl, g = jax.value_and_grad(v_loss)(p['v'])
        self._step(state, 'v', g)

        next_v = self.v.apply({'params': state['params']['v']}, next_latents, z)
        target_q = rewards + cfg.discount * (1 - dones) * next_v

        def q_loss(qs_params):
            return ((self.qs.apply({'params': qs_params}, latents, actions, z) - target_q) ** 2).mean()

        ql, g = jax.value_and_grad(q_loss)(p['qs'])
        self._step(state, 'qs', g)

        def pi_loss(actor_params):
            mu, std = self.policy(actor_params, latents, z)
            return -(exp_a * normal_log_prob(actions, mu, std).sum(-1, keepdims=True)).mean()

        pl, g = jax.value_and_grad(pi_loss)(p['actor'])
        self._step(state, 'actor', g)
        return dict(critic_loss=ql, average_q=q.mean(), v_loss=vl, policy_loss=pl)

    # planning (MPPI, as in TD-MPC2)

    def _value(self, params, latent, z, actions, key):
        cfg = self.cfg
        G, discount = 0.0, 1.0
        for t, k in enumerate(jax.random.split(key, cfg.plan_horizon)):
            r = self.wm.reward(params['model'], latent, actions[t], z, cfg.reward_mode, cfg.reward_penalty)
            latent, _ = self.wm.trans(params['model'], latent, actions[t], z, k, cfg.plan_unc_prop_mode)
            G = G + discount * r
            discount *= cfg.rho
        if cfg.plan_with_value:
            G = G + discount * self.v.apply({'params': params['v']}, latent, z)
        return G

    def _plan(self, params, latent, z, prev_mean, key, t0, deterministic):
        cfg = self.cfg
        H, N, P, A = cfg.plan_horizon, cfg.num_samples, cfg.num_pi_trajs, cfg.action_dim
        key, *ks = jax.random.split(key, 2 * H + 1)
        pi_actions = []
        if P > 0:
            lat, zz = jnp.broadcast_to(latent, (P, latent.shape[-1])), jnp.broadcast_to(z, (P, z.shape[-1]))
            for t in range(H):
                pi_actions.append(self._pi(params['actor'], lat, zz, ks[2 * t]))
                if t < H - 1:
                    lat, _ = self.wm.trans(params['model'], lat, pi_actions[-1], zz, ks[2 * t + 1], cfg.plan_unc_prop_mode)
            pi_actions = jnp.stack(pi_actions)

        latent, z = jnp.broadcast_to(latent, (N, latent.shape[-1])), jnp.broadcast_to(z, (N, z.shape[-1]))
        mean = jnp.zeros((H, A)) if t0 else jnp.zeros((H, A)).at[:-1].set(prev_mean[1:])
        std = jnp.full((H, A), cfg.max_std)
        for _ in range(cfg.iterations):
            key, k1, k2 = jax.random.split(key, 3)
            noise = jax.random.normal(k1, (H, N - P, A))
            actions = jnp.clip(mean[:, None] + std[:, None] * noise, -1, 1)
            if P > 0:
                actions = jnp.concatenate([pi_actions, actions], 1)
            value = jnp.nan_to_num(self._value(params, latent, z, actions, k2), nan=0.0)[:, 0]
            elite_value, elite_idx = jax.lax.top_k(value, cfg.num_elites)
            elite_actions = actions[:, elite_idx]
            score = jnp.exp(cfg.plan_temperature * (elite_value - elite_value.max()))
            score = score / score.sum()
            mean = (score[None, :, None] * elite_actions).sum(1) / (score.sum() + 1e-9)
            std = jnp.sqrt((score[None, :, None] * (elite_actions - mean[:, None]) ** 2).sum(1) / (score.sum() + 1e-9))
            std = jnp.clip(std, cfg.min_std, cfg.max_std)
        k1, k2 = jax.random.split(key)
        a = elite_actions[0, jax.random.categorical(k1, jnp.log(score))]
        if not deterministic:
            a = a + std[0] * jax.random.normal(k2, (A,))
        return a, mean
