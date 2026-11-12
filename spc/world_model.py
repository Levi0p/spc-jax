import jax
import jax.numpy as jnp
import optax

from .nets import MLP, Ensemble, FSQ, simnorm

KINDS = ('continuous', 'simnorm', 'discrete')
LOSSES = ('mse', 'cosine', 'ce')


class WorldModel:
    """Contextual latent world model. `cfg.world_model` is '<kind>_<loss>', e.g. 'discrete_ce'.

    continuous/simnorm/discrete set how observations are embedded; mse/cosine regress the next
    latent with a residual transition, ce predicts a categorical over FSQ codes per latent group.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.kind, self.loss_type = cfg.world_model.split('_')
        assert self.kind in KINDS and self.loss_type in LOSSES
        assert self.loss_type != 'ce' or self.kind == 'discrete', 'ce needs discrete latents'
        self.fsq = FSQ(cfg.fsq_levels) if self.kind == 'discrete' else None
        if self.loss_type == 'ce':
            self.groups = cfg.latent_dim
            self.latent_dim = cfg.latent_dim * self.fsq.channels
            trans_out = cfg.latent_dim * self.fsq.codebook_size
        else:
            self.latent_dim = trans_out = cfg.latent_dim
        self.encoder = MLP([cfg.enc_hidden_dim] * cfg.enc_hidden_depth, self.latent_dim)
        self._trans = MLP([cfg.model_hidden_dim] * cfg.model_hidden_depth, trans_out)
        self._reward = Ensemble(cfg.num_rewards, [cfg.model_hidden_dim] * cfg.model_hidden_depth, 1)
        self.rho = cfg.rho ** jnp.arange(cfg.horizon)

    def init(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        obs = jnp.zeros((1, self.cfg.obs_dim))
        x = jnp.zeros((1, self.latent_dim + self.cfg.action_dim + self.cfg.z_dim))
        return dict(enc=self.encoder.init(k1, obs)['params'],
                    trans=self._trans.init(k2, x)['params'],
                    reward=self._reward.init(k3, x)['params'])

    def encode(self, enc_params, obs):
        x = self.encoder.apply({'params': enc_params}, obs)
        if self.kind == 'simnorm':
            return simnorm(x, self.cfg.simnorm_dim), None
        if self.kind == 'discrete':
            return self.fsq(x)
        return x, None

    def trans(self, params, latent, a, z, key=None, mode=None):
        out = self._trans.apply({'params': params['trans']}, latent, a, z)
        if self.loss_type != 'ce':
            return latent + out, None
        logits = out.reshape(*out.shape[:-1], self.groups, self.fsq.codebook_size)
        mode = mode or self.cfg.unc_prop_mode
        if mode in ('mode', 'max'):
            probs = jax.nn.one_hot(logits.argmax(-1), self.fsq.codebook_size)
        elif mode == 'sample-no-grad':
            probs = jax.nn.one_hot(jax.random.categorical(key, logits), self.fsq.codebook_size)
        elif mode == 'sample':
            soft = jax.nn.softmax(logits - jnp.log(jax.random.exponential(key, logits.shape)), -1)
            hard = jax.nn.one_hot(soft.argmax(-1), self.fsq.codebook_size)
            probs = soft + jax.lax.stop_gradient(hard - soft)
        elif mode == 'weighted-avg':
            probs = jax.nn.softmax(logits, -1)
        else:
            raise ValueError(mode)
        nxt = probs @ self.fsq.codebook
        return nxt.reshape(*nxt.shape[:-2], -1), logits

    def reward(self, params, latent, a, z, mode='avg', penalty=0.0):
        r = self._reward.apply({'params': params['reward']}, latent, a, z)
        if mode == 'avg':
            return r.mean(0)
        if mode == 'min':
            return r.min(0)
        if mode == 'std_penalty':
            assert self.cfg.num_rewards > 1, 'std_penalty needs num_rewards > 1'
            return r.mean(0) - penalty * r.std(0, ddof=1)
        raise ValueError(mode)

    def loss(self, params, enc_tar, obss, actions, rewards, next_obss, dones, task_z, key):
        """Inputs are (seq_len, batch, dim); task_z is (batch, z_dim). Returns (tc_loss, reward_loss)."""
        tar, tar_idx = self.encode(enc_tar, next_obss)
        tar = jax.lax.stop_gradient(tar)
        latent, _ = self.encode(params['enc'], obss[0])
        latents, logits = [latent], []
        for t, k in enumerate(jax.random.split(key, self.cfg.horizon)):
            latent, lg = self.trans(params, latent, actions[t], task_z, k)
            latents.append(latent)
            logits.append(lg)
        latents = jnp.stack(latents)
        notdone = 1 - dones[..., 0]

        z = jnp.broadcast_to(task_z, (actions.shape[0], *task_z.shape))
        r_pred = self._reward.apply({'params': params['reward']}, latents[:-1], actions, z)[..., 0]
        r_err = ((r_pred - rewards[..., 0]) ** 2).mean(0)
        reward_loss = (self.rho * (notdone * r_err).mean(-1)).mean()

        if self.loss_type == 'ce':
            ce = optax.softmax_cross_entropy_with_integer_labels(jnp.stack(logits), tar_idx).mean(-1)
            tc = notdone * ce
        elif self.loss_type == 'mse':
            tc = ((latents[1:] - tar) ** 2).mean(-1)
        else:
            norm = lambda x: jnp.maximum(jnp.linalg.norm(x, axis=-1), 1e-8)
            tc = -(latents[1:] * tar).sum(-1) / (norm(latents[1:]) * norm(tar))
        tc_loss = (self.rho * tc.mean(-1)).mean()
        return tc_loss, reward_loss
