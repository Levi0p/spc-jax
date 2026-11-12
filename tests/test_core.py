import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from spc.agent import SPC
from spc.buffer import MultiTaskBuffer, valid_starts
from spc.nets import FSQ

T, B, S, OBS, ACT = 3, 4, 5, 3, 2


def make_cfg(**kw):
    cfg = OmegaConf.load('cfgs/spc.yaml').agent
    cfg.update(dict(obs_dim=OBS, action_dim=ACT, num_train_tasks=4, horizon=S, latent_dim=16, enc_hidden_dim=32,
                    model_hidden_dim=32, context_hidden_dims=32, actor_hidden_dim=32, q_hidden_dim=32,
                    num_samples=32, num_elites=4, num_pi_trajs=4, name='test'))
    cfg.update(kw)
    return cfg


def test_valid_starts_respects_dones_and_episode_length():
    dones = np.zeros(12)
    dones[4] = 1
    starts = valid_starts(dones, seq_len=3, ep_length=4)
    # episodes: [0..4] split by length into [0..3],[4]; then [5..8],[9..11]
    assert starts.tolist() == [0, 1, 5, 6, 9]


def test_fsq_codebook_roundtrip():
    fsq = FSQ([5, 3])
    codes, idx = fsq(jnp.linspace(-3, 3, 10)[:, None].repeat(2, 1))
    np.testing.assert_allclose(fsq.codebook[idx].reshape(codes.shape), codes)


def test_fsq_straight_through_gradient():
    g = jax.grad(lambda x: FSQ([5, 3])(x)[0].sum())(jnp.zeros((1, 2)))
    assert np.all(np.asarray(g) > 0)


@pytest.mark.parametrize('world_model', ['continuous_mse', 'simnorm_cosine', 'discrete_mse', 'discrete_ce'])
@pytest.mark.parametrize('mpc', [False, True])
def test_update_and_act_run(world_model, mpc):
    cfg = make_cfg(world_model=world_model, mpc=mpc)
    agent = SPC(cfg)
    state = agent.init(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    f = lambda *s: rng.standard_normal(s).astype(np.float32)
    batch = (f(T, B, S, OBS), np.tanh(f(T, B, S, ACT)), f(T, B, S, 1), f(T, B, S, OBS), np.zeros((T, B, S, 1), np.float32))
    context = f(T, 8, agent.ctx_in)
    for update_model in (True, False):
        state, info = agent.update(state, batch, context, np.array([0, 1, 3]), jax.random.PRNGKey(1), update_model)
        assert all(np.isfinite(v) for v in info.values())
    z = agent.infer_z(state['params']['ctx'], context)
    prev_mean = jnp.zeros((T, cfg.plan_horizon, ACT))
    a, prev_mean = agent.act(state['params'], f(T, OBS), z, jax.random.PRNGKey(2), prev_mean, False, True)
    a, _ = agent.act(state['params'], f(T, OBS), z, jax.random.PRNGKey(3), prev_mean, True, False)
    assert a.shape == (T, ACT) and np.all(np.abs(a) <= 1)


def test_buffer_shapes():
    data = [np.random.randn(50, 2 * OBS + ACT + 2).astype(np.float32) for _ in range(3)]
    for d in data:
        d[:, 0], d[:, -1] = np.arange(50), 0
    buf = MultiTaskBuffer(data, OBS, ACT, seq_len=S, ep_length=10)
    obs, act, rew, next_obs, done = buf.sample_seq([0, 2], 7)
    assert obs.shape == (2, 7, S, OBS) and act.shape == (2, 7, S, ACT) and rew.shape == done.shape == (2, 7, S, 1)
    step = obs[..., 0]
    assert np.all(np.diff(step, axis=-1) == 1) and np.all(step[..., 0] // 10 == step[..., -1] // 10)
    obs, *_ = buf.sample([1], 9)
    assert obs.shape == (1, 9, OBS)
