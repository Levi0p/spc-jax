"""Checks the JAX port against the original PyTorch implementation.

Needs torch, tensordict, einops and vector-quantize-pytorch, plus a checkout of the original repo:
    SPC_TORCH_REPO=/path/to/SPC pytest tests/test_parity.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = os.environ.get('SPC_TORCH_REPO')
torch = pytest.importorskip('torch')
if not REPO:
    pytest.skip('SPC_TORCH_REPO not set', allow_module_level=True)
sys.path.insert(0, REPO)
from agents.spc import CWM  # noqa: E402
from utils.fsq import FSQ as TorchFSQ  # noqa: E402

from spc.agent import SPC  # noqa: E402
from spc.nets import FSQ  # noqa: E402

T, B, S, N = 4, 8, 3, 16
OBS, ACT = 3, 2


def make_cfg(world_model, **kw):
    cfg = OmegaConf.load(os.path.join(REPO, 'cfgs/spc.yaml')).agent
    cfg.update(dict(world_model=world_model, obs_dim=OBS, action_dim=ACT, device='cpu', num_train_tasks=6,
                    horizon=S, latent_dim=16, enc_hidden_dim=32, model_hidden_dim=32, context_hidden_dims=32,
                    actor_hidden_dim=32, q_hidden_dim=32, num_rewards=2, model_update_freq=1, name='test'))
    cfg.update(kw)
    return cfg


def mlp_params(net):
    layers = list(net.net)
    params = {}
    for i, layer in enumerate(layers):
        params[f'fc{i}'] = dict(kernel=layer.weight.detach().numpy().T, bias=layer.bias.detach().numpy())
        if hasattr(layer, 'ln'):
            params[f'ln{i}'] = dict(scale=layer.ln.weight.detach().numpy(), bias=layer.ln.bias.detach().numpy())
    return params


def ensemble_params(ens):
    flat = dict(zip(ens.params_dict.keys(), [p.detach().numpy() for p in ens.params]))
    params = {}
    for key, value in flat.items():
        i, kind = int(key.split('.')[1]), key.split('.')[-1]
        if '.ln.' in key:
            params.setdefault(f'ln{i}', {})['scale' if kind == 'weight' else 'bias'] = value
        else:
            params.setdefault(f'fc{i}', {})['kernel' if kind == 'weight' else 'bias'] = value.swapaxes(-1, -2) if kind == 'weight' else value
    return {'mlp': params}


def torch_to_jax(agent):
    m = agent.model
    params = dict(
        model=dict(enc=mlp_params(m.encoder), trans=mlp_params(m._trans), reward=ensemble_params(m._reward)),
        ctx=mlp_params(agent.context_encoder),
        actor=mlp_params(agent.actor),
        qs=ensemble_params(agent.qs),
        v=mlp_params(agent.v),
    )
    target = dict(enc=mlp_params(m.encoder_tar), qs=ensemble_params(agent.qs_target))
    return jax.tree.map(jnp.asarray, params), jax.tree.map(jnp.asarray, target)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *shape: rng.standard_normal(shape).astype(np.float32)
    dones = (rng.random((T, S, B, 1)) < 0.1).astype(np.float32)
    batch = [f(T, S, B, OBS), np.tanh(f(T, S, B, ACT)), f(T, S, B, 1), f(T, S, B, OBS), dones]
    context = [f(T, N, OBS), np.tanh(f(T, N, ACT)), f(T, N, 1), f(T, N, OBS), np.zeros((T, N, 1), np.float32)]
    return batch, context


def max_rel_diff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / (np.max(np.abs(np.asarray(b))) + 1e-8))


def assert_same_change(new_j, new_t, old):
    for (path, a), b, o in zip(jax.tree_util.tree_leaves_with_path(new_j), jax.tree.leaves(new_t), jax.tree.leaves(old)):
        assert max_rel_diff(np.asarray(a) - o, np.asarray(b) - o) < 1e-2, jax.tree_util.keystr(path)


def test_fsq_matches():
    x = np.random.default_rng(0).standard_normal((7, 12)).astype(np.float32) * 3
    codes_t, idx_t = TorchFSQ([5, 3])(torch.from_numpy(x))
    codes_j, idx_j = FSQ([5, 3])(jnp.asarray(x))
    np.testing.assert_allclose(codes_j, codes_t.numpy(), atol=1e-6)
    np.testing.assert_array_equal(idx_j, idx_t.numpy())
    np.testing.assert_allclose(FSQ([5, 3]).codebook, TorchFSQ([5, 3])._fsq.implicit_codebook.numpy(), atol=1e-6)


@pytest.mark.parametrize('world_model', ['continuous_mse', 'continuous_cosine', 'simnorm_mse', 'simnorm_cosine',
                                         'discrete_mse', 'discrete_cosine', 'discrete_ce'])
def test_update_matches(world_model):
    torch.manual_seed(0)
    cfg = make_cfg(world_model, unc_prop_mode='weighted-avg')
    t_agent = CWM(cfg)
    params, target = torch_to_jax(t_agent)

    j_agent = SPC(cfg)
    state = j_agent.init(jax.random.PRNGKey(0))
    state['params'], state['target'] = params, target
    state['opt'] = {k: o.init(params[k]) for k, o in j_agent.opt.items()}

    batch, context = make_data()
    task_ids = np.array([0, 3, 1, 5])
    for step in range(3):
        t_info = t_agent.update([torch.from_numpy(x) for x in batch], [torch.from_numpy(x) for x in context],
                                task_ids.tolist(), step=step + 1)
        j_batch = tuple(x.swapaxes(1, 2) for x in batch)
        j_context = np.concatenate(context[:3], -1)
        state, j_info = j_agent.update(state, j_batch, j_context, task_ids, jax.random.PRNGKey(step), True)

    for k in ['tc_loss', 'reward_loss', 'infonce_loss', 'critic_loss', 'v_loss', 'policy_loss']:
        assert max_rel_diff(j_info[k], t_info[k].item()) < 1e-4, k
    new_params, new_target = torch_to_jax(t_agent)
    assert_same_change(state['params'], new_params, params)
    assert_same_change(state['target'], new_target, target)
    np.testing.assert_allclose(state['history'], t_agent.latent_history_means.numpy(), atol=1e-6)


def test_actor_critic_only_step_matches():
    torch.manual_seed(1)
    cfg = make_cfg('continuous_mse', model_update_freq=2)
    t_agent = CWM(cfg)
    params, target = torch_to_jax(t_agent)
    j_agent = SPC(cfg)
    state = j_agent.init(jax.random.PRNGKey(0))
    state['params'], state['target'] = params, target
    state['opt'] = {k: o.init(params[k]) for k, o in j_agent.opt.items()}
    batch, context = make_data(1)
    task_ids = np.array([2, 4, 0, 1])
    t_agent.update([torch.from_numpy(x) for x in batch], [torch.from_numpy(x) for x in context], task_ids.tolist(), step=1)
    state, _ = j_agent.update(state, tuple(x.swapaxes(1, 2) for x in batch), np.concatenate(context[:3], -1),
                              task_ids, jax.random.PRNGKey(0), False)
    new_params, _ = torch_to_jax(t_agent)
    unchanged = {k: params[k] for k in ('model', 'ctx')}
    assert_same_change({k: v for k, v in state['params'].items() if k not in unchanged},
                       {k: v for k, v in new_params.items() if k not in unchanged},
                       {k: v for k, v in params.items() if k not in unchanged})
    for a, b in zip(jax.tree.leaves(unchanged), jax.tree.leaves({k: state['params'][k] for k in unchanged})):
        np.testing.assert_array_equal(a, b)
