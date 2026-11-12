# spc-jax

JAX/Flax version of [SPC](https://github.com/MohammadrezaNakhaei/SPC), the code for *Temporal Consistency Improves Generalization in Contextual Offline Meta-RL* (Nakhaei et al., NeurIPS 2026, [arXiv:2603.02935](https://arxiv.org/abs/2603.02935)).

Right now it has the SPC agent (context encoder + latent world model + InfoNCE, IQL actor-critic, MPPI if you want it) and DroQ for collecting the datasets. Haven't ported the baselines (FOCAL, CSRO, UNICORN, DORA) yet.

## files

```
spc/nets.py          mlp, ensembles, simnorm, fsq
spc/world_model.py   one WorldModel class for all 7 latent variants
spc/agent.py         the SPC agent + MPPI
spc/droq.py          DroQ
spc/buffer.py        per-task datasets, sequence sampling
train_spc.py         training + the 4 eval protocols
collect_data.py      collects a dataset for one task
envs/                copied from the original repo
cfgs/                same hyperparams as the original + cfgs/env/*.yaml
```

World model is picked with `agent.world_model=<latent>_<loss>`, e.g. `discrete_ce`, `continuous_mse`, `simnorm_cosine`. `ce` only works with `discrete`.

## setup

```sh
pip install -r requirements.txt
```

MuJoCo / DMC / Meta-World envs also need `mujoco`, `dm_control`, `metaworld` (same as the original `environment.yml`). Point-robot doesn't need any of that.

## running

```sh
python collect_data.py env=ant-dir goal_idx=0      # once per goal_idx
python train_spc.py env=ant-dir agent.world_model=discrete_ce seed=0
```

Datasets go to `data/<env>/goal_idx<i>/dataset.npy`, same row layout as the original `dataset.pth`, so the published datasets should work too with `dataset_file=dataset.pth` (needs torch to load). Logs go to `logs/`, `use_wandb=true` if you want wandb.

## point-robot run (laptop CPU)

```sh
seq 0 39 | xargs -P 4 -I{} python collect_data.py env=point-robot goal_idx={} agent.hidden_dim=256 batch_size=256
python train_spc.py env=point-robot agent.world_model=continuous_mse epochs=10 iter_per_epoch=2000 batch_size=32 meta_batch_size=8 num_context=200 agent.latent_dim=128 agent.enc_hidden_dim=256 agent.model_hidden_dim=256
```

Everything is smaller than the paper so it fits on a laptop (20k transitions per task, 20k SPC updates). Avg return per episode, start → end:

| context | train | id | ood |
|---|---|---|---|
| offline | -33.4 → -20.0 | -41.1 → -23.3 | -49.6 → -32.5 |
| online | -34.1 → -31.1 | -40.7 → -40.3 | -49.2 → -49.4 |
| non-prior | -33.9 → -30.0 | -42.9 → -42.3 | -51.1 → -50.6 |
| zero-shot | -33.7 → -29.0 | -43.0 → -42.7 | -51.4 → -51.0 |

(the DroQ data averages ~-27, DroQ itself gets ~-20)

Offline context works and transfers to unseen goals. Online / non-prior / zero-shot barely move on the test tasks though. I looked into it a bit: with offline context the z's for different tasks are pretty spread out (avg distance from the mean z ~0.64), but after one online episode they're almost the same for every task (~0.03), since the agent goes to the same spot whatever the goal is. So I think the encoder is picking up the task from which states are in the data, not from the rewards, i.e. offline/online context shift (probably worse with this short run). Not sure though.

## tests

```sh
pytest tests/test_core.py
SPC_TORCH_REPO=/path/to/SPC pytest tests/test_parity.py
```

`test_parity.py` loads the original PyTorch agent, copies its weights into the JAX one and runs 3 updates on the same batch in both. Losses match within 1e-4 (relative) and every weight's change within 1%, for all 7 world models. Needs `torch`, `tensordict`, `einops`, `vector-quantize-pytorch`. `discrete_ce` is checked with `unc_prop_mode=weighted-avg` since the gumbel noise is different between torch and jax.

## differences from the original

Small stuff I changed:

- InfoNCE: the original groups the batch by task with a dict keyed on tensors, but tensors hash by id, so repeated task ids don't get merged (last one wins). Here they're averaged. Same result when all task ids in a batch are unique.
- Sequence sampling: `SeqBuffer` skips the episode after the last done flag (for point-robot / cheetah-vel that's the first episode, since there are no done flags) and the last valid start index. Here any window inside one episode can be sampled.
- MPPI works with all world models. In the original only `DiscreteCE.trans` takes `unc_prop_mode`, so planning crashes with the others.
- Eval steps each env directly instead of `SyncVectorEnv` so autoreset steps don't end up in the context. The context embedding is kept as a running mean (same thing, cheaper).
- Added `cfgs/env/` (the README mentions it but it's not in the repo). Episode lengths / task counts are from the env classes, mass/friction envs use `use_next_obs_in_context`. The point-robot DroQ budget (20k steps) is my choice, not from the paper.
- Success rate stops counting once an env has done its episodes (the original can go above 1 on Meta-World). Returns are the same.
- Not ported: FSQ/L2 options on the context encoder, the FOCAL loss in `spc.py` (all off by default), torch.compile/cuda graphs, video logging. `use_tanh` / `use_tar_enc` are always on, and `random_init_z` / `resample_every` are dropped since they don't do anything in the original.

Also wasn't sure about this one in `envs/point_robot.py`: in the hallway it does `self._state[0] = np.clip(self._state[1], -0.1, 0.1)`. Should that be `self._state[0]`? Left it as is.

## license

Apache-2.0, same as the original.
