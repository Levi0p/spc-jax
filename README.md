# SPC in JAX

This is a JAX/Flax version of [SPC](https://github.com/MohammadrezaNakhaei/SPC), the code for *Temporal Consistency Improves Generalization in Contextual Offline Meta-RL* (Nakhaei, Scannell, Luck, Pajarinen, NeurIPS 2026, [arXiv:2603.02935](https://arxiv.org/abs/2603.02935)).

So far it has the SPC agent itself (contextual latent world model, InfoNCE context loss, IQL actor-critic, optional MPPI planning) and the DroQ agent used to collect the datasets. The baselines (FOCAL, CSRO, UNICORN, DORA) aren't ported yet.

## What's where

```
spc/nets.py          MLP (Linear-LayerNorm-Mish), ensembles, SimNorm, FSQ
spc/world_model.py   one WorldModel class for all seven latent variants
spc/agent.py         the SPC agent: context encoder, updates, MPPI
spc/droq.py          DroQ for data collection
spc/buffer.py        per-task datasets and sequence sampling
train_spc.py         training + the four eval protocols (offline, online, non-prior, zero-shot)
collect_data.py      collects one dataset per task
envs/                envs from the original repo (imports are lazy, NormalizedBoxEnv moved to its own file)
cfgs/                same hyperparameters as the original, plus cfgs/env/*.yaml
```

You pick the world model with `agent.world_model=<latent>_<loss>`. `<latent>` is `continuous`, `simnorm` or `discrete` (FSQ). `<loss>` is `mse`, `cosine` or `ce` (`ce` only works with `discrete`).

## Setup

```sh
pip install -r requirements.txt
```

For the MuJoCo, DMC and Meta-World envs you also need `mujoco`, `dm_control` and `metaworld`, same as in the original `environment.yml`. Point-robot doesn't need any of them.

## Running it

Collect a dataset for every task, then train:

```sh
python collect_data.py env=ant-dir goal_idx=0            # do this for every goal_idx
python train_spc.py env=ant-dir agent.world_model=discrete_ce seed=0
```

Datasets go to `data/<env>/goal_idx<i>/dataset.npy`. The rows are laid out the same way as the original `dataset.pth` files, so the published datasets work too with `dataset_file=dataset.pth` (you need torch installed to read those).

Point-robot is small enough to run on a laptop CPU:

```sh
seq 0 39 | xargs -P 4 -I{} python collect_data.py env=point-robot goal_idx={} agent.hidden_dim=256 batch_size=256
```

Then train with the command in the next section.

Logs end up in `logs/<env>/<agent>/seed<k>/`. Add `use_wandb=true` if you want wandb too.

## Point-robot run

A small CPU run of the full pipeline: 40 DroQ datasets of 20k transitions each, then 20k SPC updates with `continuous_mse`, latent 128, hidden 256, `batch_size=32` and `meta_batch_size=8` (smaller than the paper's settings so it fits a laptop). Average return per episode at the start and end of training:

| context | train | in-distribution | out-of-distribution |
|---|---|---|---|
| offline | -33.4 → -20.0 | -41.1 → -23.3 | -49.6 → -32.5 |
| online | -34.1 → -31.1 | -40.7 → -40.3 | -49.2 → -49.4 |
| non-prior | -33.9 → -30.0 | -42.9 → -42.3 | -51.1 → -50.6 |
| zero-shot | -33.7 → -29.0 | -43.0 → -42.7 | -51.4 → -51.0 |

For reference, the DroQ data averages about -27 per episode, and DroQ itself reaches about -20.

With offline context it learns and carries over to unseen goals. The other three barely move on the test tasks. I checked why: z from the offline context differs a lot between tasks (mean distance from the average z is 0.64), but after one online episode it's almost the same for every task (0.03). The agent walks to the same spot whatever the goal, and the encoder seems to recognise tasks by which states show up in the data rather than by the rewards. So my guess is it's the usual context shift between offline and online data, made worse by the short run, not a bug in the eval. Command:

```sh
python train_spc.py env=point-robot agent.world_model=continuous_mse epochs=10 iter_per_epoch=2000 batch_size=32 meta_batch_size=8 num_context=200 agent.latent_dim=128 agent.enc_hidden_dim=256 agent.model_hidden_dim=256
```

## Tests

```sh
pytest tests/test_core.py
SPC_TORCH_REPO=/path/to/SPC pytest tests/test_parity.py
```

`test_parity.py` builds the original PyTorch agent, copies its weights into the JAX one, and runs three updates on the same batch in both. For all seven world models, the losses agree within 1e-4 (relative) and the change in every weight agrees within 1%. You need `torch`, `tensordict`, `einops` and `vector-quantize-pytorch` for this one. For `discrete_ce` it uses `unc_prop_mode=weighted-avg`, since Gumbel sampling draws different random numbers in torch and JAX.

## Things that differ from the PyTorch code

A few small things I changed on purpose:

- **InfoNCE.** The original groups the batch by task using a dict with tensor keys. Tensors hash by identity, so repeated task ids never get merged and the last one overwrites the rest. Here they're averaged. If every task id in a meta-batch is unique, both versions give the same result.
- **Sequence sampling.** `SeqBuffer` skips the episode right after the last done flag, which for datasets with no done flags at all (like point-robot and cheetah-vel) is the very first episode. It also never samples the last valid start index. Here any window that stays inside one episode can be sampled.
- **MPPI** works with every world model. In the original, `plan` passes `unc_prop_mode` to `trans`, but only `DiscreteCE.trans` accepts it.
- **Eval** steps each env directly instead of going through `SyncVectorEnv`, so autoreset steps don't leak into the context. During rollouts the context embedding is a running mean, which is the same thing as re-encoding the whole context every step, just cheaper.
- **`cfgs/env/`** is added. The original README mentions it but it isn't in the repo. Episode lengths and task counts come from the env classes, and the mass/friction envs turn on `use_next_obs_in_context` since their tasks differ in dynamics. `point-robot.yaml` also shrinks the DroQ budget (20k steps, 20k transitions per task) so it runs on a CPU; that's my choice, not from the paper.
- **Success rate** in eval stops counting once an env has done its episodes. The original keeps stepping finished envs, so on Meta-World its success rate could go above 1. Returns are the same either way.
- **Not ported:** the FSQ/L2 options on the context encoder and the FOCAL loss in `spc.py` (all off by default), `torch.compile` / CUDA graphs, and video logging. `use_tanh` and `use_tar_enc` are always on (their defaults), and `random_init_z` / `resample_every` are gone since they don't change anything in the original.

## License

Apache-2.0, same as the original. `envs/` is from the original repo.
