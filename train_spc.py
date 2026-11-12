import hydra
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from envs import make_meta_env
from spc.agent import SPC
from spc.buffer import MultiTaskBuffer, load_dataset
from spc.utils import Logger, set_seed, save


class Evaluator:
    """Runs all tasks side by side, one env per task, under the four context protocols of the paper."""

    def __init__(self, cfg, agent, buffer, key):
        self.cfg, self.agent, self.buffer, self.key = cfg, agent, buffer, key
        self.envs = [make_meta_env(cfg.env_name, idx=i, seed=cfg.seed) for i in range(cfg.num_tasks)]
        for i, env in enumerate(self.envs):
            env.reset_task(i)
            env.action_space.seed(cfg.seed + i)
        self.encode = jax.jit(agent.encode_context)

    def _next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def _reset(self):
        self.prev_mean = jnp.zeros((len(self.envs), self.cfg.agent.plan_horizon, self.cfg.agent.action_dim))
        return np.stack([env.reset()[0] for env in self.envs])

    def _act(self, params, obs, z, deterministic, t0):
        a, self.prev_mean = self.agent.act(params, jnp.asarray(obs, jnp.float32), z, self._next_key(),
                                           self.prev_mean, deterministic, t0)
        return np.asarray(a)

    def _step(self, actions, active):
        n = len(self.envs)
        next_obs, rewards = np.zeros((n, self.cfg.agent.obs_dim)), np.zeros(n)
        dones, success = np.zeros(n, bool), np.zeros(n)
        for i in np.flatnonzero(active):
            next_obs[i], rewards[i], term, trunc, info = self.envs[i].step(actions[i])
            dones[i], success[i] = term or trunc, info.get('success', 0)
        return next_obs, rewards, dones, success

    def _transition(self, obs, actions, rewards, next_obs):
        parts = [obs, actions, rewards[:, None]] + ([next_obs] if self.cfg.agent.use_next_obs_in_context else [])
        return jnp.asarray(np.concatenate(parts, -1), jnp.float32)

    def evaluate(self, params, z, n_episodes):
        n = len(self.envs)
        returns, successes, counts = np.zeros(n), np.zeros(n), np.zeros(n, int)
        obs, t0 = self._reset(), True
        while np.any(counts < n_episodes):
            active = counts < n_episodes
            actions = self._act(params, obs, z, True, t0)
            obs, rewards, dones, success = self._step(actions, active)
            returns += rewards
            t0 = False
            for i in np.flatnonzero(dones):
                obs[i], counts[i], successes[i], t0 = self.envs[i].reset()[0], counts[i] + 1, successes[i] + success[i], True
        return returns / n_episodes, successes / n_episodes

    def zero_shot(self, params, n_episodes):
        n = len(self.envs)
        returns, successes = np.zeros(n), np.zeros(n)
        for _ in range(n_episodes):
            obs, t0 = self._reset(), True
            z_sum, count = jnp.zeros((n, self.cfg.agent.z_dim)), 0
            z = jnp.zeros((n, self.cfg.agent.z_dim))
            active = np.ones(n, bool)
            while active.any():
                actions = self._act(params, obs, z, True, t0)
                next_obs, rewards, dones, success = self._step(actions, active)
                z_sum, count = z_sum + self.encode(params['ctx'], self._transition(obs, actions, rewards, next_obs)), count + 1
                z = z_sum / count
                returns += rewards
                successes += success * dones
                active &= ~dones
                obs, t0 = next_obs, False
        return returns / n_episodes, successes / n_episodes

    def collect_context(self, params, n_steps, update_z, random_steps):
        """Rolls out for n_steps and returns the task embedding from the collected transitions.
        update_z=False: the policy acts with the prior z=0 throughout (online context).
        update_z=True: z is re-inferred after every step; the first random_steps+1 actions are random."""
        n = len(self.envs)
        obs, t0 = self._reset(), True
        z_sum, z = jnp.zeros((n, self.cfg.agent.z_dim)), jnp.zeros((n, self.cfg.agent.z_dim))
        for step in range(n_steps):
            if step <= random_steps:
                actions = np.stack([env.action_space.sample() for env in self.envs])
            else:
                actions = self._act(params, obs, z, False, t0)
            t0 = False
            next_obs, rewards, dones, _ = self._step(actions, np.ones(n, bool))
            z_sum = z_sum + self.encode(params['ctx'], self._transition(obs, actions, rewards, next_obs))
            if update_z:
                z = z_sum / (step + 1)
            obs = next_obs
            for i in np.flatnonzero(dones):
                obs[i], t0 = self.envs[i].reset()[0], True
        return z_sum / n_steps

    def offline_context(self, params, n):
        obs, actions, rewards, next_obs, _ = self.buffer.sample(range(self.cfg.num_tasks), n)
        parts = [obs, actions, rewards] + ([next_obs] if self.cfg.agent.use_next_obs_in_context else [])
        return self.agent.infer_z(params['ctx'], jnp.asarray(np.concatenate(parts, -1)))

    def run(self, params):
        cfg = self.cfg
        results = dict(
            offline=self.evaluate(params, self.offline_context(params, cfg.num_context), cfg.num_eval_episodes),
            online=self.evaluate(params, self.collect_context(params, cfg.num_context, False, 0), cfg.num_eval_episodes),
            non_prior=self.evaluate(params, self.collect_context(params, cfg.num_context, True, cfg.random_np_steps), cfg.num_eval_episodes),
            zero_shot=self.zero_shot(params, cfg.num_eval_episodes),
        )
        modes = self.envs[0].task_modes()
        logs = {}
        for name, (returns, success) in results.items():
            for mode, idx in modes.items():
                logs[f'{mode}/avg_{name}'] = returns[idx].mean()
                logs[f'{mode}/avg_{name}_success'] = success[idx].mean()
                if cfg.log_all:
                    logs.update({f'{mode}/{i}_{name}': returns[i] for i in idx})
        return logs


@hydra.main(version_base='1.3', config_path='cfgs', config_name='spc')
def main(cfg):
    key = set_seed(cfg.seed)
    env = make_meta_env(cfg.env_name, seed=cfg.seed)
    cfg.agent.obs_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    train_tasks = env.task_modes()['train']
    cfg.agent.num_train_tasks = len(train_tasks)
    logger = Logger(cfg)

    datasets = [load_dataset(f'{cfg.data_dir}/{cfg.env_name}/goal_idx{i}/{cfg.dataset_file}') for i in range(cfg.num_tasks)]
    buffer = MultiTaskBuffer(datasets, cfg.agent.obs_dim, cfg.agent.action_dim, cfg.seq_len, cfg.max_episode_steps)
    agent = SPC(cfg.agent)
    key, k_init, k_eval = jax.random.split(key, 3)
    state = agent.init(k_init)
    evaluator = Evaluator(cfg, agent, buffer, k_eval)
    n_params = sum(x.size for x in jax.tree.leaves(state['params']))
    print(f'SPC ({cfg.agent.world_model}): {n_params / 1e6:.2f}M parameters, {sum(map(len, datasets))} transitions')

    for epoch in range(cfg.epochs):
        for itr in trange(cfg.iter_per_epoch, desc=f'epoch {epoch}', miniters=25):
            train_idx = np.random.randint(0, len(train_tasks), cfg.meta_batch_size)
            task_idx = train_tasks[train_idx]
            batch = buffer.sample_seq(task_idx, cfg.batch_size)
            obs, actions, rewards, next_obs, _ = buffer.sample(task_idx, cfg.context_batch_size)
            context = np.concatenate([obs, actions, rewards] + ([next_obs] if cfg.agent.use_next_obs_in_context else []), -1)
            key, k = jax.random.split(key)
            state, info = agent.update(state, batch, context, train_idx, k, (itr + 1) % cfg.agent.model_update_freq == 0)
        step = cfg.iter_per_epoch * (epoch + 1)
        logger.log(dict(step=step, **info))
        logger.log(dict(step=step, **evaluator.run(state['params'])), 'eval')
        if (epoch + 1) % cfg.save_every == 0:
            save(state['params'], f'{cfg.log_dir}/agent.pkl')
    save(state['params'], f'{cfg.log_dir}/agent.pkl')
    logger.finish()


if __name__ == '__main__':
    main()
