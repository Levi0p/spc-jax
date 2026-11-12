import time

import hydra
import jax
import numpy as np

from envs import make_meta_env
from spc.buffer import ReplayBuffer
from spc.droq import DroQ
from spc.utils import Logger, set_seed, save


@hydra.main(version_base='1.3', config_path='cfgs', config_name='data_collection')
def main(cfg):
    key = set_seed(cfg.seed)
    env, eval_env = make_meta_env(cfg.env_name, seed=cfg.seed), make_meta_env(cfg.env_name, seed=cfg.seed)
    env.reset_task(cfg.goal_idx)
    eval_env.reset_task(cfg.goal_idx)
    env.action_space.seed(cfg.seed)
    cfg.agent.obs_dim = env.observation_space.shape[0]
    cfg.agent.action_dim = env.action_space.shape[0]
    logger = Logger(cfg)
    agent = DroQ(cfg.agent)
    key, k = jax.random.split(key)
    state = agent.init(k)
    buffer = ReplayBuffer(cfg.agent.obs_dim, cfg.agent.action_dim, cfg.replay_buffer_capacity)
    dataset = ReplayBuffer(cfg.agent.obs_dim, cfg.agent.action_dim, cfg.dataset_size)

    def run_episodes(n, deterministic, store):
        nonlocal key
        total_reward = total_success = 0.0
        for _ in range(n):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                key, k = jax.random.split(key)
                action = agent.act(state, obs, k, deterministic)
                next_obs, reward, term, trunc, info = eval_env.step(action)
                if store:
                    dataset.add(obs, action, reward, next_obs, term)
                total_reward += reward
                done, obs = term or trunc, next_obs
            total_success += info.get('success', 0)
        return total_reward / n, total_success / n

    obs, info = env.reset()
    done, episode, episode_return, start, train_info = False, 0, 0.0, time.time(), {}
    for step in range(cfg.num_train_steps + 1):
        if step and step % cfg.eval_frequency == 0:
            eval_return, eval_success = run_episodes(cfg.num_eval_episodes, True, False)
            traj_return, traj_success = run_episodes(cfg.num_eval_sample_episodes, False, True)
            logger.log(dict(step=step, episode=episode, episode_reward=eval_return, episode_success=eval_success,
                            traj_returns=traj_return, traj_success=traj_success, total_time=time.time() - start), 'eval')
        if done:
            episode += 1
            logger.log(dict(step=step, episode=episode, episode_reward=episode_return, episode_success=info.get('success', 0),
                            total_time=time.time() - start, **train_info), echo=False)
            episode_return, (obs, info), done = 0.0, env.reset(), False

        key, k = jax.random.split(key)
        if step > cfg.num_seed_steps:
            state, train_info = agent.update(state, *buffer.sample(cfg.batch_size), k)
            key, k = jax.random.split(key)
            action = agent.act(state, obs, k)
        else:
            action = env.action_space.sample()
        next_obs, reward, term, trunc, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, term)
        done, episode_return, obs = term or trunc, episode_return + reward, next_obs

    logger.finish()
    dataset.save(f'{cfg.log_dir}/dataset.npy')
    if cfg.save_agent:
        save(state['params'], f'{cfg.log_dir}/agent.pkl')
    print(f'saved {len(dataset)} transitions to {cfg.log_dir}/dataset.npy')


if __name__ == '__main__':
    main()
