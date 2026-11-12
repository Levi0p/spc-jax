import importlib

ENVS = {
    'ant-dir': ('ant_dir', 'AntDirEnvCustom'),
    'ant-goal': ('ant_goal', 'AntGoalEnvCustom'),
    'cheetah-vel': ('cheetah_vel', 'HalfCheetahVelEnvCustom'),
    'hopper-mass': ('hopper_params', 'HopperParamsMass'),
    'hopper-friction': ('hopper_params', 'HopperParamsFriction'),
    'walker-mass': ('walker_params', 'WalkerParamsMass'),
    'walker-friction': ('walker_params', 'WalkerParamsFriction'),
    'walker-speed': ('dmc_speed', 'WalkerSpeed'),
    'cheetah-speed': ('dmc_speed', 'CheetahSpeed'),
    'finger-speed': ('dmc_speed', 'FingerSpeed'),
}

DMENVS = {
    'walker-ls': ('dmc_reward_dynamics', 'WalkerLS'),
    'cheetah-ls': ('dmc_reward_dynamics', 'CheetahLS'),
    'finger-ls': ('dmc_reward_dynamics', 'FingerLS'),
}


def _load(module, name):
    return getattr(importlib.import_module(f'.{module}', __name__), name)


def make_meta_env(env_id, idx=0, height=256, width=256, seed=None):
    if env_id == 'point-robot':
        from .point_robot import PointRobot
        from .normalized import NormalizedBoxEnv
        return NormalizedBoxEnv(PointRobot(idx=idx))
    if env_id in ENVS:
        from .normalized import NormalizedBoxEnv
        return NormalizedBoxEnv(_load(*ENVS[env_id])(render_mode='rgb_array', idx=idx, height=height, width=width))
    if env_id in DMENVS:
        return _load(*DMENVS[env_id])(idx, seed=seed)
    if env_id.startswith('mt-'):
        return _load('metaworld', 'MetaWorldEnv')(env_id[3:], idx=idx, seed=seed)
    raise NotImplementedError(env_id)
