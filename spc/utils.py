import csv
import pickle
import random
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(obj), f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Logger:
    """Prints scalars, appends them to <log_dir>/<category>.csv, and optionally sends them to wandb."""

    def __init__(self, cfg):
        self.log_dir = Path(cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.wandb = {}, None
        if cfg.use_wandb:
            import wandb
            self.wandb = wandb.init(project=cfg.project_name, group=cfg.group, name=cfg.run_name,
                                    config=OmegaConf.to_container(cfg, resolve=True))

    def log(self, metrics, category='train', echo=True):
        metrics = {k: float(v) for k, v in metrics.items()}
        rows = self.rows.setdefault(category, [])
        rows.append(metrics)
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(self.log_dir / f'{category}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if self.wandb:
            self.wandb.log({f'{category}/{k}': v for k, v in metrics.items()}, step=int(metrics.get('step', 0)))
        if echo:
            print(category, ' '.join(f'{k}={v:.4g}' for k, v in metrics.items() if '/' not in k or 'avg' in k))

    def finish(self):
        if self.wandb:
            self.wandb.finish()
