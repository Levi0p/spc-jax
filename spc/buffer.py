from pathlib import Path

import numpy as np


def load_dataset(path):
    """Rows are [obs, action, reward, next_obs, done]. Reads .npy, or the original .pth (needs torch)."""
    path = Path(path)
    if path.suffix == '.pth':
        import torch
        return torch.load(path, map_location='cpu').numpy().astype(np.float32)
    return np.load(path).astype(np.float32)


def split(rows, obs_dim, action_dim):
    o, a = obs_dim, action_dim
    return rows[..., :o], rows[..., o:o + a], rows[..., o + a:o + a + 1], rows[..., o + a + 1:-1], rows[..., -1:]


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity):
        self.obs_dim, self.action_dim, self.capacity = obs_dim, action_dim, capacity
        self.data = np.zeros((capacity, 2 * obs_dim + action_dim + 2), np.float32)
        self.idx, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        self.data[self.idx] = np.concatenate([obs, action, [reward], next_obs, [done]])
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, n):
        return split(self.data[np.random.randint(0, len(self), n)], self.obs_dim, self.action_dim)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.data[:len(self)])


def valid_starts(dones, seq_len, ep_length):
    """Start indices of windows of `seq_len` steps that stay inside one episode. An episode ends at a
    done flag or after `ep_length` steps, whichever comes first."""
    ends = np.flatnonzero(dones)
    first = np.concatenate([[0], ends + 1])
    first = first[first < len(dones)]
    segment = np.cumsum(np.isin(np.arange(len(dones)), first)) - 1
    step_in_segment = np.arange(len(dones)) - first[segment]
    ep = segment * (len(dones) + 1) + step_in_segment // ep_length
    starts = np.arange(len(dones) - seq_len + 1)
    return starts[ep[starts] == ep[starts + seq_len - 1]]


class MultiTaskBuffer:
    """One offline dataset per task. Samples transitions (for context) and sequences (for training)."""

    def __init__(self, datasets, obs_dim, action_dim, seq_len, ep_length):
        self.data = datasets
        self.seq_len = seq_len
        self.split = lambda rows: split(rows, obs_dim, action_dim)
        self.starts = [valid_starts(d[:, -1], seq_len, ep_length) for d in datasets]

    def sample(self, task_ids, n):
        """Tuple of (tasks, n, dim)."""
        return self.split(np.stack([self.data[t][np.random.randint(0, len(self.data[t]), n)] for t in task_ids]))

    def sample_seq(self, task_ids, n):
        """Tuple of (tasks, n, seq_len, dim)."""
        offsets = np.arange(self.seq_len)
        rows = [self.data[t][np.random.choice(self.starts[t], n)[:, None] + offsets] for t in task_ids]
        return self.split(np.stack(rows))
