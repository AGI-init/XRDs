# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import glob
import os
import random

import numpy as np

from torch import nn
from torch.utils.data import Dataset

from ML import main
from ML.World.Dataset import Lock


class NoPoolCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.CNN = \
            nn.Sequential(
                nn.Conv1d(in_channels, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

    def forward(self, obs):
        return self.CNN(obs)


class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.CNN = \
            nn.Sequential(
                nn.Conv1d(in_channels, 80, 100, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3, 2),
                nn.Conv1d(80, 80, 50, 5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3),
                nn.Conv1d(80, 80, 25, 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AvgPool1d(3),
            )

    def forward(self, obs):
        return self.CNN(obs)


class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, out_features))

    def forward(self, obs):
        return self.MLP(obs)


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_features, 4000), nn.ReLU(), nn.Dropout(0.6),
                                 nn.Linear(4000, 3000), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(3000, 1000), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(1000, 800), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(800, out_features))

    def forward(self, obs):
        return self.MLP(obs)


class XRD(Dataset):
    def __init__(self, icsd=True, open_access=True, rruff=True, soup=True, train=True, num_classes=7, seed=0,
                 sources=None, train_eval_splits=None):

        self.num_classes = num_classes

        if sources is None or train_eval_splits is None:
            roots, train_eval_splits = data_paths(icsd, open_access, rruff, soup)
        else:
            roots = [glob.glob(source.rstrip('/') + '/*.npy') for source in sources]

        self.indices = []
        self.data = {}

        for i, (root, split) in enumerate(zip(roots, train_eval_splits)):
            self.data[i] = root

            train_size = round(len(root) * split)

            full = range(len(root))

            # Each worker shares an indexing scheme
            random.seed(seed)
            train_indices = random.sample(full, train_size)
            eval_indices = set(full).difference(train_indices)

            indices = train_indices if train else eval_indices
            self.indices += zip([i] * len(indices), list(indices))

            print(f'Identified [source of length {len(root)}, split={split if train else 1 - split}, train={train}] ✓')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        root, idx = self.indices[idx]

        # Load data from hard disk
        data = np.load(self.data[root][idx], allow_pickle=True).item()
        x, y = data['features'] / 1000.0, data['labels7' if self.num_classes == 7 else 'labels230']

        return x, y


# Verify or download data
def data_paths(icsd, open_access, rruff, soup):
    roots = []
    train_eval_splits = []

    path = os.path.dirname(__file__)

    if rruff:
        if os.path.exists(path + '/Data/Generated/XRDs_RRUFF/'):
            roots.append(glob.glob(path + '/Data/Generated/XRDs_RRUFF/*.npy'))
            train_eval_splits += [0.5 if soup else 0]  # Split 50% of experimental RRUFF data just for training
        else:
            rruff = False
            print('Could not find RRUFF XRD files. Skipping souping and evaluating on '
                  '10% held-out portion of synthetic data.')

    if icsd:
        if os.path.exists(path + '/Data/Generated/XRDs_ICSD/') or os.path.exists(path + '/Data/Generated/CIFs_ICSD/'):
            if len(glob.glob(path + '/Data/Generated/XRDs_ICSD/*.npy')) < 171e3 * 7:  # Approximate length check
                from Data.CIF import generate
                with Lock(path + '/Data/Generated/CIFs_ICSD/Lock'):  # System-wide lock
                    generate(path + '/Data/Generated/CIFs_ICSD/')  # Generate data
            roots.append(glob.glob(path + '/Data/Generated/XRDs_ICSD/*.npy'))
            train_eval_splits += [1 if rruff else 0.9]  # Train on all synthetic data if evaluating on RRUFF
        else:
            icsd = False
            print('Could not find ICSD CIF files. Using open-access CIFs instead.')

    if open_access or not icsd:
        if len(glob.glob(path + '/Data/Generated/XRDs_open_access/*.npy')) < 8e3 * 7:  # Approximate length check
            with Lock(path + '/Data/Generated/CIFs_open_access/Lock'):  # System-wide lock
                from Data.CIF import generate, download
                if len(glob.glob(path + '/Data/Generated/CIFs_open_access/*.cif')) < 8e3:  # Approximate length check
                    download(path + '/Data/Generated/CIFs_open_access/')
                generate(path + '/Data/Generated/CIFs_open_access/')  # Generate data
        roots.append(glob.glob(path + '/Data/Generated/XRDs_open_access/*.npy'))
        train_eval_splits += [1 if rruff else 0.9]  # Train on all synthetic data if evaluating on RRUFF

    return roots, train_eval_splits


if __name__ == '__main__':
    main(task='NPCNN')


"""
Below are in-progress experiments
"""
import torch


# Moving noise augmentation to batch-vectorized GPU
class NoiseAug:
    def __call__(self, x):
        return torch.relu(torch.normal(mean=x, std=20))


# Peak shape and random noise transform at runtime
class PeakShapeTransform:
    def __init__(self, peak_shapes=(0, 1, 2, 3), noise=True, x_step=0.01):
        self.peak_shapes = peak_shapes
        self.noise = noise

        self.x_step = x_step

    def __call__(self, xy_merge):
        peak_shapes = [(0.05, -0.06, 0.07), (0.05, -0.01, 0.01),
                       (0.0, 0.0, 0.01), (0.0, 0.0, random.uniform(0.001, 0.1))]

        peak_shape = random.choice(self.peak_shapes)
        U, V, W = peak_shapes[peak_shape]

        H = np.zeros((xy_merge.shape[0], 1))
        H[:, 0] = (U * (np.tan(xy_merge[:, 0] * (np.pi/180)/2)) ** 2 + V *
                   np.tan(xy_merge[:, 0] * (np.pi/180)/2) + W) ** (1/2)

        pattern = []
        for x_val in range(0, int(180 / self.x_step)):
            y_val = 0
            for xy_idx in range(0, xy_merge.shape[0]):
                angle = xy_merge[xy_idx, 0]
                inten = xy_merge[xy_idx, 1]
                if (x_val * self.x_step - 5) < angle < (x_val * self.x_step + 5):
                    const_g = 4 * np.log(2)
                    value = ((const_g ** (1/2)) /
                             (np.pi ** (1/2) * H[xy_idx, 0])) * np.exp(-const_g * ((x_val * self.x_step - angle) /
                                                                                   H[xy_idx, 0])**2)
                    y_val = y_val + inten * (value * 1.5)
            pattern.append(y_val)

        x = torch.as_tensor(pattern)

        if self.noise and peak_shape != 2:  # Peak shape 2 represents a perfect crystal, so should not be augmented
            x = torch.relu(torch.normal(mean=x, std=20))

        return x
