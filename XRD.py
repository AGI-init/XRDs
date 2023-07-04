# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import os
import random

import numpy as np

import torch
from ML.World.Dataset import Lock
from torch import nn
from torch.utils.data import Dataset

from Data.Generated.Download import download
from Data.Generated.Generate import generate

import ML


class NoPoolCNN(nn.Module):
    def __init__(self, input_shape=(1,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

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
    def __init__(self, input_shape=(1,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

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
    def __init__(self, input_shape=(1024,), output_shape=(7,)):
        super().__init__()

        input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(input_dim, 2300), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(2300, 1150), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(1150, output_dim))

    def forward(self, obs):
        return self.MLP(obs)


class MLP(nn.Module):
    def __init__(self, input_shape=(8500,), output_shape=(7,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)

        self.MLP = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_channels, 4000), nn.ReLU(), nn.Dropout(0.6),
                                 nn.Linear(4000, 3000), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(3000, 1000), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(1000, 800), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(800, output_dim))

    def forward(self, obs):
        return self.MLP(obs)


class XRD(Dataset):
    def __init__(self, icsd=True, open_access=True, rruff=True, soup=True, train=True, num_classes=7, seed=0,
                 roots=None, train_eval_splits=None):

        self.num_classes = num_classes

        if roots is None and train_eval_splits is None:
            with Lock('XRD_data'):
                roots, train_eval_splits = data_paths(icsd, open_access, rruff, soup)

        self.indices = []
        self.features = {}
        self.labels = {}

        for i, (root, split) in enumerate(zip(roots, train_eval_splits)):
            features_path = root + "features.csv"
            label_path = root + f"labels{num_classes}.csv"

            print(f'Loading [root={root}, split={split if train else 1 - split}, train={train}] to CPU...')

            # Store files
            with open(features_path, "r") as f:
                self.features[i] = f.readlines()
            with open(label_path, "r") as f:
                self.labels[i] = f.readlines()
                full_size = len(self.labels[i])

            print('Data loaded ✓')

            train_size = round(full_size * split)

            full = range(full_size)

            # Each worker shares an indexing scheme
            random.seed(seed)
            train_indices = random.sample(full, train_size)
            eval_indices = set(full).difference(train_indices)

            indices = train_indices if train else eval_indices
            self.indices += zip([i] * len(indices), list(indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        root, idx = self.indices[idx]

        x = torch.FloatTensor(list(map(float, self.features[root][idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[root][idx].strip().split(',')))).argmax()

        return x, y


class NoiseAug:
    def __call__(self, x):
        return torch.relu(torch.normal(mean=x, std=20))


# TODO
#  1. What is the shape of xy_merge in data generation? - Answer: It varies (variable length x 2)
#  2. Can this "y_multi" loop be optimized / batch vectorized? - Answer: Yes!
#     - Either manually batch vectorizing
#     - Or JIT
#  ---
#  Conclusion: NoiseAug can be batch-vectorized and moved to GPU, peak shapes can be sped up but remain on CPU.
# Peak shape and random noise augmentation
class PeakShapeAug:
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


# Can use this Dataset instead to avoid loading the full dataset into RAM
class MemoryEfficientXRD(Dataset):
    def __init__(self, icsd=True, open_access=True, rruff=True, soup=True, train=True, num_classes=7, seed=0,
                 roots=None, train_eval_splits=None):

        self.num_classes = num_classes

        if roots is None and train_eval_splits is None:
            with Lock('XRD_data'):
                roots, train_eval_splits = data_paths(icsd, open_access, rruff, soup)

        self.indices = []
        self.features = {}
        self.labels = {}

        for i, (root, split) in enumerate(zip(roots, train_eval_splits)):
            features_path = root + "features.csv"
            label_path = root + f"labels{num_classes}.csv"

            print(f'Loading [root={root}, split={split if train else 1 - split}, train={train}] to CPU...')

            self.features[i] = features_path
            with open(label_path, "r") as f:
                self.labels[i] = label_path
                full_size = sum(1 for _ in f)

            print('Data loaded ✓')

            train_size = round(full_size * split)

            full = range(full_size)

            # Each worker shares an indexing scheme
            random.seed(seed)
            train_indices = random.sample(full, train_size)
            eval_indices = set(full).difference(train_indices)

            indices = train_indices if train else eval_indices
            self.indices += zip([i] * len(indices), list(indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        root, idx = self.indices[idx]

        # This reads in the lines incrementally without ever loading the whole file into memory
        with open(self.features[root], "r") as f:
            for i, features in enumerate(f):
                if i == idx:
                    break

        with open(self.labels[root], "r") as f:
            for i, labels in enumerate(f):
                if i == idx:
                    break

        x = torch.FloatTensor(list(map(float, features.strip().split(','))))
        y = np.array(list(map(float, labels.strip().split(',')))).argmax()

        return x, y


# Verify or download data
def data_paths(icsd, open_access, rruff, soup):
    roots = []
    train_eval_splits = []

    path = os.path.dirname(__file__)

    if rruff:
        if os.path.exists(path + '/Data/Generated/XRDs_RRUFF/'):
            roots.append(path + '/Data/Generated/XRDs_RRUFF/')
            train_eval_splits += [0.5 if soup else 0]  # Split 50% of experimental RRUFF data just for training
        else:
            rruff = False
            print('Could not find RRUFF XRD files. Skipping souping and evaluating on '
                  '10% held-out portion of synthetic data.')

    if icsd:
        if os.path.exists(path + '/Data/Generated/CIFs_ICSD/'):
            generate(path + '/Data/Generated/CIFs_ICSD/', path + '/Data/Generated/XRDs_ICSD/')
            roots.append(path + '/Data/Generated/XRDs_ICSD/')
            train_eval_splits += [1 if rruff else 0.9]  # Train on all synthetic data if evaluating on RRUFF
        else:
            icsd = False
            print('Could not find ICSD CIF files. Using open-access CIFs instead.')

    if open_access or not icsd:
        if not os.path.exists(path + '/Data/Generated/XRDs_open_access/'):
            if not os.path.exists(path + '/Data/Generated/CIFs_open_access/'):
                download(path + '/Data/Generated/', 'CIFs_open_access/')
            generate(path + '/Data/Generated/CIFs_open_access/', path + '/Data/Generated/XRDs_open_access/')
        roots.append(path + '/Data/Generated/XRDs_open_access/')
        train_eval_splits += [1 if rruff else 0.9]  # Train on all synthetic data if evaluating on RRUFF

    return roots, train_eval_splits


if __name__ == '__main__':
    ML.launch(task='NPCNN')
