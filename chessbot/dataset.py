import math
import random

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from chessbot.environment import LOSS_DATASET, LOSS_VAL_DATASET, WIN_DATASET, WIN_VAL_DATASET


class DeepChessDataset(Sequence):
    def __init__(self, batch_size):
        self.x = np.empty(shape=(1_000_000, 2, 773), dtype=bool)
        self.y = np.empty(shape=(1_000_000, 2), dtype=int)
        self.win = pd.read_pickle(WIN_DATASET)
        self.loss = pd.read_pickle(LOSS_DATASET)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x_batch, y_batch

    def on_epoch_end(self):
        win_sample = self.win.sample(1_000_000)["bitboard"]
        loss_sample = self.loss.sample(1_000_000)["bitboard"]
        for i, (x_win, x_loss) in enumerate(zip(win_sample.values, loss_sample.values)):
            label = random.choice(((1, 0), (0, 1)))
            position_pair = (x_win, x_loss) if label == (1, 0) else (x_loss, x_win)

            self.x[i] = position_pair
            self.y[i] = label


class Pos2VecDataset(Sequence):
    def __init__(self, x, batch_size):
        self.x = x
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x_batch, x_batch

    def on_epoch_end(self):
        np.random.shuffle(self.x)


def deepchess_dataset():
    return DeepChessDataset(50)


def pos2vec_dataset():
    win = pd.read_pickle(WIN_DATASET)
    loss = pd.read_pickle(LOSS_DATASET)

    x_train = np.vstack(pd.concat((win.sample(1_000_000), loss.sample(1_000_000)))["bitboard"].to_numpy())

    win_val = pd.read_pickle(WIN_VAL_DATASET)
    loss_val = pd.read_pickle(LOSS_VAL_DATASET)

    x_val = np.vstack(pd.concat((win_val, loss_val))["bitboard"].to_numpy())

    return Pos2VecDataset(x_train, 50), Pos2VecDataset(x_val, 50)
