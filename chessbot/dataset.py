import math
import random

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from chessbot.environment import LOSS_DATASET, LOSS_VAL_DATASET, WIN_DATASET, WIN_VAL_DATASET


class DeepChessDataset(Sequence):
    def __init__(self, win, loss, batch_size, sample_size=1_000_000):
        self.x = np.empty(shape=(sample_size, 2, 773), dtype=bool)
        self.y = np.empty(shape=(sample_size, 2), dtype=int)
        self.win = win
        self.loss = loss
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x_batch, y_batch

    def on_epoch_end(self):
        win_sample = self.win.sample(self.sample_size)["bitboard"]
        loss_sample = self.loss.sample(self.sample_size)["bitboard"]
        for i, (x_win, x_loss) in enumerate(zip(win_sample.values, loss_sample.values)):
            label = random.choice(((1, 0), (0, 1)))
            position_pair = (x_win, x_loss) if label == (1, 0) else (x_loss, x_win)

            self.x[i] = position_pair
            self.y[i] = label


class Pos2VecDataset(Sequence):
    def __init__(self, win, loss, batch_size):
        self.x = np.vstack(pd.concat((win.sample(1_000_000), loss.sample(1_000_000)))["bitboard"].to_numpy())
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        x_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return x_batch, x_batch

    def on_epoch_end(self):
        np.random.shuffle(self.x)


def deepchess_dataset():
    win = pd.read_pickle(WIN_DATASET)
    loss = pd.read_pickle(LOSS_DATASET)

    win_val = pd.read_pickle(WIN_VAL_DATASET)
    loss_val = pd.read_pickle(LOSS_VAL_DATASET)

    return DeepChessDataset(win, loss, 50), DeepChessDataset(win_val, loss_val, 1_000, 1_000)


def pos2vec_dataset():
    win = pd.read_pickle(WIN_DATASET)
    loss = pd.read_pickle(LOSS_DATASET)

    win_val = pd.read_pickle(WIN_VAL_DATASET)
    loss_val = pd.read_pickle(LOSS_VAL_DATASET)

    return Pos2VecDataset(win, loss, 50), Pos2VecDataset(win_val, loss_val, 50)
