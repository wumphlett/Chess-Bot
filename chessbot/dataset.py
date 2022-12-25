import math

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

from chessbot.environment import LOSS_DATASET, LOSS_VAL_DATASET, WIN_DATASET, WIN_VAL_DATASET


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


def pos2vec_dataset():
    win = pd.read_pickle(WIN_DATASET)
    loss = pd.read_pickle(LOSS_DATASET)

    x_train = np.vstack(pd.concat((win.sample(1_000_000), loss.sample(1_000_000)))["bitboard"].to_numpy())

    win_val = pd.read_pickle(WIN_VAL_DATASET)
    loss_val = pd.read_pickle(LOSS_VAL_DATASET)

    x_val = np.vstack(pd.concat((win_val, loss_val))["bitboard"].to_numpy())

    return Pos2VecDataset(x_train, 50), Pos2VecDataset(x_val, 500)
