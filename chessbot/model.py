from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

MODEL_DIR = Path("./models").resolve()
POS2VEC_WEIGHTS = MODEL_DIR / "pos2vec.h5"

POS2VEC_LAYERS = [600, 400, 200, 100]


# TODO mixed policy?


class Pos2Vec(Model):
    def __init__(self):
        super().__init__()
        self.encode = Sequential(
            [
                Dense(773, activation="relu"),
                Dense(600, activation="relu"),
                Dense(400, activation="relu"),
                Dense(200, activation="relu"),
                Dense(100, activation="relu"),
            ]
        )

        if POS2VEC_WEIGHTS.exists():
            self.build((None, 773))
            self.encode.load_weights(POS2VEC_WEIGHTS)

    def call(self, x, **kwargs):
        return self.encode(x)


class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        self.encode = Sequential(
            [
                Dense(773, activation="relu"),
            ]
        )
        self.decode = Sequential(
            [
                Dense(773, activation="sigmoid"),
            ]
        )

    def call(self, x, **kwargs):
        return self.decode(self.encode(x))

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch == 0:
            return lr
        return lr * 0.98


def train_pos2vec(x_train):
    tf.keras.backend.clear_session()

    ae = AutoEncoder()
    # autoencoder greedy layer-wise pretraining
    for i, size in enumerate(POS2VEC_LAYERS):
        print(f"Training layer {i+1}/{len(POS2VEC_LAYERS)}")

        for j, layer in enumerate(ae.encode.layers):
            layer.trainable = False
        ae.encode.add(Dense(size, activation="relu"))

        if i != 0:
            for j, layer in enumerate(ae.decode.layers):
                layer.trainable = False
            layers = ae.decode.layers
            ae.decode = Sequential([Dense(POS2VEC_LAYERS[i-1], activation="relu")] + layers)

        ae.compile(optimizer=Adam(learning_rate=0.005), loss=BinaryCrossentropy(), jit_compile=True)
        ae.fit(x_train, epochs=200, callbacks=[LearningRateScheduler(AutoEncoder.lr_schedule)])
    ae.encode.save_weights(POS2VEC_WEIGHTS)

    return Pos2Vec()
