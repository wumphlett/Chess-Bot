from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD

MODEL_DIR = Path("./models").resolve()
POS2VEC_WEIGHTS = MODEL_DIR / "pos2vec.h5"

POS2VEC_LAYERS = [600, 400, 200, 100]


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
                Dense(773, activation="relu", name="encode_1"),
            ]
        )
        self.decode = Sequential(
            [
                Dense(773, activation="sigmoid", name="decode_4"),
            ]
        )

    def call(self, x, **kwargs):
        return self.decode(self.encode(x))

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch == 0:
            return lr
        return lr * 0.98


def train_pos2vec(x_train, x_val):
    tf.keras.backend.clear_session()

    ae = AutoEncoder()
    # autoencoder greedy layer-wise pretraining
    for i, size in enumerate(POS2VEC_LAYERS):
        print(f"Training layer {i+1}/{len(POS2VEC_LAYERS)}")

        ae.encode.add(Dense(size, activation="relu", name=f"encode_{i+2}"))
        if i != 0:
            layers = ae.decode.layers
            ae.decode = Sequential([Dense(POS2VEC_LAYERS[i-1], activation="relu", name=f"decode_{4-i}")] + layers)

        ae.compile(optimizer=Adam(learning_rate=0.005), loss=BinaryCrossentropy(), jit_compile=True)
        ae.fit(x_train, epochs=200, callbacks=[LearningRateScheduler(AutoEncoder.lr_schedule)], workers=8, validation_data=x_val)

        for j, layer in enumerate(ae.encode.layers):
            layer.trainable = False
        for j, layer in enumerate(ae.decode.layers):
            layer.trainable = False

    ae.encode.save_weights(POS2VEC_WEIGHTS)

    return Pos2Vec()
