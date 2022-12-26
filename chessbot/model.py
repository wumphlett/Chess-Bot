from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD

MODEL_DIR = Path("./models").resolve()
POS2VEC_WEIGHTS = MODEL_DIR / "pos2vec.h5"

POS2VEC_LAYERS = [600, 400, 200, 100]


class Pos2Vec(Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(
            [
                Input(shape=(773,)),
                Dense(600, activation="relu", name="encoder_1"),
                Dense(400, activation="relu", name="encoder_2"),
                Dense(200, activation="relu", name="encoder_3"),
                Dense(100, activation="relu", name="encoder_4"),
            ], name="p2v_encoder"
        )

        if POS2VEC_WEIGHTS.exists():
            print(f"Pos2Vec weights found at {POS2VEC_WEIGHTS}, loading")
            self.encoder.load_weights(POS2VEC_WEIGHTS)

    def call(self, x, **kwargs):
        return self.encoder(x)


class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(
            [
                Input(shape=(773,)),
            ], name="ae_encoder"
        )
        self.decoder = Sequential(
            [
                Dense(773, activation="sigmoid", name="decoder_4"),
            ], name="ae_decoder"
        )

    def call(self, x, **kwargs):
        return self.decoder(self.encoder(x))

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch == 0:
            return lr
        return lr * 0.98


def train_pos2vec(train, val):
    tf.keras.backend.clear_session()

    ae = AutoEncoder()
    # autoencoder greedy layer-wise pretraining
    for i, size in enumerate(POS2VEC_LAYERS):
        print(f"Training layer {i+1}/{len(POS2VEC_LAYERS)}")

        ae.encoder.add(Dense(size, activation="relu", name=f"encoder_{i+1}"))
        if i != 0:
            layers = ae.decoder.layers
            ae.decoder = Sequential([Dense(POS2VEC_LAYERS[i-1], activation="relu", name=f"decoder_{4-i}")] + layers)

        ae.compile(optimizer=SGD(learning_rate=0.005), loss=BinaryCrossentropy(), jit_compile=True)
        ae.fit(train, epochs=200, callbacks=[LearningRateScheduler(AutoEncoder.lr_schedule)], workers=8, validation_data=val)

        ae.encoder.summary()
        ae.decoder.summary()

        for j, layer in enumerate(ae.encoder.layers):
            layer.trainable = False
        for j, layer in enumerate(ae.decoder.layers):
            layer.trainable = False
    ae.encode.save_weights(POS2VEC_WEIGHTS)

    return Pos2Vec()
