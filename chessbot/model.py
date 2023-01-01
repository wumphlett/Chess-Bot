from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Concatenate, Dense, Input, LeakyReLU
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD

MODEL_DIR = Path("./models").resolve()
DEEPCHESS_WEIGHTS = MODEL_DIR / "deepchess.h5"
POS2VEC_WEIGHTS = MODEL_DIR / "pos2vec.h5"

POS2VEC_LAYERS = [600, 400, 200, 100]


class DeepChess(Model):
    def __init__(self, load_weights=False):
        super().__init__()

        pos2vec = Pos2Vec(load_weights=not load_weights)
        left_input, right_input = Input(shape=(773,), name="dc_input_left"), Input(shape=(773,), name="dc_input_right")
        join = Concatenate(axis=1)([pos2vec(left_input), pos2vec(right_input)])
        dc_1 = Dense(400, activation="relu", name="dc_1")(join)
        dc_2 = Dense(200, activation="relu", name="dc_2")(dc_1)
        dc_3 = Dense(100, activation="relu", name="dc_3")(dc_2)
        dc_4 = Dense(2, activation="softmax", name="dc_4")(dc_3)

        self.deepchess = Model(inputs=[left_input, right_input], outputs=dc_4)

        if load_weights:
            print(f"DeepChess weights loading from {DEEPCHESS_WEIGHTS}")
            self.deepchess.load_weights(DEEPCHESS_WEIGHTS)

    def call(self, inputs, **kwargs):
        return self.deepchess(inputs)

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch == 0:
            return lr
        return lr * 0.99


class Pos2Vec(Model):
    def __init__(self, load_weights=False):
        super().__init__()
        self.encoder = Sequential(
            [
                Input(shape=(773,), name="p2v_input"),
                Dense(773, activation="relu", name="p2v_1"),
                Dense(600, activation="relu", name="p2v_2"),
                Dense(400, activation="relu", name="p2v_3"),
                Dense(200, activation="relu", name="p2v_4"),
                Dense(100, activation="relu", name="p2v_5"),
            ],
            name="pos2vec",
        )

        if load_weights:
            print(f"Pos2Vec weights loading from {POS2VEC_WEIGHTS}")
            self.encoder.load_weights(POS2VEC_WEIGHTS)

    def call(self, inputs, **kwargs):
        return self.encoder(inputs)


class AutoEncoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = Sequential(
            [
                Input(shape=(773,)),
                Dense(773, activation="relu", name="encoder_1"),
            ],
            name="ae_encoder",
        )
        self.decoder = Sequential(
            [
                Dense(773, activation="sigmoid", name="decoder_4"),
            ],
            name="ae_decoder",
        )

    def call(self, inputs, **kwargs):
        # TODO cache result of predict
        return self.decoder(self.encoder(inputs))

    @staticmethod
    def lr_schedule(epoch, lr):
        if epoch == 0:
            return lr
        return lr * 0.98


def get_deepchess():
    tf.keras.backend.clear_session()

    dc = DeepChess(load_weights=True)
    dc.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"], jit_compile=True)

    return dc


def train_deepchess(train, val=None):
    tf.keras.backend.clear_session()

    dc = DeepChess()

    dc.compile(
        optimizer=Adam(learning_rate=0.01), loss=CategoricalCrossentropy(), metrics=["accuracy"], jit_compile=True
    )
    dc.fit(
        train, epochs=1_000, callbacks=[LearningRateScheduler(DeepChess.lr_schedule)], workers=8, validation_data=val
    )

    dc.deepchess.save_weights(DEEPCHESS_WEIGHTS)

    return get_deepchess()


def get_pos2vec():
    tf.keras.backend.clear_session()

    p2v = Pos2Vec(load_weights=True)
    p2v.compile(optimizer=SGD(), loss=MeanSquaredError(), jit_compile=True)

    return p2v


def train_pos2vec(train, val=None):
    tf.keras.backend.clear_session()

    ae = AutoEncoder()
    # autoencoder greedy layer-wise pretraining
    for i, size in enumerate(POS2VEC_LAYERS):
        print(f"Training layer {i+1}/{len(POS2VEC_LAYERS)}")

        ae.encoder.add(Dense(size, activation="relu", name=f"encoder_{i+2}"))
        if i != 0:
            layers = ae.decoder.layers
            ae.decoder = Sequential(
                [Dense(POS2VEC_LAYERS[i - 1], activation="relu", name=f"decoder_{4-i}")] + layers, name="ae_decoder"
            )

        ae.compile(optimizer=SGD(learning_rate=0.005), loss=MeanSquaredError(), metrics=["binary_accuracy"], jit_compile=True)
        ae.fit(
            train, epochs=100, callbacks=[LearningRateScheduler(AutoEncoder.lr_schedule)], workers=8, validation_data=val
        )

        for j, layer in enumerate(ae.encoder.layers):
            if j == len(ae.encoder.layers) - 1:
                break
            layer.trainable = False
        for j, layer in enumerate(ae.decoder.layers):
            layer.trainable = False
    ae.encoder.save_weights(POS2VEC_WEIGHTS)

    return get_pos2vec()
