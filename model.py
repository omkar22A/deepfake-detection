"""
Advanced BiLSTM + MultiHead Attention Model
"""

from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    Add,
    GlobalAveragePooling1D,
    MultiHeadAttention
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(sequence_length=30, feature_dim=1280):

    inputs = Input(shape=(sequence_length, feature_dim))

    x = Bidirectional(
        LSTM(
            256,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        )
    )(inputs)

    x = LayerNormalization()(x)

    x2 = Bidirectional(
        LSTM(
            128,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2
        )
    )(x)

    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.2
    )(x2, x2)

    x = Add()([x2, attention])

    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(0.5)(x)

    x = Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=2e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model