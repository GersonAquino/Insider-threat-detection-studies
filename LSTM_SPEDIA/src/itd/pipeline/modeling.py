
# Define o modelo LSTM (Long Short-Term Memory; Memória de Curto e Longo Prazo) para sequências por janela
from __future__ import annotations

import tensorflow as tf

from .. import config

@tf.keras.utils.register_keras_serializable(package="itd")
class BinaryF1(tf.keras.metrics.Metric):
    def __init__(self, name: str = "f1", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp = tf.reduce_sum(sw * (y_true * y_pred))
            fp = tf.reduce_sum(sw * ((1.0 - y_true) * y_pred))
            fn = tf.reduce_sum(sw * (y_true * (1.0 - y_pred)))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        denom = (2.0 * self.tp) + self.fp + self.fn + tf.keras.backend.epsilon()
        return (2.0 * self.tp) / denom

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

@tf.keras.utils.register_keras_serializable(package="itd")
class BinaryBalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name: str = "balanced_accuracy", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, tf.float32)

        tp = y_true * y_pred
        tn = (1.0 - y_true) * (1.0 - y_pred)
        fp = (1.0 - y_true) * y_pred
        fn = y_true * (1.0 - y_pred)

        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)
            tp = sw * tp
            tn = sw * tn
            fp = sw * fp
            fn = sw * fn

        self.tp.assign_add(tf.reduce_sum(tp))
        self.tn.assign_add(tf.reduce_sum(tn))
        self.fp.assign_add(tf.reduce_sum(fp))
        self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        tpr = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        tnr = self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())
        return 0.5 * (tpr + tnr)

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

def build_model(
    hp: dict,
    normalizer: tf.keras.layers.Layer,
    window_normalizer: tf.keras.layers.Layer | None = None,
    max_events: int | None = None,
) -> tf.keras.Model:
    in_event = tf.keras.Input(shape=(None,), dtype=tf.int32, name="event_id")
    in_from  = tf.keras.Input(shape=(None,), dtype=tf.int32, name="from_domain_id")
    in_to    = tf.keras.Input(shape=(None,), dtype=tf.int32, name="to_domain_id")
    in_url   = tf.keras.Input(shape=(None,), dtype=tf.int32, name="url_domain_id")
    in_ext   = tf.keras.Input(shape=(None,), dtype=tf.int32, name="file_ext_id")
    in_num   = tf.keras.Input(shape=(None, config.NUM_FEATURES), dtype=tf.float32, name="num")
    in_win   = tf.keras.Input(shape=(config.WIN_FEATURES,), dtype=tf.float32, name="win_num")

    mask = tf.keras.ops.not_equal(in_event, 0)

    emb_dim_event = hp.get("emb_dim_event", 64)
    emb_dim_aux = hp.get("emb_dim_aux", 16)

    emb_event = tf.keras.layers.Embedding(config.HASH_BINS_EVENT, emb_dim_event, mask_zero=True, name="emb_event")(in_event)
    emb_from  = tf.keras.layers.Embedding(config.HASH_BINS_DOMAIN, emb_dim_aux, mask_zero=True, name="emb_from")(in_from)
    emb_to    = tf.keras.layers.Embedding(config.HASH_BINS_DOMAIN, emb_dim_aux, mask_zero=True, name="emb_to")(in_to)
    emb_url   = tf.keras.layers.Embedding(config.HASH_BINS_URL, emb_dim_aux, mask_zero=True, name="emb_url")(in_url)
    emb_ext   = tf.keras.layers.Embedding(config.HASH_BINS_EXT, emb_dim_aux, mask_zero=True, name="emb_ext")(in_ext)

    num_norm = normalizer(in_num)
    win_norm = window_normalizer(in_win) if window_normalizer is not None else in_win

    x = tf.keras.layers.Concatenate(name="concat")([emb_event, emb_from, emb_to, emb_url, emb_ext, num_norm])

    x = tf.keras.layers.LSTM(
        hp.get("lstm_units", 128),
        dropout=hp.get("dropout", 0.2),
        recurrent_dropout=hp.get("recurrent_dropout", 0.0),
        return_sequences=False,
        name="lstm",
    )(x, mask=mask)

    x = tf.keras.layers.Concatenate(name="merge_window_feats")([x, win_norm])
    x = tf.keras.layers.Dense(hp.get("dense_units", 64), activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(hp.get("dropout", 0.2), name="dense_dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="y")(x)

    model = tf.keras.Model(
        inputs={
            "event_id": in_event,
            "from_domain_id": in_from,
            "to_domain_id": in_to,
            "url_domain_id": in_url,
            "file_ext_id": in_ext,
            "num": in_num,
            "win_num": in_win,
        },
        outputs=out,
        name="itd_lstm_window_binary",
    )

    lr = float(hp.get("learning_rate", 1e-3))
    wd = float(hp.get("weight_decay", 1e-5))
    cn = float(hp.get("clipnorm", 1.0))
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=wd,
            clipnorm=cn,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            BinaryF1(name="f1", threshold=0.5),
            BinaryBalancedAccuracy(name="balanced_accuracy", threshold=0.5),
            tf.keras.metrics.AUC(curve="ROC", name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
        ],
    )
    return model
