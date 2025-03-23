from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.model_builder_port import \
    ModelBuilderPort
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


class TensorflowRNNModelBuilder(ModelBuilderPort):
    """Implementación de ModelBuilderPort usando TensorFlow."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def _set_seed(self):
        """Establece semillas aleatorias para reproducibilidad."""
        tf.random.set_seed(self.random_seed)
        tf.keras.utils.set_random_seed(self.random_seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except:
            pass

    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        Construye un modelo RNN con TensorFlow.

        Args:
            input_shape: Forma de los datos de entrada

        Returns:
            Modelo TensorFlow construido
        """
        # Establecer configuración determinística
        tf.keras.backend.clear_session()
        self._set_seed()

        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            BatchNormalization(),
            Dropout(0.15),

            LSTM(64),
            BatchNormalization(),
            Dropout(0.15),

            Dense(32, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            Dense(1)
        ])

        # Optimizador con learning rate reducido
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0003,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model