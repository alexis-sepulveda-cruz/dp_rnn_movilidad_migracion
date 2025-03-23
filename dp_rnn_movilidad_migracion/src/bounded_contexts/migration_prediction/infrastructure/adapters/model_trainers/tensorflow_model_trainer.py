from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.model_trainer_port import \
    ModelTrainerPort
import tensorflow as tf
import numpy as np


class TensorflowModelTrainer(ModelTrainerPort):
    """Implementación de ModelTrainerPort usando TensorFlow."""

    def train(self, model: tf.keras.Model, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2, epochs: int = 100,
              batch_size: int = 128) -> tf.keras.callbacks.History:
        """
        Entrena un modelo TensorFlow con los datos proporcionados.

        Args:
            model: Modelo TensorFlow a entrenar
            X: Datos de entrada
            y: Datos de salida
            validation_split: Fracción de datos para validación
            epochs: Número de épocas
            batch_size: Tamaño del lote

        Returns:
            Historial del entrenamiento
        """
        return model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=25,
                    restore_best_weights=True,
                    min_delta=0.0001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=15,
                    min_lr=0.000001
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1
                )
            ]
        )