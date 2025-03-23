from abc import ABC, abstractmethod
import typing as t
import numpy as np


class ModelTrainerPort(ABC):
    """Puerto para entrenar modelos de predicción."""

    @abstractmethod
    def train(self, model: t.Any, X: np.ndarray, y: np.ndarray,
              validation_split: float, epochs: int, batch_size: int) -> t.Any:
        """
        Entrena un modelo con los datos proporcionados.

        Args:
            model: Modelo a entrenar
            X: Datos de entrada
            y: Datos de salida
            validation_split: Fracción de datos para validación
            epochs: Número de épocas
            batch_size: Tamaño del lote

        Returns:
            Historial del entrenamiento
        """
        pass