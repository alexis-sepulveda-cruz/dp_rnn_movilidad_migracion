from abc import ABC, abstractmethod
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult


class PredictionRepositoryPort(ABC):
    """Puerto para persistir y recuperar predicciones."""

    @abstractmethod
    def save_prediction(self, prediction: PredictionResult) -> None:
        """
        Guarda una predicción.

        Args:
            prediction: Resultado de predicción a guardar
        """
        pass

    @abstractmethod
    def get_prediction(self, state: str) -> PredictionResult:
        """
        Recupera una predicción para un estado.

        Args:
            state: Nombre del estado

        Returns:
            Resultado de predicción recuperado
        """
        pass