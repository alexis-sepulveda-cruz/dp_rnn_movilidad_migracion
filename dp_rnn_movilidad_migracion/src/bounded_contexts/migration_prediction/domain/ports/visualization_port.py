from abc import ABC, abstractmethod
from typing import Dict, List, Any

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult


class VisualizationPort(ABC):
    """Puerto para visualizar predicciones."""

    @abstractmethod
    def plot_predictions_with_uncertainty(self, prediction: PredictionResult) -> None:
        """
        Visualiza predicciones con bandas de incertidumbre.

        Args:
            prediction: Resultado de predicción a visualizar
        """
        pass
    
    @abstractmethod
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None) -> None:
        """
        Visualiza el historial de entrenamiento de un modelo.
        
        Args:
            history: Diccionario con histórico de métricas durante el entrenamiento.
                Debe contener al menos las claves 'loss', 'val_loss', 'mae', 'val_mae'.
            save_path: Ruta donde guardar la visualización. Si es None, se usa el directorio predeterminado.
        """
        pass
    
    @abstractmethod
    def plot_state_comparison(self, predictions: Dict[str, PredictionResult]) -> None:
        """
        Visualiza comparación de predicciones entre diferentes estados.
        
        Args:
            predictions: Diccionario con estados como claves y predicciones como valores.
        """
        pass