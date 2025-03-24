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

    @abstractmethod
    def plot_reliability_comparison(self, reliability_scores: Dict[str, float]) -> None:
        """
        Visualiza comparación de scores de confiabilidad entre diferentes estados.
        
        Args:
            reliability_scores: Diccionario con estados como claves y scores de
                confiabilidad como valores.
        """
        pass

    @abstractmethod
    def plot_state_detail(self, prediction: PredictionResult) -> None:
        """
        Visualiza el detalle de predicción para un estado específico.
        
        Crea un gráfico detallado enfocado en un único estado, mostrando
        la evolución de sus predicciones con formato y anotaciones adaptadas
        a la magnitud de los valores.
        
        Args:
            prediction: Resultado de predicción del estado a visualizar
        """
        pass

    @abstractmethod
    def plot_features_distribution(self, features_data: Dict[str, float], title: str = "Distribución de Características", 
                                  save_path: str = None, threshold: float = 0.05) -> None:
        """
        Visualiza la distribución relativa de características como un gráfico circular.
        
        Permite visualizar la importancia relativa o distribución de diferentes 
        características, agrupando automáticamente aquellas con menor peso 
        para mejorar la legibilidad.
        
        Args:
            features_data: Diccionario con nombres de características y sus valores/pesos
            title: Título del gráfico
            save_path: Ruta donde guardar la visualización (opcional)
            threshold: Umbral para agrupar características pequeñas (como porcentaje)
        """
        pass