"""
Puerto para el análisis de incertidumbre en predicciones.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import UncertaintyMetrics


class UncertaintyAnalyzerPort(ABC):
    """
    Interfaz para analizadores de incertidumbre.
    
    Abstrae los métodos para analizar y cuantificar la incertidumbre
    en predicciones probabilísticas.
    """
    
    @abstractmethod
    def analyze_uncertainty(
        self, 
        predictions: pd.DataFrame, 
        uncertainty_data: List[dict]
    ) -> UncertaintyMetrics:
        """
        Analiza la incertidumbre en las predicciones y genera métricas.
        
        Args:
            predictions: DataFrame con predicciones estadísticas
            uncertainty_data: Datos adicionales de incertidumbre por paso
            
        Returns:
            Objeto UncertaintyMetrics con las métricas calculadas
        """
        pass
    
    @abstractmethod
    def calculate_prediction_statistics(
        self,
        all_trajectories: any,
        base_year: int,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calcula estadísticas a partir de múltiples trayectorias de predicción.
        
        Args:
            all_trajectories: Matriz con múltiples trayectorias de predicción
            base_year: Año base desde donde parten las predicciones
            confidence_level: Nivel de confianza para los intervalos
            
        Returns:
            DataFrame con años y estadísticas calculadas
        """
        pass
    
    @abstractmethod
    def calculate_detailed_metrics(
        self,
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calcula métricas detalladas de incertidumbre para cada punto de predicción.
        
        Enriquece el DataFrame de predicciones con métricas adicionales como niveles
        de incertidumbre, detección de outliers y puntuaciones z.
        
        Args:
            predictions: DataFrame con predicciones
            
        Returns:
            DataFrame enriquecido con métricas adicionales
        """
        pass
