"""
Puerto para el análisis de incertidumbre en predicciones.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any

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

    @abstractmethod
    def generate_reliability_report(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un reporte detallado de confiabilidad para predicciones.
        
        Analiza las métricas de incertidumbre presentes en las predicciones
        y genera un reporte estructurado con múltiples indicadores de
        confiabilidad, distribución de incertidumbre y análisis de outliers.
        
        Args:
            predictions: DataFrame con predicciones enriquecidas con métricas 
                        de incertidumbre (debe incluir columnas CV, is_outlier)
                        
        Returns:
            Diccionario con métricas organizadas por categorías
        """
        pass
    
    @abstractmethod
    def print_detailed_report(self, report: Dict[str, Any]) -> None:
        """
        Imprime un reporte detallado de confiabilidad en formato legible.
        
        Args:
            report: Diccionario con el reporte generado por generate_reliability_report
        """
        pass
