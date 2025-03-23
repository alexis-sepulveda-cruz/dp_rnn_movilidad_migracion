"""
Puerto para la preparación de datos para modelos de ML.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


class DataPreparationPort(ABC):
    """
    Puerto para la preparación de datos de modelos.
    
    Define el contrato para preparar datos para entrenar modelos predictivos.
    """
    
    @abstractmethod
    def prepare_model_data(
        self, 
        temporal_data: pd.DataFrame, 
        static_data: pd.DataFrame,
        target_column: str,
        sequence_length: int,
        temporal_features: Optional[List[str]] = None,
        static_features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento de modelos.
        
        Args:
            temporal_data: DataFrame con datos temporales (series históricas)
            static_data: DataFrame con datos estáticos (características constantes)
            target_column: Nombre de la columna objetivo a predecir
            sequence_length: Longitud de secuencia para datos temporales
            temporal_features: Lista de características temporales a usar
            static_features: Lista de características estáticas a usar
            
        Returns:
            Tupla con datos de entrada (X) y salida (y) preparados para el modelo
        """
        pass
