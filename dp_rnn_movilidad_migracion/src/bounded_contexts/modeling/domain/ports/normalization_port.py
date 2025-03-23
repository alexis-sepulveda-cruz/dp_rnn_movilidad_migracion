"""
Puerto para normalización de datos.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class NormalizationPort(ABC):
    """
    Puerto para normalización de datos.
    
    Define el contrato para normalizar datos para modelos.
    """
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Entrena el normalizador y transforma los datos.
        
        Args:
            data: DataFrame a normalizar
            
        Returns:
            Array numpy con datos normalizados
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transforma datos usando el normalizador ya entrenado.
        
        Args:
            data: DataFrame a normalizar
            
        Returns:
            Array numpy con datos normalizados
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Desnormaliza datos.
        
        Args:
            data: Array numpy a desnormalizar
            
        Returns:
            Array numpy con datos desnormalizados
        """
        pass
