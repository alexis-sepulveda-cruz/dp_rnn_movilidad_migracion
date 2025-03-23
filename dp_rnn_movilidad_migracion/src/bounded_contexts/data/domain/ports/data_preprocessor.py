"""
Módulo que define las interfaces para preprocesadores de datos.
"""

from abc import ABC, abstractmethod
import pandas as pd

class DataPreprocessor(ABC):
    """
    Interfaz para preprocesadores de datos.
    
    Define el contrato que deben implementar todos los preprocesadores
    que transforman datos cargados de fuentes externas.
    """
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa un DataFrame según reglas específicas.
        
        Args:
            data: DataFrame a preprocesar.
            
        Returns:
            DataFrame preprocesado.
        """
        pass
