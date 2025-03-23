"""
Puerto de preprocesamiento de datos.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class DataPreprocessor(ABC):
    """
    Interfaz para preprocesadores de datos.
    
    Define el contrato que deben cumplir las implementaciones
    de preprocesadores de diferentes fuentes de datos.
    """
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos según las reglas específicas de la fuente.
        
        Args:
            data: DataFrame con los datos crudos
            
        Returns:
            DataFrame procesado
        """
        pass
    
    @abstractmethod
    def get_processed_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles después del preprocesamiento.
        
        Returns:
            Diccionario con las características procesadas y su configuración.
        """
        pass
