"""
Módulo que define las interfaces para repositorios de datos.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import pandas as pd


class DataRepository(ABC):
    """
    Interfaz para repositorios de datos.

    Define el contrato que deben implementar todos los repositorios
    que proporcionan acceso a datos externos.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Carga datos desde una fuente externa.

        Los parámetros de configuración se obtienen directamente del contenedor
        de inyección de dependencias y no necesitan ser pasados como argumentos.

        Returns:
            DataFrame con los datos cargados.
        """
        pass

    @abstractmethod
    def get_raw_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles en los datos crudos.

        Returns:
            Diccionario con los nombres de las características originales.
        """
        pass