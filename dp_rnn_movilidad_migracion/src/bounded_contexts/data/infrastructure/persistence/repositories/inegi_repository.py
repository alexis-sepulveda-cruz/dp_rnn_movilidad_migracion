"""
Implementación del repositorio para datos de INEGI.
"""
import os
from typing import Dict, Any
import pandas as pd

from dp_rnn_movilidad_migracion.src.shared.domain.ports.logger_port import LoggerPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.inegi_schema import (
    NUMERIC_FEATURES,
    INEGI_STATIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_FEATURES_FLAT,
    INEGI_DERIVED_FEATURES
)


class InegiRepository(DataRepository):
    """
    Repositorio para cargar datos de INEGI.

    Implementa la interfaz DataRepository para proporcionar
    acceso a datos de INEGI sin realizar transformaciones.
    """

    def __init__(
        self,
        logger: LoggerPort,
        inegi_path: str,
        inegi_file: str
    ):
        """
        Inicializa el repositorio de INEGI.

        Args:
            logger: Instancia del logger para registrar eventos.
            inegi_path: Ruta base de los archivos de INEGI.
            inegi_file: Nombre del archivo de INEGI.
            data_processor: Servicio de transformación de datos.
        """
        self.logger = logger
        self.inegi_path = inegi_path
        self.inegi_file = inegi_file

        self.logger.info("Repositorio de INEGI inicializado")
        self.logger.debug(f"Ruta configurada: {self.inegi_path}/{self.inegi_file}")

    def load_data(self) -> pd.DataFrame:
        """
        Carga datos de INEGI desde un archivo CSV y aplica transformaciones.

        Returns:
            DataFrame con los datos de INEGI procesados.
        """
        self.logger.info(f"Cargando datos de INEGI desde {self.inegi_path}/{self.inegi_file}")

        # Cargar datos crudos del archivo
        file_path = os.path.join(self.inegi_path, self.inegi_file)
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

        self.logger.debug(f"Archivo cargado. Shape inicial: {df.shape}")

        return df

    def get_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles en INEGI.

        Returns:
            Diccionario con las características y su configuración.
        """
        return {
            'numeric_features': NUMERIC_FEATURES,
            'static_features': INEGI_STATIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'categorical_flat': CATEGORICAL_FEATURES_FLAT,
            'derived_features': INEGI_DERIVED_FEATURES,
            'id_columns': ['ENT', 'NOM_ENT', 'MUN', 'NOM_MUN', 'LOC', 'NOM_LOC']
        }