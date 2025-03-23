"""
Implementación del repositorio para datos de CONAPO.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.conapo_schema import (
    TEMPORAL_FEATURES,
    CONAPO_DERIVED_FEATURES,
    CONAPO_TARGET_VARIABLES
)
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class ConapoRepository(DataRepository):
    """
    Repositorio para cargar datos de CONAPO.

    Implementa la interfaz DataRepository para proporcionar
    acceso a datos de CONAPO.
    """

    def __init__(
        self,
        conapo_path: str,
        conapo_file: str
    ):
        """
        Inicializa el repositorio de CONAPO.

        Args:
            conapo_path: Ruta base de los archivos de CONAPO.
            conapo_file: Nombre del archivo de CONAPO.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.conapo_path = conapo_path
        self.conapo_file = conapo_file

        self.logger.info("Repositorio de CONAPO inicializado")
        self.logger.debug(f"Ruta configurada: {self.conapo_path}/{self.conapo_file}")

    def load_data(self) -> pd.DataFrame:
        """
        Carga datos de CONAPO desde un archivo Excel.

        Returns:
            DataFrame con los datos de CONAPO sin procesar.
        """
        file_path = Path(self.conapo_path) / self.conapo_file
        self.logger.info(f"Cargando datos de CONAPO desde {file_path}")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            self.logger.debug(f"Datos cargados exitosamente: {df.shape} filas")
            return df
        except Exception as e:
            self.logger.error(f"Error al cargar datos de CONAPO: {str(e)}")
            raise

    def get_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles en CONAPO.

        Returns:
            Diccionario con las características y su configuración.
        """
        return {
            'temporal_features': TEMPORAL_FEATURES,
            'derived_features': CONAPO_DERIVED_FEATURES,
            'target_variables': CONAPO_TARGET_VARIABLES,
            'id_columns': ['AÑO', 'CVE_GEO', 'ENTIDAD']
        }