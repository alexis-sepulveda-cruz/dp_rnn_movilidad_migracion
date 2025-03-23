"""
Implementación del repositorio para datos de CONAPO.
"""

from pathlib import Path
from typing import Dict, Any, List
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
        conapo_file: str,
        start_year: int,
        end_year: int,
        include_derived: bool = False,
        include_targets: bool = False
    ):
        """
        Inicializa el repositorio de CONAPO.

        Args:
            logger: Instancia del logger para registrar eventos.
            conapo_path: Ruta base de los archivos de CONAPO.
            conapo_file: Nombre del archivo de CONAPO.
            start_year: Año inicial para filtrar los datos.
            end_year: Año final para filtrar los datos.
            include_derived: Indicador para incluir características derivadas.
            include_targets: Indicador para incluir variables objetivo.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.conapo_path = conapo_path
        self.conapo_file = conapo_file
        self.start_year = start_year
        self.end_year = end_year
        self.include_derived = include_derived
        self.include_targets = include_targets

        self.logger.info("Repositorio de CONAPO inicializado")
        self.logger.debug(f"Ruta configurada: {self.conapo_path}/{self.conapo_file}")
        self.logger.debug(f"Años: {self.start_year}-{self.end_year}")

    def load_data(self) -> pd.DataFrame:
        """
        Carga datos de CONAPO desde un archivo Excel.

        La configuración se obtiene directamente del contenedor de inyección de dependencias
        configurado durante la inicialización del repositorio.

        Returns:
            DataFrame con los datos de CONAPO.
        """
        file_path = Path(self.conapo_path) / self.conapo_file
        self.logger.info(f"Cargando datos de CONAPO desde {file_path}")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            self.logger.debug(f"Datos cargados exitosamente: {df.shape} filas")
        except Exception as e:
            self.logger.error(f"Error al cargar datos de CONAPO: {str(e)}")
            raise

        # Aplicar filtros según la configuración
        df = self._apply_filters(df)

        # Seleccionar columnas
        id_columns = ['AÑO', 'CVE_GEO', 'ENTIDAD']
        selected_columns = id_columns + self._get_feature_columns()

        self.logger.info(f"Datos de CONAPO procesados: {df.shape} filas, {len(selected_columns)} columnas")
        return df[selected_columns]

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

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica filtros al DataFrame según la configuración.

        Args:
            df: DataFrame a filtrar.

        Returns:
            DataFrame filtrado.
        """
        # Filtrar por rango de años
        self.logger.debug(f"Filtrando datos por años: {self.start_year} - {self.end_year}")
        df = df[
            (df['AÑO'] >= self.start_year) &
            (df['AÑO'] <= self.end_year)
        ]

        # Filtrar República Mexicana (total nacional)
        df = df[df['ENTIDAD'] != 'República Mexicana']

        return df

    def _get_feature_columns(self) -> List[str]:
        """
        Determina qué columnas de características incluir según la configuración.

        Returns:
            Lista de nombres de columnas a incluir.
        """
        features = list(TEMPORAL_FEATURES)  # Copiar para no modificar la lista original

        # Incluir características derivadas si se solicita
        if self.include_derived:
            self.logger.debug("Incluyendo características derivadas")
            features.extend([f for f in CONAPO_DERIVED_FEATURES if f not in features])

        # Incluir variables objetivo si se solicita
        if self.include_targets:
            self.logger.debug("Incluyendo variables objetivo")
            features.extend([f for f in CONAPO_TARGET_VARIABLES if f not in features])

        return features