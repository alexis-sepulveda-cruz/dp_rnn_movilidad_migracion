"""
Implementación del preprocesador para datos de CONAPO.
"""

from typing import List, Dict, Any
import pandas as pd

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_preprocessor import DataPreprocessor
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.conapo_schema import (
    TEMPORAL_FEATURES,
    CONAPO_DERIVED_FEATURES,
    CONAPO_TARGET_VARIABLES
)
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class ConapoPreprocessor(DataPreprocessor):
    """
    Preprocesador para datos de CONAPO.
    
    Implementa la interfaz DataPreprocessor para proporcionar
    funcionalidad de preprocesamiento específica para datos de CONAPO.
    """
    
    def __init__(
        self,
        start_year: int,
        end_year: int,
        include_derived: bool = False,
        include_targets: bool = False
    ):
        """
        Inicializa el preprocesador de CONAPO.
        
        Args:
            start_year: Año inicial para filtrar los datos.
            end_year: Año final para filtrar los datos.
            include_derived: Indicador para incluir características derivadas.
            include_targets: Indicador para incluir variables objetivo.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.start_year = start_year
        self.end_year = end_year
        self.include_derived = include_derived
        self.include_targets = include_targets
        
        self.logger.info("Preprocesador de CONAPO inicializado")
        self.logger.debug(f"Años: {self.start_year}-{self.end_year}")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de CONAPO.
        
        Args:
            data: DataFrame con datos brutos de CONAPO.
            
        Returns:
            DataFrame preprocesado.
        """
        # Aplicar filtros
        filtered_df = self._apply_filters(data)
        
        # Seleccionar columnas
        id_columns = ['AÑO', 'CVE_GEO', 'ENTIDAD']
        selected_columns = id_columns + self._get_feature_columns()
        
        result_df = filtered_df[selected_columns]
        self.logger.info(f"Datos de CONAPO procesados: {result_df.shape} filas, {len(selected_columns)} columnas")
        return result_df
    
    def get_processed_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles después del preprocesamiento.
        
        Returns:
            Diccionario con las características procesadas y su configuración.
        """
        features = list(TEMPORAL_FEATURES)
        
        # Incluir características derivadas si se solicita
        if self.include_derived:
            features.extend([f for f in CONAPO_DERIVED_FEATURES if f not in features])

        # Incluir variables objetivo si se solicita
        if self.include_targets:
            features.extend([f for f in CONAPO_TARGET_VARIABLES if f not in features])
            
        return {
            'temporal_features': TEMPORAL_FEATURES,
            'derived_features': CONAPO_DERIVED_FEATURES if self.include_derived else [],
            'target_variables': CONAPO_TARGET_VARIABLES if self.include_targets else [],
            'processed_features': features,
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
