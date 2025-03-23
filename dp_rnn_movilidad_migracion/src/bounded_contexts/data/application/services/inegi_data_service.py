"""
Servicio de datos INEGI.
"""
import pandas as pd
from typing import Dict, Any

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_preprocessor import DataPreprocessor
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class InegiDataService:
    """
    Servicio para orquestar la carga y procesamiento de datos de INEGI.
    
    Este servicio coordina el uso del repositorio para cargar datos
    y el preprocesador para transformarlos.
    """
    
    def __init__(
        self,
        repository: DataRepository,
        preprocessor: DataPreprocessor
    ):
        """
        Inicializa el servicio con sus dependencias.
        
        Args:
            inegi_repository: Implementación del repositorio de datos INEGI
            inegi_preprocessor: Implementación del preprocesador de datos INEGI
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.inegi_repository = repository
        self.inegi_preprocessor = preprocessor
        
        self.logger.info("Servicio de datos INEGI inicializado")
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Obtiene los datos de INEGI procesados y listos para usar.
        
        Este método orquesta la carga de datos crudos mediante el repositorio
        y su posterior procesamiento con el preprocesador.
        
        Returns:
            DataFrame con los datos de INEGI procesados
        """
        self.logger.info("Obteniendo datos procesados de INEGI")
        
        # Cargar datos crudos
        raw_data = self.inegi_repository.load_data()
        
        # Preprocesar datos
        processed_data = self.inegi_preprocessor.preprocess(raw_data)
        
        self.logger.info(f"Datos procesados de INEGI obtenidos. Shape: {processed_data.shape}")
        return processed_data
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Obtiene metadatos de las características de los datos.
        
        Returns:
            Diccionario con metadatos de las características
        """
        # Combinamos metadatos del repositorio y del preprocesador
        repo_features = self.inegi_repository.get_raw_feature_names()
        processed_features = self.inegi_preprocessor.get_processed_feature_names()
        
        return {
            'raw': repo_features,
            'processed': processed_features
        }
