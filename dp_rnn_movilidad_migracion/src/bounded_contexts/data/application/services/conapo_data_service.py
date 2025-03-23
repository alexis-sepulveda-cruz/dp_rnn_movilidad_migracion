"""
Servicio de aplicación para orquestar operaciones con datos de CONAPO.
"""

import pandas as pd

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_preprocessor import DataPreprocessor
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class ConapoDataService:
    """
    Servicio de aplicación para orquestar operaciones con datos de CONAPO.
    
    Coordina la carga y el preprocesamiento de datos de CONAPO.
    """
    
    def __init__(
        self,
        repository: DataRepository,
        preprocessor: DataPreprocessor
    ):
        """
        Inicializa el servicio de datos de CONAPO.
        
        Args:
            repository: Repositorio para cargar datos de CONAPO.
            preprocessor: Preprocesador para datos de CONAPO.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.repository = repository
        self.preprocessor = preprocessor
        
        self.logger.info("Servicio de datos de CONAPO inicializado")
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Obtiene datos de CONAPO procesados.
        
        Este método orquesta la carga y el preprocesamiento de datos.
        
        Returns:
            DataFrame con datos de CONAPO procesados.
        """
        self.logger.info("Iniciando obtención de datos procesados")
        
        # Cargar datos brutos
        raw_data = self.repository.load_data()
        
        # Preprocesar datos
        processed_data = self.preprocessor.preprocess(raw_data)
        
        self.logger.info("Datos procesados obtenidos correctamente")
        return processed_data
    
    def get_feature_names(self) -> dict:
        """
        Obtiene los nombres de las características.
        
        Returns:
            Diccionario con los nombres de las características.
        """
        return self.repository.get_feature_names()
