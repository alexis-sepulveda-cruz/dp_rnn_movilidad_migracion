"""
Adaptador para normalización usando scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dp_rnn_movilidad_migracion.src.bounded_contexts.modeling.domain.ports.normalization_port import NormalizationPort
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class SklearnNormalizer(NormalizationPort):
    """
    Implementación de normalización usando sklearn.
    """
    
    def __init__(self, feature_range=(0, 1)):
        """
        Inicializa el normalizador.
        
        Args:
            feature_range: Rango al que normalizar los valores
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.logger.info(f"Normalizador inicializado con rango {feature_range}")
        
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Entrena el normalizador y transforma los datos.
        """
        self.logger.debug(f"Entrenando y transformando datos con shape {data.shape}")
        return self.scaler.fit_transform(data)
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transforma datos usando el normalizador ya entrenado.
        """
        self.logger.debug(f"Transformando datos con shape {data.shape}")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Desnormaliza datos.
        """
        self.logger.debug(f"Desnormalizando datos con shape {data.shape}")
        return self.scaler.inverse_transform(data)
