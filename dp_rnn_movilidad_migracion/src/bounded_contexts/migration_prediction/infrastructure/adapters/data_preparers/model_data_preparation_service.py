"""
Servicio para preparar datos para modelos predictivos.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.data_preparation_port import DataPreparationPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.normalization_port import NormalizationPort
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.shared.domain.services.state_name_normalizer import StateNameNormalizer


class ModelDataPreparationService(DataPreparationPort):
    """
    Servicio para preparar datos para modelos predictivos.
    
    Implementa la lógica para normalizar y secuenciar datos
    para su uso en entrenamiento de modelos.
    """
    
    def __init__(
        self,
        temporal_normalizer: NormalizationPort,
        target_normalizer: NormalizationPort,
        static_normalizer: NormalizationPort,
        random_seed: int = 42
    ):
        """
        Inicializa el servicio de preparación de datos.
        
        Args:
            temporal_normalizer: Normalizador para datos temporales
            target_normalizer: Normalizador para variables objetivo
            static_normalizer: Normalizador para datos estáticos
            random_seed: Semilla aleatoria para reproducibilidad
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.temporal_normalizer = temporal_normalizer
        self.target_normalizer = target_normalizer
        self.static_normalizer = static_normalizer
        self.random_seed = random_seed
        
        # Configurar semilla para reproducibilidad
        np.random.seed(random_seed)
        self.logger.info("Servicio de preparación de datos inicializado")
        
    def prepare_model_data(
        self, 
        temporal_data: pd.DataFrame, 
        static_data: pd.DataFrame,
        target_column: str,
        sequence_length: int,
        temporal_features: Optional[List[str]] = None,
        static_features: Optional[List[str]] = None,
        id_column: str = 'ENTIDAD',
        time_column: str = 'AÑO'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento de modelos.
        
        Args:
            temporal_data: DataFrame con datos temporales (series históricas)
            static_data: DataFrame con datos estáticos (características constantes)
            target_column: Nombre de la columna objetivo a predecir
            sequence_length: Longitud de secuencia para datos temporales
            temporal_features: Lista de características temporales a usar
            static_features: Lista de características estáticas a usar
            id_column: Nombre de la columna de identificación (entidad)
            time_column: Nombre de la columna temporal (año)
            
        Returns:
            Tupla con datos de entrada (X) y salida (y) preparados para el modelo
        """
        self.logger.info("Iniciando preparación de datos para modelo")
        
        # Asegurar que los datos estén ordenados
        temporal_data = temporal_data.sort_values([id_column, time_column])
        
        # Seleccionar columnas relevantes si se especifican
        if temporal_features is not None:
            temporal_features_df = temporal_data[temporal_features]
        else:
            temporal_features_df = temporal_data.drop([id_column, time_column, target_column], axis=1)
            temporal_features = temporal_features_df.columns.tolist()
        
        if static_features is not None:
            static_features_df = static_data[static_features]
        else:
            static_features_df = static_data.drop(['NOM_ENT'], axis=1)
            static_features = static_features_df.columns.tolist()
        
        self.logger.info(f"Usando {len(temporal_features)} características temporales y {len(static_features)} características estáticas")
        
        # Normalizar los datos
        temporal_normalized = self.temporal_normalizer.fit_transform(temporal_features_df)
        target_normalized = self.target_normalizer.fit_transform(temporal_data[[target_column]])
        static_normalized = self.static_normalizer.fit_transform(static_features_df)
        
        # Validar que no hay NaNs
        if np.isnan(temporal_normalized).any():
            self.logger.error("Datos temporales contienen valores NaN")
            raise ValueError("Temporal data contains NaN values")
        if np.isnan(static_normalized).any():
            self.logger.error("Datos estáticos contienen valores NaN")
            raise ValueError("Static data contains NaN values")
        
        # Crear secuencias para entrenamiento
        X, y = self._create_sequences(
            temporal_normalized, 
            target_normalized, 
            static_normalized,
            temporal_data, 
            static_data['NOM_ENT'].values, 
            sequence_length,
            id_column
        )
        
        self.logger.info(f"Preparación completada. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def _create_sequences(
        self,
        temporal_data: np.ndarray,
        target_data: np.ndarray,
        static_data: np.ndarray,
        df_temporal: pd.DataFrame,
        state_names: np.ndarray,
        sequence_length: int,
        id_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias para el entrenamiento.

        Args:
            temporal_data: Datos temporales normalizados
            target_data: Datos objetivo normalizados
            static_data: Datos estáticos normalizados
            df_temporal: DataFrame original temporal para mapeo de estados
            state_names: Nombres de los estados 
            sequence_length: Longitud de la secuencia
            id_column: Nombre de la columna de identificación (entidad)

        Returns:
            Tupla con datos de entrada (X) y salida (y) preparados para el modelo
        """
        # Crear diccionario para mapear estados a sus características estáticas
        static_features_dict = {
            state: features 
            for state, features in zip(state_names, static_data)
        }
        
        X, y = [], []
        for state in df_temporal[id_column].unique():
            state_mask = df_temporal[id_column] == state
            state_indices = np.where(state_mask)[0]
            
            # Obtener datos temporales y objetivo del estado
            state_temporal = temporal_data[state_indices]
            state_target = target_data[state_indices]
            
            # Obtener características estáticas del estado
            static_state_features = static_features_dict.get(
                state, np.zeros(static_data.shape[1])
            )
            
            for i in range(len(state_temporal) - sequence_length):
                # Secuencia temporal
                seq = state_temporal[i:(i + sequence_length)]
                
                # Combinar con características estáticas
                seq_with_static = np.column_stack([
                    seq,
                    np.tile(static_state_features, (sequence_length, 1))
                ])
                
                X.append(seq_with_static)
                y.append(state_target[i + sequence_length])
        
        return np.array(X), np.array(y)

    def prepare_prediction_sequence(
        self, 
        entity_id: str, 
        temporal_data: pd.DataFrame,
        static_data: pd.DataFrame, 
        sequence_length: int,
        temporal_features: Optional[List[str]] = None,
        static_features: Optional[List[str]] = None,
        id_column: str = 'ENTIDAD',
        time_column: str = 'AÑO'
    ) -> np.ndarray:
        """
        Prepara una secuencia para predicción.
        
        Args:
            entity_id: Identificador de la entidad (ej: nombre del estado)
            temporal_data: DataFrame con datos temporales
            static_data: DataFrame con datos estáticos
            sequence_length: Longitud de la secuencia
            temporal_features: Lista de características temporales a usar
            static_features: Lista de características estáticas a usar
            id_column: Nombre de la columna de identificación (entidad)
            time_column: Nombre de la columna temporal (año)
            
        Returns:
            Secuencia preparada para predicción
        """
        self.logger.info(f"Preparando secuencia de predicción para {entity_id}")
        
        # Filtrar y ordenar datos temporales para la entidad específica
        entity_temporal = temporal_data[temporal_data[id_column] == entity_id].sort_values(time_column)
        
        if entity_temporal.empty:
            raise ValueError(f"No se encontraron datos temporales para {entity_id}")
            
        # Tomar la última secuencia disponible
        last_sequence = entity_temporal.tail(sequence_length)
        
        if len(last_sequence) < sequence_length:
            self.logger.warning(f"Datos insuficientes para {entity_id}. Se requieren {sequence_length} registros, encontrados {len(last_sequence)}")
            raise ValueError(f"Datos temporales insuficientes para {entity_id}")
        
        # Obtener nombre oficial para buscar en datos estáticos
        official_name = StateNameNormalizer.to_official_name(entity_id)
        
        # Usar tanto el nombre original como el oficial para la búsqueda
        search_names = [entity_id]
        if official_name:
            search_names.append(official_name)
            
        # Filtrar datos estáticos para la entidad usando nombres alternativos
        entity_static = None
        for name in search_names:
            matches = static_data[static_data['NOM_ENT'] == name]
            if not matches.empty:
                entity_static = matches
                self.logger.debug(f"Encontrada coincidencia de datos estáticos con nombre: {name}")
                break
        
        if entity_static is None or entity_static.empty:
            self.logger.error(f"No se encontraron datos estáticos para {entity_id} ni sus aliases")
            self.logger.debug(f"Nombres de estados disponibles: {static_data['NOM_ENT'].unique()}")
            raise ValueError(f"No se encontraron datos estáticos para {entity_id}")
        
        # Seleccionar características relevantes
        if temporal_features is None:
            temporal_features_df = last_sequence.drop([id_column, time_column], axis=1, errors='ignore')
        else:
            temporal_features_df = last_sequence[temporal_features]
            
        if static_features is None:
            static_features_df = entity_static.drop(['NOM_ENT'], axis=1, errors='ignore')
        else:
            static_features_df = entity_static[static_features]
            
        # Normalizar los datos usando los normalizadores ya entrenados
        temporal_normalized = self.temporal_normalizer.transform(temporal_features_df)
        static_normalized = self.static_normalizer.transform(static_features_df)
        
        # Combinar datos temporales y estáticos
        sequence_with_static = np.column_stack([
            temporal_normalized,
            np.tile(static_normalized[0], (sequence_length, 1))
        ])
        
        self.logger.info(f"Secuencia preparada con forma {sequence_with_static.shape}")
        return sequence_with_static
