"""
Implementación del preprocesador para datos de INEGI.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_preprocessor import DataPreprocessor
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.inegi_schema import (
    NUMERIC_FEATURES,
    INEGI_STATIC_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_FEATURES_FLAT,
    INEGI_DERIVED_FEATURES
)
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class InegiPreprocessor(DataPreprocessor):
    """
    Preprocesador para datos de INEGI.
    
    Implementa el puerto DataPreprocessor para transformar datos
    específicos de INEGI, incluyendo agregación por estado y 
    cálculo de tasas e índices.
    """
    
    def __init__(self):
        """Inicializa el preprocesador de INEGI."""
        self.logger = LoggerFactory.get_composite_logger(__name__)
        
        # Inicializar escaladores
        self.scaler_temporal = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_static = MinMaxScaler(feature_range=(0, 1))
        
        # Inicializar encoders para variables categóricas
        self._initialize_categorical_encoders()
        
        self.logger.info("Preprocesador de INEGI inicializado")
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa los datos de INEGI, incluyendo agregación por estado.
        
        Args:
            df: DataFrame con datos crudos de INEGI a nivel localidad
            
        Returns:
            DataFrame procesado con datos agregados por estado
        """
        self.logger.info("Iniciando preprocesamiento de datos INEGI")
        
        # Reemplazar valores faltantes
        df = df.replace({'*': np.nan, 'Nulo': np.nan})
        
        # Convertir columnas numéricas
        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Preprocesar variables categóricas
        df = self._preprocess_categorical(df)
        
        # Definir funciones de agregación
        agg_functions = {
            # Variables numéricas que se suman
            **{col: 'sum' for col in NUMERIC_FEATURES[:12]},
            # Variables que se promedian
            'GRAPROES': 'mean',
            # Variables categóricas procesadas
            **self._get_categorical_agg_functions(df)
        }
   
        # Agregar por entidad
        self.logger.info("Agregando datos por entidad federativa")
        df_state = df.groupby(['ENT', 'NOM_ENT']).agg(agg_functions).reset_index()
        
        # Calcular tasas e índices
        self._calculate_rates(df_state)
        
        self.logger.info(f"Preprocesamiento completado. Shape final: {df_state.shape}")
        return df_state
    
    def get_processed_feature_names(self) -> Dict[str, Any]:
        """
        Obtiene los nombres de las características disponibles después del preprocesamiento.
        
        Returns:
            Diccionario con las características procesadas y su configuración.
        """
        derived_features = INEGI_DERIVED_FEATURES + [
            'TASA_DESEMPLEO', 
            'TASA_INACTIVIDAD', 
            'TASA_ANALFABETISMO', 
            'TASA_INDIGENA', 
            'TASA_CARENCIA_SERVICIOS',
            'INDICE_INFRAESTRUCTURA',
            'INDICE_CONFLICTOS'
        ]
        
        return {
            'numeric_features': NUMERIC_FEATURES,
            'static_features': INEGI_STATIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'categorical_flat': CATEGORICAL_FEATURES_FLAT,
            'derived_features': derived_features,
            'id_columns': ['ENT', 'NOM_ENT']
        }
    
    def _preprocess_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa variables categóricas antes de la agregación.
        
        Args:
            df: DataFrame con variables categóricas crudas
            
        Returns:
            DataFrame con variables categóricas procesadas
        """
        # Procesar variables binarias
        for col, mapping in CATEGORICAL_FEATURES['binary'].items():
            df[col] = df[col].map(mapping)
        
        # Procesar variables ordinales
        for col, config in CATEGORICAL_FEATURES['ordinal'].items():
            if 'mapping' in config:
                df[col] = df[col].map(config['mapping'])
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Procesar variables nominales (one-hot encoding)
        for prefix, source_col in CATEGORICAL_FEATURES['nominal_prefixes'].items():
            dummy_cols = pd.get_dummies(df[source_col], prefix=prefix, dummy_na=True)
            df = pd.concat([df, dummy_cols], axis=1)
        
        return df
    
    def _get_categorical_agg_functions(self, df: pd.DataFrame) -> dict:
        """
        Define funciones de agregación para variables categóricas procesadas.

        Args:
            df: DataFrame con las columnas categóricas procesadas

        Returns:
            dict: Diccionario con las funciones de agregación para cada columna
        """
        agg_functions = {}

        # Para variables one-hot: proporción de cada categoría
        for prefix in CATEGORICAL_FEATURES['nominal_prefixes'].keys():
            one_hot_cols = [col for col in df.columns if col.startswith(prefix)]
            for col in one_hot_cols:
                agg_functions[col] = 'mean'
        
        # Para variables binarias
        binary_columns = set(CATEGORICAL_FEATURES['binary'].keys()).intersection(df.columns)
        for col in binary_columns:
            agg_functions[col] = 'mean'
        
        # Para variables ordinales
        ordinal_columns = set(CATEGORICAL_FEATURES['ordinal'].keys()).intersection(df.columns)
        for col in ordinal_columns:
            # Para ordinales usamos la mediana como medida representativa
            agg_functions[col] = 'median'
        
        return agg_functions
    
    def _calculate_rates(self, df: pd.DataFrame) -> None:
        """
        Calcula tasas e índices para los datos agregados.

        Args:
            df: DataFrame con datos agregados por estado
        """
        total_pob = df['PEA'] + df['PE_INAC']
        total_viviendas = (df['VPH_NDEAED'] + df['VPH_S_ELEC'] + 
                          df['VPH_AGUAFV'] + df['VPH_NODREN'])
        
        # Calcular tasas básicas
        df['TASA_DESEMPLEO'] = (df['PDESOCUP'] / df['PEA'] * 100)
        df['TASA_INACTIVIDAD'] = (df['PE_INAC'] / total_pob * 100)
        df['TASA_ANALFABETISMO'] = (df['P15YM_AN'] / total_pob * 100)
        df['TASA_INDIGENA'] = (df['P3YM_HLI'] / total_pob * 100)
        df['TASA_CARENCIA_SERVICIOS'] = (
            (df['VPH_NDEAED'] + df['VPH_S_ELEC'] + 
             df['VPH_AGUAFV'] + df['VPH_NODREN']) / total_viviendas * 100
        )
        
        # Calcular índices compuestos para variables categóricas
        df['INDICE_INFRAESTRUCTURA'] = df[list(CATEGORICAL_FEATURES['binary'].keys()) + 
                                        list(CATEGORICAL_FEATURES['ordinal'].keys())].mean(axis=1)
        df['INDICE_CONFLICTOS'] = df[[col for col in df.columns if col.startswith('C_CONFLICTO')]].mean(axis=1)
    
    def _initialize_categorical_encoders(self):
        """Inicializa encoders para variables categóricas con manejo de errores."""
        try:
            self.categorical_encoders = {
                'actividad': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                'trabajo': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                'conflictos': LabelEncoder(),
                'problema': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                'infraestructura': LabelEncoder()
            }
        except Exception as e:
            self.logger.error(f"Error al inicializar encoders categóricos: {str(e)}")
            raise RuntimeError(f"Error al inicializar encoders categóricos: {str(e)}")
