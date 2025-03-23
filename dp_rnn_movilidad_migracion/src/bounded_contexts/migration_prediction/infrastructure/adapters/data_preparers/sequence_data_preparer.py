from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.data_preparation_port import \
    DataPreparationPort
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SequenceDataPreparer(DataPreparationPort):
    """Implementación de DataPreparationPort para preparar secuencias temporales."""

    def __init__(self, temporal_features: list[str], static_features: list[str],
                 sequence_length: int = 5):
        self.temporal_features = temporal_features
        self.static_features = static_features
        self.sequence_length = sequence_length

        # Inicializar escaladores
        self.scaler_temporal = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_static = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df_temporal: pd.DataFrame,
                     df_static: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara los datos combinando series temporales con características estáticas.

        Args:
            df_temporal: DataFrame con datos temporales
            df_static: DataFrame con datos estáticos

        Returns:
            Tupla con datos de entrada (X) y salida (y) preparados para el modelo
        """
        # Asegurar que los datos estén ordenados
        df_temporal = df_temporal.sort_values(['ENTIDAD', 'AÑO'])

        # Normalizar características de entrada
        temporal_data = self.scaler_temporal.fit_transform(
            df_temporal[self.temporal_features]
        )

        # Normalizar variable objetivo por separado
        target_data = self.scaler_target.fit_transform(
            df_temporal[['CRE_NAT']].values
        )

        # Normalizar características estáticas
        static_data = self.scaler_static.fit_transform(
            df_static[self.static_features]
        )

        # Validar datos
        if np.isnan(temporal_data).any():
            raise ValueError("Temporal data contains NaN values")
        if np.isnan(static_data).any():
            raise ValueError("Static data contains NaN values")

        return self._create_sequences(temporal_data, target_data, static_data,
                                      df_temporal, df_static['NOM_ENT'])

    def _create_sequences(self, temporal_data: np.ndarray,
                          target_data: np.ndarray,
                          static_data: np.ndarray,
                          df_temporal: pd.DataFrame,
                          state_names: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias para el entrenamiento.

        Args:
            temporal_data: Datos temporales normalizados
            target_data: Datos objetivo normalizados
            static_data: Datos estáticos normalizados
            df_temporal: DataFrame original de datos temporales
            state_names: Nombres de los estados

        Returns:
            Tupla con datos de entrada (X) y salida (y) preparados para el modelo
        """
        # Crear diccionario para mapear estados a sus características estáticas
        static_features_dict = {
            state: features
            for state, features in zip(state_names, static_data)
        }

        X, y = [], []
        for state in df_temporal['ENTIDAD'].unique():
            state_mask = df_temporal['ENTIDAD'] == state
            state_indices = np.where(state_mask)[0]

            # Obtener datos temporales y objetivo del estado
            state_temporal = temporal_data[state_indices]
            state_target = target_data[state_indices]

            # Obtener características estáticas del estado
            static_state_features = static_features_dict.get(
                state, np.zeros(static_data.shape[1])
            )

            for i in range(len(state_temporal) - self.sequence_length):
                # Secuencia temporal
                seq = state_temporal[i:(i + self.sequence_length)]

                # Combinar con características estáticas
                seq_with_static = np.column_stack([
                    seq,
                    np.tile(static_state_features, (self.sequence_length, 1))
                ])

                X.append(seq_with_static)
                y.append(state_target[i + self.sequence_length])

        return np.array(X), np.array(y)

    def prepare_prediction_sequence(self, state: str, df_temporal: pd.DataFrame,
                                    df_static: pd.DataFrame,
                                    sequence_length: int) -> np.ndarray:
        """
        Prepara una secuencia para predicción.

        Args:
            state: Nombre del estado
            df_temporal: DataFrame con datos temporales
            df_static: DataFrame con datos estáticos
            sequence_length: Longitud de la secuencia

        Returns:
            Secuencia preparada para predicción
        """
        # Preparar datos iniciales
        state_temporal = df_temporal[df_temporal['ENTIDAD'] == state].sort_values('AÑO')
        last_sequence = state_temporal.iloc[-sequence_length:]
        state_static = df_static[df_static['NOM_ENT'] == state]

        # Transformar datos
        temporal_data = self.scaler_temporal.transform(
            last_sequence[self.temporal_features]
        )
        static_data = self.scaler_static.transform(
            state_static[self.static_features]
        )

        # Combinar datos
        sequence_with_static = np.column_stack([
            temporal_data,
            np.tile(static_data[0], (sequence_length, 1))
        ])

        return sequence_with_static