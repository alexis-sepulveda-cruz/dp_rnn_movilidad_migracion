import pandas as pd

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.model_builder_port import \
    ModelBuilderPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.model_trainer_port import \
    ModelTrainerPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.data_preparation_port import \
    DataPreparationPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.prediction_repository_port import \
    PredictionRepositoryPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.visualization_port import \
    VisualizationPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import \
    PredictionRequestDTO
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_result_dto import \
    PredictionResultDTO
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.conapo_schema import TEMPORAL_FEATURES
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.inegi_schema import INEGI_STATIC_FEATURES
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class MigrationPredictionService:
    """Servicio de aplicación para predecir migración."""

    def __init__(
        self, model_builder: ModelBuilderPort,
        model_trainer: ModelTrainerPort,
        data_preparer: DataPreparationPort,
        prediction_repository: PredictionRepositoryPort,
        visualizer: VisualizationPort
    ):
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.model_builder = model_builder
        self.model_trainer = model_trainer
        self.data_preparer = data_preparer
        self.prediction_repository = prediction_repository
        self.visualizer = visualizer
        self.model = None

    def train_model(self, temporal_data: pd.DataFrame, static_data: pd.DataFrame,
                    sequence_length: int = 5, validation_split: float = 0.2,
                    epochs: int = 100, batch_size: int = 128) -> None:
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            temporal_data: DataFrame con datos temporales
            static_data: DataFrame con datos estáticos
            sequence_length: Longitud de secuencia
            validation_split: Fracción de datos para validación
            epochs: Número de épocas
            batch_size: Tamaño del lote
        """
        self.logger.info("Preparando datos para entrenamiento")
        X, y = self.data_preparer.prepare_model_data(
            temporal_data=temporal_data,
            static_data=static_data,
            target_column='CRE_NAT',
            sequence_length=sequence_length,
            temporal_features=TEMPORAL_FEATURES,
            static_features=INEGI_STATIC_FEATURES
        )

        # Verificar que los datos están correctamente formados
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Número de muestras inconsistente: X={X.shape[0]}, y={y.shape[0]}")

        self.logger.info("Construyendo modelo")
        self.model = self.model_builder.build_model(X.shape[1:])

        self.logger.info("Entrenando modelo")
        history = self.model_trainer.train(
            self.model, X, y, validation_split, epochs, batch_size
        )

        self.logger.info(f"Entrenamiento completado. Pérdida final: {history.history['loss'][-1]}")

    def predict_migration(self, request: PredictionRequestDTO) -> PredictionResultDTO:
        """
        Predice la migración para un estado.

        Args:
            request: DTO con parámetros de predicción

        Returns:
            DTO con resultados de predicción
        """
        if not self.model:
            raise ValueError("El modelo no ha sido entrenado")

        self.logger.info(f"Prediciendo migración para {request.state}")

        # Implementar lógica de predicción con Monte Carlo
        # ...

        # Crear objeto de dominio con el resultado
        result = PredictionResult(...)

        # Persistir predicción
        self.prediction_repository.save_prediction(result)

        # Visualizar si se solicita
        if request.visualize:
            self.visualizer.plot_predictions_with_uncertainty(result)

        # Crear y devolver DTO con el resultado
        return PredictionResultDTO.from_domain(result)