import pandas as pd
import numpy as np

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
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.prediction_generator_port import \
    PredictionGeneratorPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.uncertainty_analyzer_port import \
    UncertaintyAnalyzerPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import \
    PredictionRequestDTO
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_result_dto import \
    PredictionResultDTO
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.conapo_schema import TEMPORAL_FEATURES
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.inegi_schema import INEGI_STATIC_FEATURES
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import UncertaintyMetrics


class MigrationPredictionService:
    """Servicio de aplicación para predecir migración."""

    def __init__(
        self, model_builder: ModelBuilderPort,
        model_trainer: ModelTrainerPort,
        data_preparer: DataPreparationPort,
        prediction_repository: PredictionRepositoryPort,
        visualizer: VisualizationPort,
        prediction_generator: PredictionGeneratorPort = None,
        uncertainty_analyzer: UncertaintyAnalyzerPort = None
    ):
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.model_builder = model_builder
        self.model_trainer = model_trainer
        self.data_preparer = data_preparer
        self.prediction_repository = prediction_repository
        self.visualizer = visualizer
        self.prediction_generator = prediction_generator
        self.uncertainty_analyzer = uncertainty_analyzer
        self.model = None
        
        self.logger.info("Servicio de predicción de migración inicializado")

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

    def predict_migration(self, request: PredictionRequestDTO, 
                         temporal_data: pd.DataFrame, 
                         static_data: pd.DataFrame) -> PredictionResultDTO:
        """
        Predice la migración para un estado usando simulación Monte Carlo.

        Args:
            request: DTO con parámetros de predicción
            temporal_data: DataFrame con datos temporales históricos (CONAPO)
            static_data: DataFrame con datos estáticos (INEGI)

        Returns:
            DTO con resultados de predicción
        """
        if not self.model:
            error_msg = "El modelo no ha sido entrenado"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Validar parámetros de entrada
        if not request.validate():
            error_msg = f"Parámetros de solicitud inválidos para {request.state}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Prediciendo migración para {request.state} "
                        f"({request.future_years} años, {request.monte_carlo_samples} muestras)")

        try:
            # 1. Preparar secuencia de predicción
            sequence = self.data_preparer.prepare_prediction_sequence(
                entity_id=request.state,
                temporal_data=temporal_data,
                static_data=static_data,
                sequence_length=self.model.get_sequence_length()
            )

            # 2. Ejecutar simulaciones Monte Carlo utilizando el generador de predicciones
            self.logger.info(f"Iniciando simulación Monte Carlo con {request.monte_carlo_samples} muestras")
            all_trajectories, uncertainty_data = self.prediction_generator.generate_predictions(
                model=self.model,
                initial_sequence=sequence,
                future_steps=request.future_years,
                num_samples=request.monte_carlo_samples
            )

            # 3. Calcular estadísticas y métricas de incertidumbre usando el analizador
            predictions = self.uncertainty_analyzer.calculate_prediction_statistics(
                all_trajectories=all_trajectories,
                base_year=temporal_data[temporal_data['ENTIDAD'] == request.state]['AÑO'].max(),
                confidence_level=request.confidence_level
            )

            # 4. Analizar incertidumbre
            uncertainty_metrics = self.uncertainty_analyzer.analyze_uncertainty(
                predictions=predictions, 
                uncertainty_data=uncertainty_data
            )

            # 5. Crear entidad de dominio con el resultado
            result = self._create_prediction_result(request.state, predictions, uncertainty_metrics)

            # 6. Persistir predicción
            self.prediction_repository.save_prediction(result)
            
            # 7. Visualizar resultados si se solicita
            if hasattr(request, 'visualize') and request.visualize:
                self.visualizer.plot_predictions_with_uncertainty(result)

            # 8. Retornar DTO
            self.logger.info(f"Predicción para {request.state} completada con éxito")
            return PredictionResultDTO.from_domain(result)
            
        except Exception as e:
            self.logger.error(f"Error al predecir migración para {request.state}: {str(e)}")
            raise

    def _create_prediction_result(self, state: str, predictions: pd.DataFrame, 
                                uncertainty_metrics: UncertaintyMetrics) -> PredictionResult:
        """
        Crea una entidad de dominio PredictionResult.
        
        Args:
            state: Nombre del estado
            predictions: DataFrame con predicciones
            uncertainty_metrics: Métricas de incertidumbre
            
        Returns:
            Entidad PredictionResult con datos completos
        """
        return PredictionResult(
            state=state,
            years=predictions['AÑO'].tolist(),
            values=predictions['CRE_NAT'].tolist(),
            lower_bounds=predictions['CRE_NAT_lower'].tolist(),
            upper_bounds=predictions['CRE_NAT_upper'].tolist(),
            std_devs=predictions['CRE_NAT_std'].tolist(),
            uncertainty_metrics=uncertainty_metrics
        )