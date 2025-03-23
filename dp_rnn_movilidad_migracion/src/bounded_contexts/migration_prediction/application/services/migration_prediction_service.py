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

            # 2. Ejecutar simulaciones Monte Carlo
            self.logger.info(f"Iniciando simulación Monte Carlo con {request.monte_carlo_samples} muestras")
            all_trajectories, uncertainty_data = self._run_monte_carlo_simulations(
                sequence=sequence,
                future_years=request.future_years,
                mc_samples=request.monte_carlo_samples
            )

            # 3. Calcular estadísticas y métricas de incertidumbre
            predictions = self._calculate_prediction_statistics(
                all_trajectories=all_trajectories,
                base_year=temporal_data[temporal_data['ENTIDAD'] == request.state]['AÑO'].max(),
                confidence_level=request.confidence_level
            )

            # 4. Analizar incertidumbre y generar reporte
            uncertainty_metrics = self._analyze_uncertainty(predictions, uncertainty_data)

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

    def _run_monte_carlo_simulations(self, sequence: np.ndarray, 
                                    future_years: int, 
                                    mc_samples: int) -> tuple:
        """
        Ejecuta simulaciones Monte Carlo para estimar incertidumbre.
        
        Args:
            sequence: Secuencia inicial para predicción
            future_years: Número de años a predecir
            mc_samples: Número de simulaciones Monte Carlo
            
        Returns:
            Tupla con (matriz de trayectorias, datos de incertidumbre)
        """
        # Almacenar todas las trayectorias Monte Carlo
        all_trajectories = np.zeros((mc_samples, future_years))
        current_sequence = sequence.copy()
        uncertainty_data = []
        
        batch_size = min(20, mc_samples)  # Procesar en batches para eficiencia
        
        for year in range(future_years):
            self.logger.debug(f"Prediciendo año {year+1} de {future_years}")
            
            # Realizar múltiples predicciones en lotes
            predictions_batches = []
            
            for i in range(0, mc_samples, batch_size):
                batch_end = min(i + batch_size, mc_samples)
                batch_size_actual = batch_end - i
                sequence_batch = np.repeat([current_sequence], batch_size_actual, axis=0)
                
                # Usar el modelo para predecir
                batch_predictions = self.model.predict(sequence_batch)
                predictions_batches.append(batch_predictions)
            
            # Combinar predicciones de todos los lotes
            predictions = np.concatenate(predictions_batches).flatten()
            
            # Monitorear varianza para estabilidad
            pred_std = predictions.std()
            pred_mean = predictions.mean()
            cv = (pred_std / abs(pred_mean)) if abs(pred_mean) > 1e-6 else 0
            
            uncertainty_data.append({
                'year': year,
                'mean': pred_mean,
                'std': pred_std,
                'cv': cv
            })
            
            # Control de varianza para predicciones futuras
            if cv > 0.2:  # Limitar varianza extrema
                self.logger.warning(f"Varianza alta detectada en año {year+1}, aplicando corrección")
                scaled_std = abs(pred_mean) * 0.2
                predictions = np.random.normal(pred_mean, scaled_std, mc_samples)
            
            # Almacenar predicciones para este año
            all_trajectories[:, year] = predictions
            
            # Actualizar secuencia para el siguiente paso
            current_sequence = self._update_sequence(current_sequence, pred_mean)
        
        return all_trajectories, uncertainty_data
    
    def _update_sequence(self, current_sequence: np.ndarray, new_value: float) -> np.ndarray:
        """
        Actualiza la secuencia desplazándola y añadiendo el nuevo valor.
        
        Args:
            current_sequence: Secuencia actual
            new_value: Nuevo valor a añadir
            
        Returns:
            Secuencia actualizada
        """
        # Crear copia para evitar modificar el original
        updated_sequence = current_sequence.copy()
        
        # Desplazar valores (eliminar el más antiguo)
        updated_sequence = np.roll(updated_sequence, -1, axis=0)
        
        # Añadir nuevo valor en la última posición
        feature_count = updated_sequence.shape[1]
        if feature_count > 1:
            # Si hay múltiples características, mantener las estáticas
            static_features = updated_sequence[-1, 1:]
            updated_sequence[-1] = np.concatenate([[new_value], static_features])
        else:
            updated_sequence[-1] = new_value
            
        return updated_sequence
    
    def _calculate_prediction_statistics(self, all_trajectories: np.ndarray, 
                                       base_year: int,
                                       confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Calcula estadísticas de las predicciones incluyendo intervalos de confianza.
        
        Args:
            all_trajectories: Matriz con todas las trayectorias Monte Carlo
            base_year: Año base desde el que parten las predicciones
            confidence_level: Nivel de confianza para los intervalos
            
        Returns:
            DataFrame con años y estadísticas de predicción
        """
        # Calcular estadísticas básicas
        future_years = all_trajectories.shape[1]
        mean_predictions = np.mean(all_trajectories, axis=0)
        std_predictions = np.std(all_trajectories, axis=0)
        
        # Calcular percentiles para intervalos de confianza
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        lower_bound = np.percentile(all_trajectories, lower_percentile, axis=0)
        upper_bound = np.percentile(all_trajectories, upper_percentile, axis=0)
        
        # Desnormalizar predicciones si es necesario
        if hasattr(self.data_preparer, 'denormalize_target'):
            mean_predictions = self.data_preparer.denormalize_target(mean_predictions.reshape(-1, 1)).flatten()
            std_predictions = self.data_preparer.denormalize_target(std_predictions.reshape(-1, 1)).flatten()
            lower_bound = self.data_preparer.denormalize_target(lower_bound.reshape(-1, 1)).flatten()
            upper_bound = self.data_preparer.denormalize_target(upper_bound.reshape(-1, 1)).flatten()
        
        # Crear DataFrame con resultados
        years = range(base_year + 1, base_year + future_years + 1)
        
        return pd.DataFrame({
            'AÑO': years,
            'CRE_NAT': mean_predictions,
            'CRE_NAT_std': std_predictions,
            'CRE_NAT_lower': lower_bound,
            'CRE_NAT_upper': upper_bound
        })
    
    def _analyze_uncertainty(self, predictions: pd.DataFrame, 
                           uncertainty_data: list) -> UncertaintyMetrics:
        """
        Analiza la incertidumbre en las predicciones y calcula métricas.
        
        Args:
            predictions: DataFrame con predicciones
            uncertainty_data: Lista de diccionarios con datos de incertidumbre
            
        Returns:
            Objeto UncertaintyMetrics con métricas de incertidumbre
        """
        # Calcular CV (coeficiente de variación) para cada año
        abs_values = predictions['CRE_NAT'].abs()
        cv_values = (predictions['CRE_NAT_std'] / (abs_values + 1e-6)) * 100
        
        # Identificar años con alta incertidumbre (CV > 45%)
        high_uncertainty_years = predictions.loc[cv_values > 45, 'AÑO'].tolist()
        
        # Calcular score de confiabilidad (función exponencial inversa del CV medio)
        mean_cv = cv_values.mean()
        reliability_score = 100 * np.exp(-mean_cv / 50)
        
        # Calcular tendencia de incertidumbre
        uncertainty_trend = cv_values.diff().mean()
        
        # Crear objeto de valor con métricas
        return UncertaintyMetrics(
            reliability_score=reliability_score,
            mean_cv=mean_cv,
            median_cv=cv_values.median(),
            min_cv=cv_values.min(),
            max_cv=cv_values.max(),
            uncertainty_trend=uncertainty_trend,
            high_uncertainty_years=high_uncertainty_years
        )
    
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