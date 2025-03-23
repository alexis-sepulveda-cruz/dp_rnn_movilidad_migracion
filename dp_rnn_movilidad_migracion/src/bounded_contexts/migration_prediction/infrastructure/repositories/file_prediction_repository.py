import os
import json
import pandas as pd
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.prediction_repository_port import \
    PredictionRepositoryPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import \
    UncertaintyMetrics
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory

class FilePredictionRepository(PredictionRepositoryPort):
    """Implementación de PredictionRepositoryPort que guarda predicciones en archivos."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = LoggerFactory.get_composite_logger(__name__)
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Repositorio de predicciones inicializado con directorio: {self.output_dir}")

    def save_prediction(self, prediction: PredictionResult) -> None:
        """
        Guarda una predicción en un archivo CSV y JSON.

        Args:
            prediction: Resultado de predicción a guardar
        """
        # Crear DataFrame
        df = pd.DataFrame({
            'AÑO': prediction.years,
            'CRE_NAT': prediction.values,
            'CRE_NAT_std': prediction.std_devs,
            'CRE_NAT_lower': prediction.lower_bounds,
            'CRE_NAT_upper': prediction.upper_bounds
        })

        # Guardar CSV
        csv_path = os.path.join(self.output_dir, f'{prediction.state}_prediccion.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Predicción guardada en CSV: {csv_path}")

        # Guardar metadatos en JSON
        metadata = {
            'state': prediction.state,
            'reliability_score': prediction.uncertainty_metrics.reliability_score,
            'mean_cv': prediction.uncertainty_metrics.mean_cv,
            'median_cv': prediction.uncertainty_metrics.median_cv,
            'min_cv': prediction.uncertainty_metrics.min_cv,
            'max_cv': prediction.uncertainty_metrics.max_cv,
            'uncertainty_trend': prediction.uncertainty_metrics.uncertainty_trend,
            'high_uncertainty_years': prediction.uncertainty_metrics.high_uncertainty_years
        }

        json_path = os.path.join(self.output_dir, f'{prediction.state}_metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadatos de predicción guardados en JSON: {json_path}")
        self.logger.debug(f"Detalles de fiabilidad: score={metadata['reliability_score']:.2f}, "
                         f"CV medio={metadata['mean_cv']:.2f}")

    def get_prediction(self, state: str) -> PredictionResult:
        """
        Recupera una predicción para un estado desde archivos.

        Args:
            state: Nombre del estado

        Returns:
            Resultado de predicción recuperado
        """
        self.logger.info(f"Intentando recuperar predicción para: {state}")
        
        # Verificar si existen los archivos
        csv_path = os.path.join(self.output_dir, f'{state}_prediccion.csv')
        json_path = os.path.join(self.output_dir, f'{state}_metadata.json')

        if not (os.path.exists(csv_path) and os.path.exists(json_path)):
            self.logger.error(f"No se encontraron archivos de predicción para {state}")
            raise FileNotFoundError(f"No se encontraron predicciones para {state}")

        # Cargar CSV
        self.logger.debug(f"Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)

        # Cargar JSON
        self.logger.debug(f"Cargando metadatos desde: {json_path}")
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Crear objeto de valor UncertaintyMetrics
        uncertainty_metrics = UncertaintyMetrics(
            reliability_score=metadata['reliability_score'],
            mean_cv=metadata['mean_cv'],
            median_cv=metadata['median_cv'],
            min_cv=metadata['min_cv'],
            max_cv=metadata['max_cv'],
            uncertainty_trend=metadata['uncertainty_trend'],
            high_uncertainty_years=metadata['high_uncertainty_years']
        )

        # Crear y devolver entidad PredictionResult
        result = PredictionResult(
            state=state,
            years=df['AÑO'].tolist(),
            values=df['CRE_NAT'].tolist(),
            lower_bounds=df['CRE_NAT_lower'].tolist(),
            upper_bounds=df['CRE_NAT_upper'].tolist(),
            std_devs=df['CRE_NAT_std'].tolist(),
            uncertainty_metrics=uncertainty_metrics
        )
        
        self.logger.info(f"Predicción recuperada correctamente para {state}, "
                        f"años: {min(result.years)}-{max(result.years)}")
        return result