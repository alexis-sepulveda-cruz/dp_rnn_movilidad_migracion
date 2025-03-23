from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import UncertaintyMetrics
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory

class PredictionResult:
    """
    Entidad que representa el resultado de una predicción de migración.
    
    Esta clase encapsula todos los aspectos de una predicción, incluyendo
    los valores pronosticados, intervalos de confianza, y métricas de
    incertidumbre asociadas. Proporciona una representación completa
    del resultado de un modelo predictivo.
    
    Attributes:
        state (str): Nombre o identificador del estado/entidad para 
            el que se realizó la predicción.
            
        years (list[int]): Lista de años para los que se realizaron
            predicciones, en orden cronológico.
            
        values (list[float]): Valores predichos para cada año, donde
            values[i] corresponde al pronóstico para years[i].
            
        lower_bounds (list[float]): Límites inferiores del intervalo
            de confianza para cada predicción.
            
        upper_bounds (list[float]): Límites superiores del intervalo
            de confianza para cada predicción.
            
        std_devs (list[float]): Desviaciones estándar asociadas con
            cada valor predicho, cuantificando la incertidumbre directa.
            
        uncertainty_metrics (UncertaintyMetrics): Objeto de valor que
            contiene métricas agregadas de incertidumbre para toda
            la predicción.
    """

    def __init__(self, state: str, years: list[int], values: list[float],
                 lower_bounds: list[float], upper_bounds: list[float],
                 std_devs: list[float], uncertainty_metrics: UncertaintyMetrics):
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.state = state
        self.years = years
        self.values = values
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.std_devs = std_devs
        self.uncertainty_metrics = uncertainty_metrics
        
        self.logger.info(f"Predicción creada para {state}, periodo {min(years)}-{max(years)}")
        self.logger.debug(f"Fiabilidad: {uncertainty_metrics.reliability_score:.2f}, "
                         f"CV medio: {uncertainty_metrics.mean_cv:.2f}%")
        if uncertainty_metrics.high_uncertainty_years:
            self.logger.warning(f"Años con alta incertidumbre detectados: {uncertainty_metrics.high_uncertainty_years}")

    def is_reliable(self) -> bool:
        """
        Determina si la predicción es confiable basado en métricas de incertidumbre.
        
        Evalúa la confiabilidad de la predicción utilizando el score de confiabilidad
        de las métricas de incertidumbre. Una predicción se considera confiable si
        su score supera el umbral del 70%.
        
        Returns:
            bool: True si la predicción es considerada confiable, False en caso contrario.
        """
        is_reliable = self.uncertainty_metrics.reliability_score > 70
        
        if is_reliable:
            self.logger.info(f"Predicción para {self.state} considerada confiable "
                           f"(score: {self.uncertainty_metrics.reliability_score:.2f})")
        else:
            self.logger.warning(f"Predicción para {self.state} NO es confiable "
                              f"(score: {self.uncertainty_metrics.reliability_score:.2f})")
        
        return is_reliable