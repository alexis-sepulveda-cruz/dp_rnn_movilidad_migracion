from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult


class PredictionResultDTO:
    """
    Objeto de transferencia de datos (DTO) para resultados de predicción de migración.
    
    Esta clase proporciona una representación simplificada y serializable de los 
    resultados de predicción, facilitando la transferencia de datos entre capas
    de la aplicación sin exponer los detalles internos del dominio.
    
    Attributes:
        state (str): Nombre o identificador del estado/entidad para 
            el que se realizó la predicción.
            
        years (list[int]): Lista de años para los que se realizaron
            predicciones, en orden cronológico.
            
        values (list[float]): Valores predichos para cada año.
        
        lower_bounds (list[float]): Límites inferiores del intervalo
            de confianza para cada predicción.
            
        upper_bounds (list[float]): Límites superiores del intervalo
            de confianza para cada predicción.
            
        std_devs (list[float]): Desviaciones estándar asociadas con
            cada valor predicho.
            
        reliability_score (float): Puntuación de confiabilidad global (0-100),
            extraída de las métricas de incertidumbre.
            
        mean_cv (float): Coeficiente de variación medio, indicando la
            dispersión relativa promedio de las predicciones.
            
        high_uncertainty_years (list[int]): Lista de años con alta incertidumbre,
            identificados a partir de coeficientes de variación elevados.
    """

    def __init__(self, state: str, years: list[int], values: list[float],
                 lower_bounds: list[float], upper_bounds: list[float],
                 std_devs: list[float], reliability_score: float,
                 mean_cv: float, high_uncertainty_years: list[int]):
        self.state = state
        self.years = years
        self.values = values
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.std_devs = std_devs
        self.reliability_score = reliability_score
        self.mean_cv = mean_cv
        self.high_uncertainty_years = high_uncertainty_years

    @classmethod
    def from_domain(cls, prediction: PredictionResult) -> 'PredictionResultDTO':
        """
        Crea un DTO a partir de una entidad de dominio.
        
        Este método de fábrica convierte una entidad de dominio PredictionResult
        en un objeto DTO adecuado para transferencia entre capas, extrayendo
        y aplanando los datos relevantes, especialmente desde el objeto de valor
        UncertaintyMetrics.
        
        Args:
            prediction: Entidad de dominio PredictionResult a convertir
            
        Returns:
            Nueva instancia de PredictionResultDTO con los datos extraídos
        """
        return cls(
            state=prediction.state,
            years=prediction.years,
            values=prediction.values,
            lower_bounds=prediction.lower_bounds,
            upper_bounds=prediction.upper_bounds,
            std_devs=prediction.std_devs,
            reliability_score=prediction.uncertainty_metrics.reliability_score,
            mean_cv=prediction.uncertainty_metrics.mean_cv,
            high_uncertainty_years=prediction.uncertainty_metrics.high_uncertainty_years
        )