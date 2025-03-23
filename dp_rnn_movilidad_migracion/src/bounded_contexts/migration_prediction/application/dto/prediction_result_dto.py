from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult


class PredictionResultDTO:
    """DTO para devolver resultados de predicción."""

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
        """Crea un DTO a partir de una entidad de dominio."""
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