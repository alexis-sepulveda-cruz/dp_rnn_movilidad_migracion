from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import UncertaintyMetrics


class PredictionResult:
    """Entidad que representa el resultado de una predicción."""

    def __init__(self, state: str, years: list[int], values: list[float],
                 lower_bounds: list[float], upper_bounds: list[float],
                 std_devs: list[float], uncertainty_metrics: UncertaintyMetrics):
        self.state = state
        self.years = years
        self.values = values
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.std_devs = std_devs
        self.uncertainty_metrics = uncertainty_metrics

    def is_reliable(self) -> bool:
        """Determina si la predicción es confiable basado en métricas de incertidumbre."""
        return self.uncertainty_metrics.reliability_score > 70