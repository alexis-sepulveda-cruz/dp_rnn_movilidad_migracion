
class UncertaintyMetrics:
    """Objeto de valor que encapsula métricas de incertidumbre para una predicción."""

    def __init__(self, reliability_score: float, mean_cv: float, median_cv: float,
                 min_cv: float, max_cv: float, uncertainty_trend: float,
                 high_uncertainty_years: list[int]):
        self.reliability_score = reliability_score
        self.mean_cv = mean_cv
        self.median_cv = median_cv
        self.min_cv = min_cv
        self.max_cv = max_cv
        self.uncertainty_trend = uncertainty_trend
        self.high_uncertainty_years = high_uncertainty_years