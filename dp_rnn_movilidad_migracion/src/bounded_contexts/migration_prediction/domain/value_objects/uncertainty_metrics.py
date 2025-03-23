class UncertaintyMetrics:
    """
    Objeto de valor que encapsula métricas de incertidumbre para una predicción.
    
    Esta clase representa una colección inmutable de métricas que cuantifican
    la incertidumbre en predicciones de modelos probabilísticos, especialmente
    útil para evaluar la confiabilidad de predicciones de modelos de migración.
    
    Attributes:
        reliability_score (float): Puntuación de confiabilidad global (0-100).
            Valores más altos indican mayor confiabilidad. Se calcula como una
            función exponencial inversa del coeficiente de variación medio.
            
        mean_cv (float): Coeficiente de variación medio de todas las predicciones.
            Representa la dispersión relativa promedio (desviación estándar / media).
            
        median_cv (float): Coeficiente de variación mediano. Menos sensible a 
            valores atípicos que el CV medio.
            
        min_cv (float): Coeficiente de variación mínimo observado en las predicciones.
            Indica el punto de mayor certeza en las predicciones.
            
        max_cv (float): Coeficiente de variación máximo observado en las predicciones.
            Indica el punto de mayor incertidumbre en las predicciones.
            
        uncertainty_trend (float): Tendencia de la incertidumbre a lo largo del tiempo.
            Valores positivos indican aumento de incertidumbre en el futuro;
            valores negativos indican disminución.
            
        high_uncertainty_years (list[int]): Lista de años con alta incertidumbre,
            típicamente definidos como aquellos con CV > 45%.
    """

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