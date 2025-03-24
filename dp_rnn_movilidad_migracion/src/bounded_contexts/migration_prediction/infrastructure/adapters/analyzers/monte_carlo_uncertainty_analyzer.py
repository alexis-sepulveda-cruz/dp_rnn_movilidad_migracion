"""
Analizador de incertidumbre para predicciones Monte Carlo.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.uncertainty_analyzer_port import UncertaintyAnalyzerPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.value_objects.uncertainty_metrics import UncertaintyMetrics
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class MonteCarloUncertaintyAnalyzer(UncertaintyAnalyzerPort):
    """
    Implementación de UncertaintyAnalyzerPort para predicciones Monte Carlo.
    
    Analiza la incertidumbre en predicciones generadas mediante simulación
    Monte Carlo y calcula métricas para evaluar su confiabilidad.
    """
    
    def __init__(self, high_uncertainty_threshold: float = 45.0, target_normalizer: Optional[any] = None):
        """
        Inicializa el analizador de incertidumbre.
        
        Args:
            high_uncertainty_threshold: Umbral de CV% para considerar alta incertidumbre
            target_normalizer: Normalizador para desnormalizar predicciones (opcional)
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.high_uncertainty_threshold = high_uncertainty_threshold
        self.target_normalizer = target_normalizer
        
        if target_normalizer:
            self.logger.info("Analizador inicializado con capacidad de desnormalización")
        else:
            self.logger.info("Analizador inicializado sin desnormalización")
    
    def analyze_uncertainty(
        self, 
        predictions: pd.DataFrame, 
        uncertainty_data: List[dict]
    ) -> UncertaintyMetrics:
        """
        Analiza la incertidumbre en las predicciones y genera métricas.
        
        Args:
            predictions: DataFrame con predicciones estadísticas
            uncertainty_data: Datos adicionales de incertidumbre por paso
            
        Returns:
            Objeto UncertaintyMetrics con las métricas calculadas
        """
        self.logger.info("Analizando incertidumbre en predicciones")
        
        # Calcular coeficientes de variación usando método compartido
        cv_values = self._calculate_cv_values(predictions)
        
        # Identificar años con alta incertidumbre
        high_uncertainty_years = predictions.loc[cv_values > self.high_uncertainty_threshold, 'AÑO'].tolist()
        
        # Calcular score de confiabilidad (función exponencial inversa del CV medio)
        mean_cv = cv_values.mean()
        reliability_score = 100 * np.exp(-mean_cv / 50)
        
        # Calcular tendencia de incertidumbre
        uncertainty_trend = cv_values.diff().mean()
        
        # Crear objeto de valor con métricas
        metrics = UncertaintyMetrics(
            reliability_score=reliability_score,
            mean_cv=mean_cv,
            median_cv=cv_values.median(),
            min_cv=cv_values.min(),
            max_cv=cv_values.max(),
            uncertainty_trend=uncertainty_trend,
            high_uncertainty_years=high_uncertainty_years
        )
        
        self.logger.info(f"Análisis completado: score={reliability_score:.2f}, CV medio={mean_cv:.2f}")
        if high_uncertainty_years:
            self.logger.warning(f"Se detectaron {len(high_uncertainty_years)} años con alta incertidumbre")
            
        return metrics
    
    def calculate_detailed_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas detalladas de incertidumbre para cada punto de predicción.
        
        Enriquece el DataFrame de predicciones con métricas adicionales como niveles
        de incertidumbre, detección de outliers y puntuaciones z.
        
        Args:
            predictions: DataFrame con predicciones
            
        Returns:
            DataFrame enriquecido con métricas adicionales
        """
        self.logger.info("Calculando métricas detalladas de incertidumbre")
        
        # Crear copia del DataFrame para no modificar el original
        detailed_df = predictions.copy()
        
        # Calcular relative_std y CV reutilizando el método compartido
        cv_values = self._calculate_cv_values(detailed_df)
        detailed_df['CV'] = cv_values
        detailed_df['relative_std'] = cv_values / 100  # Convertir de porcentaje a proporción
        
        # Clasificar niveles de incertidumbre
        detailed_df['uncertainty_level'] = pd.cut(
            detailed_df['CV'],
            bins=[0, 15, 30, 45, 60, 100],
            labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta']
        )
        
        # Análisis de outliers
        mean_pred = detailed_df['CRE_NAT'].mean()
        std_pred = detailed_df['CRE_NAT'].std()
        detailed_df['z_score'] = (detailed_df['CRE_NAT'] - mean_pred) / (std_pred + 1e-6)
        detailed_df['is_outlier'] = abs(detailed_df['z_score']) > 2
        
        # Contabilizar anomalías
        outlier_count = detailed_df['is_outlier'].sum()
        if outlier_count > 0:
            self.logger.warning(f"Se detectaron {outlier_count} outliers en las predicciones")
            
        self.logger.debug("Métricas detalladas calculadas con éxito")
        return detailed_df
    
    def calculate_prediction_statistics(
        self,
        all_trajectories: np.ndarray,
        base_year: int,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Calcula estadísticas a partir de múltiples trayectorias de predicción.
        
        Args:
            all_trajectories: Matriz con múltiples trayectorias de predicción [samples, steps]
            base_year: Año base desde donde parten las predicciones
            confidence_level: Nivel de confianza para los intervalos
            
        Returns:
            DataFrame con años y estadísticas calculadas
        """
        self.logger.info(f"Calculando estadísticas con nivel de confianza {confidence_level}")
        
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
        
        # Desnormalizar predicciones si hay un normalizador disponible
        if self.target_normalizer is not None:
            self.logger.debug("Desnormalizando predicciones")
            mean_predictions = self.target_normalizer.inverse_transform(mean_predictions.reshape(-1, 1)).flatten()
            std_predictions = self.target_normalizer.inverse_transform(std_predictions.reshape(-1, 1)).flatten()
            lower_bound = self.target_normalizer.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
            upper_bound = self.target_normalizer.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
        else:
            self.logger.debug("Omitiendo desnormalización: no hay normalizador configurado")
        
        # Crear DataFrame con resultados
        years = range(base_year + 1, base_year + future_years + 1)
        
        df = pd.DataFrame({
            'AÑO': years,
            'CRE_NAT': mean_predictions,
            'CRE_NAT_std': std_predictions,
            'CRE_NAT_lower': lower_bound,
            'CRE_NAT_upper': upper_bound
        })
        
        self.logger.debug(f"Estadísticas calculadas para {future_years} años futuros")
        return df
    
    def _calculate_cv_values(self, predictions: pd.DataFrame) -> pd.Series:
        """
        Calcula los coeficientes de variación para un DataFrame de predicciones.
        
        Este método compartido evita duplicación de código entre analyze_uncertainty
        y calculate_detailed_metrics.
        
        Args:
            predictions: DataFrame con predicciones
            
        Returns:
            Serie con los coeficientes de variación para cada fila
        """
        # Calcular coeficientes de variación de manera robusta
        abs_values = predictions['CRE_NAT'].abs()
        cv_values = (predictions['CRE_NAT_std'] / (abs_values + 1e-6)) * 100
        cv_values = np.clip(cv_values, 0, 100)  # Limitar a rango 0-100%
        
        return cv_values
    
    def generate_reliability_report(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un reporte detallado de confiabilidad para predicciones.
        
        Args:
            predictions: DataFrame con predicciones enriquecidas con métricas 
                        de incertidumbre
                        
        Returns:
            Diccionario con métricas organizadas por categorías
        """
        self.logger.info("Generando reporte de confiabilidad detallado")
        
        # Si las métricas detalladas no están presentes, calcularlas
        if 'CV' not in predictions.columns:
            self.logger.debug("Métricas detalladas no encontradas, calculándolas")
            predictions = self.calculate_detailed_metrics(predictions)
        
        # Calcular score de confiabilidad usando una función exponencial
        mean_cv = predictions['CV'].mean()
        reliability_score = 100 * np.exp(-mean_cv / 50)  # Decae exponencialmente con CV
        
        # Analizar la dispersión de las predicciones
        relative_range = (predictions['CRE_NAT_upper'] - predictions['CRE_NAT_lower']) / (
            predictions['CRE_NAT'].abs() + 1e-6
        )
        
        # Categorizar años según niveles de incertidumbre más detallados
        stability_levels = {
            'very_stable_years': len(predictions[predictions['CV'] <= 15]),
            'stable_years': len(predictions[(predictions['CV'] > 15) & (predictions['CV'] <= 30)]),
            'moderate_years': len(predictions[(predictions['CV'] > 30) & (predictions['CV'] <= 45)]),
            'unstable_years': len(predictions[(predictions['CV'] > 45) & (predictions['CV'] <= 60)]),
            'very_unstable_years': len(predictions[predictions['CV'] > 60])
        }
        
        # Analizar la tendencia de la incertidumbre
        uncertainty_trend = predictions['CV'].diff().mean()
        
        report = {
            'reliability_metrics': {
                'reliability_score': reliability_score,
                'mean_cv': mean_cv,
                'median_cv': predictions['CV'].median(),
                'min_cv': predictions['CV'].min(),
                'max_cv': predictions['CV'].max()
            },
            'uncertainty_analysis': {
                'mean_relative_range': relative_range.mean(),
                'uncertainty_trend': uncertainty_trend,
                'years_high_uncertainty': predictions[predictions['CV'] > 45]['AÑO'].tolist()
            },
            'stability_distribution': stability_levels,
            'prediction_ranges': {
                'mean_width': (predictions['CRE_NAT_upper'] - predictions['CRE_NAT_lower']).mean(),
                'relative_width': relative_range.mean() * 100,
                'outliers_detected': predictions['is_outlier'].sum()
            }
        }
        
        self.logger.info(f"Reporte generado: confiabilidad={reliability_score:.2f}%, "
                        f"CV medio={mean_cv:.2f}%, "
                        f"outliers={report['prediction_ranges']['outliers_detected']}")
        
        return report
    
    def print_detailed_report(self, report: Dict[str, Any]) -> None:
        """
        Imprime un reporte detallado de confiabilidad en formato legible.
        
        Args:
            report: Diccionario con el reporte generado por generate_reliability_report
        """
        self.logger.info("Imprimiendo reporte detallado")
        
        print("\nREPORTE DE CONFIABILIDAD DETALLADO")
        print("="*50)
        
        print("\n1. Métricas de Confiabilidad:")
        print(f"  Score de Confiabilidad: {report['reliability_metrics']['reliability_score']:.1f}%")
        print(f"  CV Medio: {report['reliability_metrics']['mean_cv']:.1f}%")
        print(f"  CV Mediano: {report['reliability_metrics']['median_cv']:.1f}%")
        print(f"  Rango CV: [{report['reliability_metrics']['min_cv']:.1f}% - {report['reliability_metrics']['max_cv']:.1f}%]")
        
        print("\n2. Análisis de Incertidumbre:")
        print(f"  Rango Relativo Medio: {report['uncertainty_analysis']['mean_relative_range']:.2f}")
        print(f"  Tendencia de Incertidumbre: {report['uncertainty_analysis']['uncertainty_trend']:.2f}")
        print(f"  Años con Alta Incertidumbre: {report['uncertainty_analysis']['years_high_uncertainty']}")
        
        print("\n3. Distribución de Estabilidad:")
        for level, count in report['stability_distribution'].items():
            print(f"  {level.replace('_', ' ').title()}: {count}")
        
        print("\n4. Rangos de Predicción:")
        print(f"  Ancho Medio del Intervalo: {report['prediction_ranges']['mean_width']:.2f}")
        print(f"  Ancho Relativo (%): {report['prediction_ranges']['relative_width']:.1f}%")
        print(f"  Outliers Detectados: {report['prediction_ranges']['outliers_detected']}")
