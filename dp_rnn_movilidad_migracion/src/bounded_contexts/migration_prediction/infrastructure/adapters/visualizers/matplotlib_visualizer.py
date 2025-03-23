from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.visualization_port import \
    VisualizationPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult
import matplotlib.pyplot as plt
import seaborn as sns
import os


class MatplotlibVisualizer(VisualizationPort):
    """Implementación de VisualizationPort usando Matplotlib y Seaborn."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_predictions_with_uncertainty(self, prediction: PredictionResult) -> None:
        """
        Visualiza predicciones con bandas de incertidumbre.

        Args:
            prediction: Resultado de predicción a visualizar
        """
        # Configurar el estilo general
        sns.set_theme('paper')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])

        # Gráfica principal
        ax1.plot(prediction.years, prediction.values,
                 'b-', label='Predicción media', linewidth=2)

        # Banda de 1 desviación estándar
        ax1.fill_between(prediction.years,
                         [v - s for v, s in zip(prediction.values, prediction.std_devs)],
                         [v + s for v, s in zip(prediction.values, prediction.std_devs)],
                         color='b', alpha=0.2, label='±1 std')

        # Banda de intervalo de confianza
        ax1.fill_between(prediction.years,
                         prediction.lower_bounds,
                         prediction.upper_bounds,
                         color='b', alpha=0.1, label='95% IC')

        # Marcar años con alta incertidumbre
        high_uncertainty_indices = [i for i, year in enumerate(prediction.years)
                                    if year in prediction.uncertainty_metrics.high_uncertainty_years]

        if high_uncertainty_indices:
            high_uncertainty_years = [prediction.years[i] for i in high_uncertainty_indices]
            high_uncertainty_values = [prediction.values[i] for i in high_uncertainty_indices]
            ax1.scatter(high_uncertainty_years, high_uncertainty_values,
                        color='red', marker='o', label='Alta incertidumbre')

        ax1.set_title('Predicción de Crecimiento Natural con Bandas de Incertidumbre')
        ax1.set_xlabel('Año')
        ax1.set_ylabel('Crecimiento Natural')
        ax1.legend()
        ax1.grid(True)

        # Gráfica de Coeficiente de Variación
        # Calcular CV para cada año
        cv_values = [(s / abs(v)) * 100 if abs(v) > 1e-6 else 0
                     for v, s in zip(prediction.values, prediction.std_devs)]

        ax2.bar(prediction.years, cv_values, color='lightblue')
        ax2.set_title('Coeficiente de Variación por Año')
        ax2.set_ylabel('CV (%)')
        ax2.set_ylim(0, 100)

        # Añadir líneas de referencia para los niveles de incertidumbre
        ax2.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Baja')
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Alta')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prediction.state}_predicciones_con_incertidumbre.png'))
        plt.close()