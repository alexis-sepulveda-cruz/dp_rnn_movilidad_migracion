"""
Implementación del visualizador utilizando Matplotlib y Seaborn.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.visualization_port import \
    VisualizationPort
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import \
    PredictionResult
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class MatplotlibVisualizer(VisualizationPort):
    """Implementación de VisualizationPort usando Matplotlib y Seaborn."""

    def __init__(self, output_dir: str):
        """
        Inicializa el visualizador.
        
        Args:
            output_dir: Directorio donde se guardarán las visualizaciones.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.output_dir = output_dir
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar estilo de visualización
        sns.set_theme('paper')
        
        self.logger.info(f"Visualizador inicializado. Directorio de salida: {output_dir}")

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

    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None) -> None:
        """
        Visualiza el historial de entrenamiento de un modelo.
        
        Args:
            history: Diccionario con histórico de métricas durante el entrenamiento
            save_path: Ruta donde guardar la visualización. Si es None, se usa el directorio predeterminado.
        """
        self.logger.info("Generando visualización del historial de entrenamiento")
        
        plt.figure(figsize=(12, 4))
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Gráfico de error absoluto medio
        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar figura si se proporciona ruta o usar directorio predeterminado
        if save_path:
            output_path = save_path
        else:
            output_path = os.path.join(self.output_dir, 'training_history.png')
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Historial de entrenamiento guardado en: {output_path}")
        
        # Cerrar figura para liberar recursos
        plt.close()
        
    def plot_state_comparison(self, predictions: Dict[str, PredictionResult]) -> None:
        """
        Visualiza comparación de predicciones entre diferentes estados.
        
        Args:
            predictions: Diccionario con estados como claves y predicciones como valores.
        """
        self.logger.info(f"Generando visualización comparativa para {len(predictions)} estados")
        
        # Configurar el estilo
        sns.set_theme('paper')
        plt.figure(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for (state, prediction), color in zip(predictions.items(), colors[:len(predictions)]):
            plt.plot(prediction.years, prediction.values, 
                    '-o', label=state, color=color, linewidth=2,
                    markersize=8, markerfacecolor='white')
            
            # Añadir valores sobre los puntos
            for x, y in zip(prediction.years, prediction.values):
                plt.annotate(f'{y:,.0f}', 
                            (x, y), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=8)
        
        plt.title('Predicción de Crecimiento Natural por Estado', 
                 pad=20, fontsize=12)
        plt.xlabel('Año')
        plt.ylabel('Crecimiento Natural')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Guardar gráfico
        output_path = os.path.join(self.output_dir, 'comparacion_estados.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Visualización comparativa guardada en: {output_path}")
        
        plt.close()