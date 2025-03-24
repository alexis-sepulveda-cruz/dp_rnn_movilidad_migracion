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
            output_dir: Directorio base donde se guardarán las visualizaciones.
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.output_dir = output_dir
        
        # Definir subdirectorios para organización
        self.dirs = {
            'monte_carlo': os.path.join(self.output_dir, 'monte_carlo'),
            'resultados': os.path.join(self.output_dir, 'resultados'),
            'presentacion': os.path.join(self.output_dir, 'presentacion'),
            'entrenamiento': os.path.join(self.output_dir, 'entrenamiento'),
            'detalle': os.path.join(self.output_dir, 'detalle')
        }
        
        # Crear todos los directorios
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Crear directorio base también por si acaso
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar estilo de visualización
        sns.set_theme('paper')
        
        self.logger.info(f"Visualizador inicializado. Directorio base: {output_dir}")
        self.logger.debug(f"Subdirectorios creados: {list(self.dirs.keys())}")

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
        
        # Guardar en directorio Monte Carlo
        output_path = os.path.join(self.dirs['monte_carlo'], f'{prediction.state}_predicciones_con_incertidumbre.png')
        plt.savefig(output_path)
        self.logger.info(f"Visualización de incertidumbre guardada en: {output_path}")
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
            output_path = os.path.join(self.dirs['entrenamiento'], 'training_history.png')
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Historial de entrenamiento guardado en: {output_path}")
        
        # Cerrar figura para liberar recursos
        plt.close()
        
    def plot_state_comparison(self, predictions: Dict[str, PredictionResult]) -> None:
        """
        Visualiza comparación de predicciones entre diferentes estados.
        
        Genera visualizaciones adaptativas según el número de estados:
        - Para 1-8 estados: Una sola gráfica con todos los estados
        - Para más estados: Múltiples gráficas agrupadas por regiones o
          divididas automáticamente en grupos manejables
        
        Args:
            predictions: Diccionario con estados como claves y predicciones como valores.
        """
        num_states = len(predictions)
        self.logger.info(f"Generando visualización comparativa para {num_states} estados")
        
        if num_states == 0:
            self.logger.warning("No hay estados para comparar")
            return
        
        # Determinar el tipo de visualización según el número de estados
        if num_states <= 8:
            self._plot_single_comparison(predictions)
        else:
            self._plot_grouped_comparison(predictions)
        
    def _plot_single_comparison(self, predictions: Dict[str, PredictionResult]) -> None:
        """
        Genera una única gráfica con todos los estados.
        
        Optimizada para visualizar hasta 8 estados en un solo gráfico.
        
        Args:
            predictions: Diccionario con estados como claves y predicciones como valores.
        """
        # Configurar el estilo
        sns.set_theme('paper')
        
        # Ajustar tamaño de la figura según el número de estados
        num_states = len(predictions)
        fig_width = 12 + (num_states - 4) * 0.5 if num_states > 4 else 12
        plt.figure(figsize=(fig_width, 7))
        
        # Generar paleta de colores dinámica
        colors = plt.cm.tab10.colors if num_states <= 10 else plt.cm.tab20.colors
        
        # Obtener el máximo valor para formateo adaptativo
        max_value = max(
            max(prediction.values) 
            for prediction in predictions.values()
        )
        
        # Formateo adaptativo según magnitud de valores
        formatter = None
        if max_value > 1e6:
            formatter = lambda x: f'{x/1e6:.1f}M'
            ylabel = 'Crecimiento Natural (millones)'
        elif max_value > 1e3:
            formatter = lambda x: f'{x/1e3:.1f}K'
            ylabel = 'Crecimiento Natural (miles)'
        else:
            formatter = lambda x: f'{x:.0f}'
            ylabel = 'Crecimiento Natural'
        
        # Graficación por estado
        for i, (state, prediction) in enumerate(predictions.items()):
            color_idx = i % len(colors)  # Asegurarse de no exceder la paleta
            plt.plot(prediction.years, prediction.values, 
                    '-o', label=state, color=colors[color_idx], linewidth=2,
                    markersize=6, markerfacecolor='white')
            
            # Etiquetar solo puntos clave si hay muchos estados
            if num_states <= 4:
                # Etiquetar todos los puntos
                for x, y in zip(prediction.years, prediction.values):
                    plt.annotate(formatter(y), 
                                (x, y), 
                                textcoords="offset points", 
                                xytext=(0, 8), 
                                ha='center',
                                fontsize=7)
            else:
                # Etiquetar solo el último punto para cada estado
                x = prediction.years[-1]
                y = prediction.values[-1]
                plt.annotate(f'{state}: {formatter(y)}', 
                            (x, y), 
                            textcoords="offset points", 
                            xytext=(15, 0), 
                            ha='left',
                            fontsize=7,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Configurar gráfico
        plt.title('Predicción de Crecimiento Natural por Estado', 
                 pad=20, fontsize=12)
        plt.xlabel('Año', fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Optimizar leyenda según número de estados
        if num_states <= 8:
            # Leyenda a la derecha para pocos estados
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        else:
            # Leyenda abajo para muchos estados
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                      fancybox=True, shadow=True, ncol=min(4, num_states))
        
        plt.tight_layout()
        
        # Guardar gráfico
        output_path = os.path.join(self.dirs['presentacion'], 'comparacion_estados.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Visualización comparativa guardada en: {output_path}")
        
        plt.close()

    def _plot_grouped_comparison(self, predictions: Dict[str, PredictionResult]) -> None:
        """
        Genera múltiples gráficas agrupadas para visualizar muchos estados.
        
        Optimizada para visualizar más de 8 estados, agrupándolos en múltiples
        gráficas organizadas por regiones geográficas o en grupos secuenciales.
        
        Args:
            predictions: Diccionario con estados como claves y predicciones como valores.
        """
        # Intentar importar el servicio de selección de estados para agrupar por región
        try:
            from dp_rnn_movilidad_migracion.src.shared.domain.services.state_selection_service import StateSelectionService
            regions = StateSelectionService.group_states_by_region()
            use_regions = True
        except ImportError:
            use_regions = False
            self.logger.warning("StateSelectionService no disponible, usando agrupación secuencial")
        
        if use_regions:
            # Agrupar estados por región
            grouped_predictions = {}
            for region_name, region_states in regions.items():
                region_dict = {
                    state: prediction 
                    for state, prediction in predictions.items() 
                    if state in region_states
                }
                if region_dict:  # Solo incluir regiones con estados presentes
                    grouped_predictions[region_name] = region_dict
            
            # Validar que todos los estados tengan una región asignada
            all_grouped_states = [state for states in grouped_predictions.values() for state in states]
            missing_states = set(predictions.keys()) - set(all_grouped_states)
            
            if missing_states:
                self.logger.warning(f"Estados sin región asignada: {missing_states}")
                # Crear grupo "Otros" para estados sin región
                grouped_predictions["Otros"] = {
                    state: predictions[state] for state in missing_states
                }
        else:
            # Agrupar secuencialmente, máximo 6 estados por grupo
            MAX_STATES_PER_GROUP = 6
            states = list(predictions.keys())
            grouped_predictions = {}
            
            for i in range(0, len(states), MAX_STATES_PER_GROUP):
                group_states = states[i:i+MAX_STATES_PER_GROUP]
                group_name = f"Grupo {i//MAX_STATES_PER_GROUP + 1}"
                grouped_predictions[group_name] = {
                    state: predictions[state] for state in group_states
                }
        
        # Preparar la figura principal
        num_groups = len(grouped_predictions)
        fig_height = 6 * num_groups
        fig, axes = plt.subplots(num_groups, 1, figsize=(12, fig_height), 
                                 sharex=True, constrained_layout=True)
        
        # Manejar caso de un solo grupo (axes no es un array)
        if num_groups == 1:
            axes = [axes]
        
        # Obtener el máximo valor para formateo adaptativo
        max_value = max(
            max(prediction.values) 
            for prediction in predictions.values()
        )
        
        # Formateo adaptativo
        if max_value > 1e6:
            formatter = lambda x: f'{x/1e6:.1f}M'
            ylabel = 'Crecimiento Natural (millones)'
        elif max_value > 1e3:
            formatter = lambda x: f'{x/1e3:.1f}K'
            ylabel = 'Crecimiento Natural (miles)'
        else:
            formatter = lambda x: f'{x:.0f}'
            ylabel = 'Crecimiento Natural'
        
        # Generar gráficas por grupo
        for i, (group_name, group_predictions) in enumerate(grouped_predictions.items()):
            ax = axes[i]
            colors = plt.cm.tab10.colors
            
            # Título del grupo
            ax.set_title(f"{group_name}", fontsize=11)
            
            # Graficar cada estado del grupo
            for j, (state, prediction) in enumerate(group_predictions.items()):
                color_idx = j % len(colors)
                ax.plot(prediction.years, prediction.values, 
                       '-o', label=state, color=colors[color_idx], linewidth=2,
                       markersize=6, markerfacecolor='white')
                
                # Etiquetar último punto
                last_year = prediction.years[-1]
                last_value = prediction.values[-1]
                ax.annotate(formatter(last_value), 
                           (last_year, last_value), 
                           textcoords="offset points", 
                           xytext=(5, 0), 
                           ha='left',
                           fontsize=7)
            
            # Configurar eje
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(loc='upper left', fontsize=8)
        
        # Configurar eje x compartido
        axes[-1].set_xlabel('Año', fontsize=10)
        
        # Título general
        fig.suptitle('Predicción de Crecimiento Natural por Estado', 
                    fontsize=14, y=0.98)
        
        # Guardar gráfica
        output_path = os.path.join(self.dirs['presentacion'], 'comparacion_estados_grupos.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Visualización comparativa por grupos guardada en: {output_path}")
        
        plt.close()
        
    def plot_reliability_comparison(self, reliability_scores: Dict[str, float]) -> None:
        """
        Visualiza comparación de scores de confiabilidad entre diferentes estados.
        
        Args:
            reliability_scores: Diccionario con estados como claves y scores de
                confiabilidad como valores.
        """
        self.logger.info(f"Generando visualización de confiabilidad para {len(reliability_scores)} estados")
        
        # Obtener datos para el gráfico
        states = list(reliability_scores.keys())
        scores = list(reliability_scores.values())
        
        # Configurar y crear figura
        plt.figure(figsize=(10, 6))
        
        # Crear gráfico de barras con colores personalizados
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(range(len(states)), scores, color=colors[:len(states)])
        
        # Personalizar el gráfico
        plt.title('Score de Confiabilidad por Estado\nBasado en Análisis Monte Carlo', 
                 pad=20, fontsize=12)
        plt.xlabel('Estado', fontsize=10)
        plt.ylabel('Score de Confiabilidad (%)', fontsize=10)
        
        # Configurar eje X
        plt.xticks(range(len(states)), states, rotation=45, ha='right')
        
        # Configurar eje Y
        min_score = min(scores) - 1
        max_score = max(scores) + 1
        plt.ylim(min_score, max_score)
        
        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontsize=9)
        
        # Añadir grid y ajustar layout
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Guardar gráfico en directorio de presentación
        output_path = os.path.join(self.dirs['presentacion'], 'confiabilidad_estados.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Visualización de confiabilidad guardada en: {output_path}")
        
        plt.close()
        
    def plot_state_detail(self, prediction: PredictionResult) -> None:
        """
        Visualiza el detalle de predicción para un estado específico.
        
        Args:
            prediction: Resultado de predicción del estado a visualizar
        """
        self.logger.info(f"Generando visualización detallada para {prediction.state}")
        
        # Configurar el estilo general
        sns.set_theme('paper')
        
        # Crear la figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar el título principal
        fig.suptitle(f'Análisis Detallado del Crecimiento Natural - {prediction.state}\n' +
                    f'Predicciones {min(prediction.years)}-{max(prediction.years)}', 
                    fontsize=14, y=1.05)
        
        # Calcular los valores para los límites de los ejes
        cre_values = prediction.values
        cre_min, cre_max = min(cre_values), max(cre_values)
        cre_range = cre_max - cre_min
        
        # Gráfico de Crecimiento Natural
        line = ax.plot(prediction.years, cre_values,
                      '-o', color='#1f77b4', linewidth=2, 
                      markersize=8, markerfacecolor='white',
                      markeredgewidth=2, 
                      label='Crecimiento Natural')
        
        # Configurar límites y formato del eje Y
        ax.set_ylim(cre_min - cre_range*0.1, cre_max + cre_range*0.1)
        
        # Si los valores son muy grandes, mostrar en miles o millones
        if abs(cre_max) > 1e6:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.2f}M'))
            ylabel = 'Millones de habitantes'
        elif abs(cre_max) > 1e3:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.1f}K'))
            ylabel = 'Miles de habitantes'
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
            ylabel = 'Habitantes'
        
        # Configurar el grid y los ejes
        ax.grid(True, linestyle='--', alpha=0.3, which='both')
        ax.set_title('Predicción del Crecimiento Natural', pad=20, fontsize=12)
        ax.set_xlabel('Año', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Añadir valores sobre los puntos
        for x, y in zip(prediction.years, cre_values):
            if abs(y) > 1e6:
                label = f'{y/1e6:.2f}M'
            elif abs(y) > 1e3:
                label = f'{y/1e3:.1f}K'
            else:
                label = f'{y:.0f}'
                
            ax.annotate(label, 
                       (x, y),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Si hay una métrica de confiabilidad disponible, mostrarla
        reliability_info = f"Fiabilidad: {prediction.uncertainty_metrics.reliability_score:.1f}%"
        ax.text(0.02, 0.95, reliability_info,
               transform=ax.transAxes, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Añadir leyenda
        ax.legend()
        
        # Ajustar el layout
        plt.tight_layout()
        
        # Guardar gráfico en directorio de detalle
        output_path = os.path.join(self.dirs['detalle'], f'{prediction.state}_detalle.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        self.logger.info(f"Visualización detallada guardada en: {output_path}")
        
        plt.close()