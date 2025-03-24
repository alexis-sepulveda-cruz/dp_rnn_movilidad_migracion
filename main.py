"""
Punto de entrada principal de la aplicación.

Este módulo coordina el flujo de trabajo completo de entrenamiento
y predicción del modelo para la migración en México.
"""
import pandas as pd
import logging
from typing import Dict, List

from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.services.migration_prediction_service import MigrationPredictionService
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import PredictionRequestDTO
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.entities.prediction_result import PredictionResult
from dp_rnn_movilidad_migracion.src.shared.domain.value_objects.mexican_states import MexicanState
from dp_rnn_movilidad_migracion.src.shared.domain.services.state_selection_service import StateSelectionService
from dp_rnn_movilidad_migracion.src.shared.domain.services.state_name_normalizer import StateNameNormalizer


def run_model_training(
    conapo_data: pd.DataFrame,
    inegi_data: pd.DataFrame,
    migration_prediction_service: MigrationPredictionService,
    logger: logging.Logger = None
) -> Dict[str, List[float]]:
    """
    Entrena el modelo con los datos proporcionados.
    
    Args:
        conapo_data: DataFrame con datos temporales de CONAPO
        inegi_data: DataFrame con datos estáticos de INEGI
        migration_prediction_service: Servicio de predicción inyectado
        logger: Logger para mensajes de información
        
    Returns:
        Diccionario con historial de métricas de entrenamiento
    """
    if logger is None:
        logger = LoggerFactory.get_composite_logger("model_training")
        
    logger.info("Entrenando modelo de predicción de migración")
    
    history = migration_prediction_service.train_model(
        temporal_data=conapo_data,
        static_data=inegi_data,
        visualize_history=True
    )
    
    logger.info(f"Entrenamiento completado con {len(history['loss'])} épocas")
    return history


def run_state_predictions(
    states: List[str],
    conapo_data: pd.DataFrame,
    inegi_data: pd.DataFrame,
    migration_prediction_service: MigrationPredictionService,
    visualizer = None,
    monte_carlo_samples: int = 500,
    logger: logging.Logger = None
) -> Dict[str, PredictionResult]:
    """
    Genera predicciones para los estados especificados.
    
    Args:
        states: Lista de estados para los que se generarán predicciones
        conapo_data: DataFrame con datos temporales de CONAPO
        inegi_data: DataFrame con datos estáticos de INEGI
        migration_prediction_service: Servicio de predicción inyectado
        visualizer: Servicio de visualización inyectado
        monte_carlo_samples: Número de muestras para simulación Monte Carlo
        logger: Logger para mensajes de información
        
    Returns:
        Diccionario con resultados de predicción por estado
    """
    if logger is None:
        logger = LoggerFactory.get_composite_logger("state_predictions")
        
    results = {}
    
    for state in states:
        logger.info(f"Prediciendo migración para: {state}")
        try:
            # Configurar y ejecutar solicitud de predicción
            prediction_dto = migration_prediction_service.predict_migration(
                request=PredictionRequestDTO(
                    state=state,
                    future_years=5,
                    monte_carlo_samples=monte_carlo_samples,
                    target_variable='CRE_NAT',
                    confidence_level=0.95,
                    visualize=True
                ),
                temporal_data=conapo_data,
                static_data=inegi_data
            )
            
            # Recuperar entidad completa desde el repositorio
            results[state] = migration_prediction_service.prediction_repository.get_prediction(state)
            logger.info(f"Predicción completada para {state}")
            
        except Exception as e:
            logger.error(f"Error al generar predicción para {state}: {str(e)}")
    
    # Generar visualizaciones comparativas si hay resultados
    if results and visualizer:
        # Visualización comparativa general
        visualizer.plot_state_comparison(results)
        
        # Visualizaciones detalladas por estado
        for state, prediction in results.items():
            visualizer.plot_state_detail(prediction)
    
    return results


def analyze_prediction_results(
    results: Dict[str, PredictionResult],
    uncertainty_analyzer,
    visualizer = None,
    logger: logging.Logger = None
) -> Dict[str, float]:
    """
    Analiza los resultados de predicción y genera reportes de confiabilidad.
    
    Args:
        results: Diccionario con resultados de predicción por estado
        uncertainty_analyzer: Analizador de incertidumbre inyectado
        visualizer: Servicio de visualización inyectado
        logger: Logger para mensajes de información
        
    Returns:
        Diccionario con scores de confiabilidad por estado
    """
    if logger is None:
        logger = LoggerFactory.get_composite_logger("results_analysis")
        
    reliability_scores = {}
    
    # Analizar cada predicción y generar reportes
    for state, prediction in results.items():
        logger.info(f"Analizando resultados para: {state}")
        
        # Convertir la entidad a DataFrame para análisis detallado
        prediction_df = pd.DataFrame({
            'AÑO': prediction.years,
            'CRE_NAT': prediction.values,
            'CRE_NAT_std': prediction.std_devs,
            'CRE_NAT_lower': prediction.lower_bounds,
            'CRE_NAT_upper': prediction.upper_bounds
        })
        
        # Calcular métricas detalladas
        detailed_predictions = uncertainty_analyzer.calculate_detailed_metrics(prediction_df)
        
        # Generar y mostrar reporte de confiabilidad
        report = uncertainty_analyzer.generate_reliability_report(detailed_predictions)
        reliability_scores[state] = report['reliability_metrics']['reliability_score']
        
        # Imprimir reporte detallado
        print(f"\nAnálisis para {state}")
        print("="*50)
        uncertainty_analyzer.print_detailed_report(report)
    
    # Visualizar comparación de confiabilidad
    if reliability_scores and visualizer:
        visualizer.plot_reliability_comparison(reliability_scores)
    
    return reliability_scores


@inject
def main(
    conapo_service = Provide[ApplicationContainer.conapo_data_service],
    inegi_service = Provide[ApplicationContainer.inegi_data_service],
    visualizer = Provide[ApplicationContainer.visualizer],
    migration_prediction_service = Provide[ApplicationContainer.migration_prediction_service],
    uncertainty_analyzer = Provide[ApplicationContainer.uncertainty_analyzer],
):
    """
    Función principal que coordina el flujo de trabajo completo.
    """
    # Configurar logger para la función principal
    logger = LoggerFactory.get_composite_logger(__name__)
    
    try:
        # 1. Cargar datos
        logger.info("Cargando datos de CONAPO y INEGI")
        conapo_data = conapo_service.get_processed_data()
        inegi_data = inegi_service.get_processed_data()
        logger.info(f"Datos cargados: CONAPO ({conapo_data.shape[0]} filas), INEGI ({inegi_data.shape[0]} filas)")
        
        # 2. Entrenar modelo - IMPORTANTE: pasar el servicio explícitamente a cada función
        history = run_model_training(
            conapo_data=conapo_data, 
            inegi_data=inegi_data, 
            migration_prediction_service=migration_prediction_service,
            logger=logger
        )
        
        # 3. Generar predicciones por estado - usando nombres normalizados
        states_to_compare = MexicanState.get_all_states()
        
        # Información para diagnóstico
        logger.info("Comparando nombres de estados entre INEGI y MexicanState")
        inegi_states = set(inegi_data['NOM_ENT'].unique())
        mexican_state_values = set(state for state in states_to_compare)
        
        logger.info(f"Estados en INEGI ({len(inegi_states)}): {sorted(inegi_states)}")
        logger.info(f"Estados en MexicanState ({len(mexican_state_values)}): {sorted(mexican_state_values)}")
        
        # Mapeo de diagnóstico
        for state in states_to_compare:
            official = StateNameNormalizer.to_official_name(state)
            if official in inegi_states:
                logger.debug(f"Mapeo correcto: {state} -> {official}")
            else:
                logger.warning(f"Posible problema de mapeo: {state} -> {official}")
        
        results = run_state_predictions(
            states=states_to_compare,
            conapo_data=conapo_data, 
            inegi_data=inegi_data,
            migration_prediction_service=migration_prediction_service,
            visualizer=visualizer,
            logger=logger
        )
        
        # 4. Analizar resultados y generar reportes
        reliability_scores = analyze_prediction_results(
            results=results, 
            uncertainty_analyzer=uncertainty_analyzer,
            visualizer=visualizer,
            logger=logger
        )
        
        # 5. Mostrar resumen final
        logger.info("Proceso completado con éxito")
        logger.info(f"Scores de confiabilidad: {', '.join([f'{s}: {v:.1f}%' for s, v in reliability_scores.items()])}")
        
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()