"""
Punto de entrada principal de la aplicación.
"""
import pandas as pd
from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.services.migration_prediction_service import MigrationPredictionService
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import PredictionRequestDTO

@inject
def main(
    conapo_service = Provide[ApplicationContainer.conapo_data_service],
    inegi_service = Provide[ApplicationContainer.inegi_data_service],
    visualizer = Provide[ApplicationContainer.visualizer],
    migration_prediction_service: MigrationPredictionService = Provide[ApplicationContainer.migration_prediction_service],
    uncertainty_analyzer = Provide[ApplicationContainer.uncertainty_analyzer],
):
    # Cargar datos
    logger = LoggerFactory.get_composite_logger(__name__)

    # Obtener datos procesados
    logger.info("Cargando datos de CONAPO")
    conapo_data = conapo_service.get_processed_data()
    logger.info(f"Datos de CONAPO cargados: {conapo_data.shape} filas")

    logger.info("Cargando datos de INEGI")
    inegi_data = inegi_service.get_processed_data()
    logger.info(f"Datos de INEGI cargados: {inegi_data.shape} filas")

    logger.info("Migration Prediction Service")
    history = migration_prediction_service.train_model(
        temporal_data=conapo_data,
        static_data=inegi_data,
        visualize_history=True  # Habilitar visualización del historial
    )
    
    logger.info(f"Entrenamiento completado con {len(history['loss'])} épocas")

    # Análisis por estados
    states_to_compare = ["Ciudad de México", "Jalisco", "Nuevo León", "Chiapas"]
    results = {}
    
    for state in states_to_compare:
        prediction_dto = migration_prediction_service.predict_migration(
            request=PredictionRequestDTO(
                state=state,
                future_years=5,
                monte_carlo_samples=500,
                target_variable='CRE_NAT',
                confidence_level=0.95
            ),
            temporal_data=conapo_data, 
            static_data=inegi_data
        )
        # Obtener la entidad del repositorio para tener el objeto completo
        results[state] = migration_prediction_service.prediction_repository.get_prediction(state)
    
    # Generar visualización comparativa y por estado
    visualizer.plot_state_comparison(results)
    
    # Visualizar detalles individuales de cada estado
    for state, prediction in results.items():
        visualizer.plot_state_detail(prediction)

    # Almacenar los scores de confiabilidad
    reliability_scores = {}

    for state, prediction in results.items():
        print(f"\nAnálisis para {state}")
        print("="*50)
        
        # Convertir la entidad a DataFrame con métricas detalladas
        detailed_predictions = uncertainty_analyzer.calculate_detailed_metrics(
            pd.DataFrame({
                'AÑO': prediction.years,
                'CRE_NAT': prediction.values,
                'CRE_NAT_std': prediction.std_devs,
                'CRE_NAT_lower': prediction.lower_bounds,
                'CRE_NAT_upper': prediction.upper_bounds
            })
        )
        
        # Generar reporte de confiabilidad
        report = uncertainty_analyzer.generate_reliability_report(detailed_predictions)
        
        # Guardar el score de confiabilidad para cada estado
        reliability_scores[state] = report['reliability_metrics']['reliability_score']
        
        # Imprimir reporte
        uncertainty_analyzer.print_detailed_report(report)

    # Visualizar scores de confiabilidad
    visualizer.plot_reliability_comparison(reliability_scores)


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()