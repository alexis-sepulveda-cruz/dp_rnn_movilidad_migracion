"""
Punto de entrada principal de la aplicación.
"""
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
    migration_prediction_service: MigrationPredictionService = Provide[ApplicationContainer.migration_prediction_service]
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

    # Visualizar historial de entrenamiento
    visualizer.plot_training_history(history)

    # Análisis por estados
    states_to_compare = ["Ciudad de México", "Jalisco", "Nuevo León", "Chiapas"]
    results = {
        state: migration_prediction_service.predict_migration(
            request=PredictionRequestDTO(
                state=state,
                future_years=5,
                monte_carlo_samples=100,
                target_variable='CRE_NAT',
                confidence_level=0.95
            ),
            temporal_data=conapo_data, 
            static_data=inegi_data
        )
        for state in states_to_compare
    }


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()