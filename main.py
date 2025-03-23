"""
Punto de entrada principal de la aplicación.
"""
from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.services.migration_prediction_service import MigrationPredictionService


@inject
def main(
    conapo_service = Provide[ApplicationContainer.conapo_data_service],
    inegi_service = Provide[ApplicationContainer.inegi_data_service],
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
    migration_prediction_service.train_model(
        temporal_data=conapo_data,
        static_data=inegi_data
    )


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()