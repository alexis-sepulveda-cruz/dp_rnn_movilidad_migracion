"""
Punto de entrada principal de la aplicación.
"""
from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.conapo_schema import TEMPORAL_FEATURES
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.schemas.inegi_schema import INEGI_STATIC_FEATURES

@inject
def main(
    conapo_service = Provide[ApplicationContainer.conapo_data_service],
    inegi_service = Provide[ApplicationContainer.inegi_data_service],
    data_preparation_service = Provide[ApplicationContainer.model_data_preparation_service]
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

    # Preparar datos para el modelo
    logger.info("Preparando datos para el modelo")
    X, y = data_preparation_service.prepare_model_data(
        temporal_data=conapo_data,
        static_data=inegi_data,
        target_column='CRE_NAT',
        sequence_length=5,
        temporal_features=TEMPORAL_FEATURES,
        static_features=INEGI_STATIC_FEATURES
    )
    logger.info(f"Datos preparados: X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()