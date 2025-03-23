"""
Punto de entrada principal de la aplicación.
"""
from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.conapo_data_service import ConapoDataService

@inject
def main(
    inegi_repository: DataRepository = Provide[ApplicationContainer.inegi_repository]
):
    # Cargar datos
    logger = LoggerFactory.get_composite_logger(__name__)
    logger.info("Cargando datos de CONAPO")
    conapo_service = ConapoDataService()
    conapo_data = conapo_service.get_processed_data()
    logger.info(f"Datos de CONAPO cargados: {conapo_data.shape} filas")

    logger.info("Cargando datos de INEGI")
    inegi_repository.load_data()


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()