"""
Punto de entrada principal de la aplicación.
"""
from dependency_injector.wiring import inject, Provide
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.domain.ports.data_repository import DataRepository

@inject
def main(
    logger=Provide[ApplicationContainer.logger],
    conapo_repository: DataRepository = Provide[ApplicationContainer.conapo_repository],
    inegi_repository: DataRepository = Provide[ApplicationContainer.inegi_repository]
):
    logger.info("Cargando datos de CONAPO")
    conapo_repository.load_data()

    logger.info("Cargando datos de INEGI")
    inegi_repository.load_data()


if __name__ == "__main__":
    # Inicializar la aplicación y exponer el contenedor
    container = bootstrap_app()

    # Conectar el contenedor al módulo actual
    container.wire(modules=[__name__])

    main()