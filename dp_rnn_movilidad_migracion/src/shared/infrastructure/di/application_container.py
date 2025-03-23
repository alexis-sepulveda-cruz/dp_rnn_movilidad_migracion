"""
application_container.py - Contenedor de aplicación centralizado
"""
from dependency_injector import containers, providers

from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.conapo_repository import ConapoRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.inegi_repository import InegiRepository


class ApplicationContainer(containers.DeclarativeContainer):
    """Contenedor único y centralizado para toda la aplicación."""

    # Configuración
    config = providers.Configuration()

    # Configuración básica por defecto
    config.set_default_values({
        "logging": {
            "project_name": "dp_rnn_movilidad_migracion",
            "log_dir": "logs"
        },
        "paths": {
            "base": "",
            "conapo": "",
            "inegi": ""
        },
        "files": {
            "conapo": "5_Indicadores_demográficos_proyecciones.xlsx",
            "inegi": "RESLOC_NACCSV2020.csv"
        },
        "data": {
            "years": {
                "start": 1970,
                "end": 2019
            },
            "sequence_length": 5,
            "random_seed": 42
        }
    })

    # Factory de loggers
    logger_factory = providers.Resource(
        LoggerFactory.configure,
        project_name=config.logging.project_name,
        log_dir=config.logging.log_dir
    )

    # Provider genérico de loggers
    logger = providers.Factory(
        LoggerFactory.get_composite_logger,
        module_name=providers.Callable(lambda: __name__)
    )

    # Repositorios
    conapo_repository = providers.Singleton(
        ConapoRepository,
        logger=logger.provider(
            module_name='dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.conapo_repository'
        ),
        conapo_path=config.paths.conapo,
        conapo_file=config.files.conapo,
        start_year=config.data.years.start,
        end_year=config.data.years.end,
        include_derived=True,
        include_targets=True
    )

    inegi_repository = providers.Singleton(
        InegiRepository,
        logger=logger.provider(
            module_name='dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.inegi_repository'
        ),
        inegi_path=config.paths.inegi,
        inegi_file=config.files.inegi
    )