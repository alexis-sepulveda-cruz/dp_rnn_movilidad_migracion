"""
application_container.py - Contenedor de aplicación centralizado
"""
from dependency_injector import containers, providers

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.conapo_repository import ConapoRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.inegi_repository import InegiRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.preprocessors.conapo_preprocessor import ConapoPreprocessor


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

    # Repositorios
    conapo_repository = providers.Singleton(
        ConapoRepository,
        conapo_path=config.paths.conapo,
        conapo_file=config.files.conapo,
    )

    inegi_repository = providers.Singleton(
        InegiRepository,
        inegi_path=config.paths.inegi,
        inegi_file=config.files.inegi
    )

    # Preprocesadores
    conapo_preprocessor = providers.Singleton(
        ConapoPreprocessor,
        start_year=config.data.years.start,
        end_year=config.data.years.end,
        include_derived=True,
        include_targets=True
    )