"""
application_container.py - Contenedor de aplicación centralizado
"""
import os
from dependency_injector import containers, providers

from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.conapo_repository import ConapoRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.repositories.inegi_repository import InegiRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.preprocessors.conapo_preprocessor import ConapoPreprocessor
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.infrastructure.persistence.preprocessors.inegi_preprocessor import InegiPreprocessor
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.conapo_data_service import ConapoDataService
from dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.inegi_data_service import InegiDataService
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.adapters.normalizer.sklearn_normalizer import SklearnNormalizer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.adapters.data_preparers.model_data_preparation_service import ModelDataPreparationService
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.adapters.model_builders.tensorflow_rnn_model_builder import TensorflowRNNModelBuilder
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.adapters.model_trainers.tensorflow_model_trainer import TensorflowModelTrainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.adapters.visualizers.matplotlib_visualizer import MatplotlibVisualizer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.infrastructure.repositories.file_prediction_repository import FilePredictionRepository
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.services.migration_prediction_service import MigrationPredictionService


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
    
    inegi_preprocessor = providers.Singleton(
        InegiPreprocessor
    )
    
    # Servicios de datos
    conapo_data_service = providers.Singleton(
        ConapoDataService,
        repository=conapo_repository,
        preprocessor=conapo_preprocessor
    )

    inegi_data_service = providers.Singleton(
        InegiDataService,
        repository=inegi_repository,
        preprocessor=inegi_preprocessor
    )
    
    # Normalización para modelos
    temporal_normalizer = providers.Factory(
        SklearnNormalizer,
        feature_range=(0, 1)
    )
    
    target_normalizer = providers.Factory(
        SklearnNormalizer,
        feature_range=(-1, 1)
    )
    
    static_normalizer = providers.Factory(
        SklearnNormalizer,
        feature_range=(0, 1)
    )
    
    # Servicio de preparación de datos para modelos
    model_data_preparation_service = providers.Singleton(
        ModelDataPreparationService,
        temporal_normalizer=temporal_normalizer,
        target_normalizer=target_normalizer,
        static_normalizer=static_normalizer,
        random_seed=config.data.random_seed
    )

    # Adaptadores
    model_builder = providers.Factory(
        TensorflowRNNModelBuilder,
        random_seed=config.data.random_seed
    )
    
    model_trainer = providers.Factory(
        TensorflowModelTrainer
    )

    visualization_dir = os.path.join('/Users/armandoalexissepulveda/Documents/Maestria_Ciencia_Datos/Tesis/dp_rnn_movilidad_migracion', 'graficos', 'monte_carlo')

    
    visualizer = providers.Factory(
        MatplotlibVisualizer,
        output_dir=visualization_dir
    )
    
    prediction_output_dir = os.path.join('/Users/armandoalexissepulveda/Documents/Maestria_Ciencia_Datos/Tesis/dp_rnn_movilidad_migracion', 'resultados', 'predicciones')
    
    prediction_repository = providers.Factory(
        FilePredictionRepository,
        output_dir=prediction_output_dir
    )
    
    # Servicios
    migration_prediction_service = providers.Factory(
        MigrationPredictionService,
        model_builder=model_builder,
        model_trainer=model_trainer,
        data_preparer=model_data_preparation_service,
        prediction_repository=prediction_repository,
        visualizer=visualizer,
    )
