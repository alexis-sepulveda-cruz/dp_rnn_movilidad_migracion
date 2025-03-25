"""
Script para generar predicciones de ejemplo.

Este script entrena el modelo y genera predicciones para un conjunto
selecto de estados, para asegurar que haya datos disponibles para exportar
a Power BI.
"""
from dependency_injector.wiring import inject, Provide

from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import PredictionRequestDTO
from dp_rnn_movilidad_migracion.src.shared.domain.services.state_selection_service import StateSelectionService
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


@inject
def generate_example_predictions(
    conapo_service = Provide[ApplicationContainer.conapo_data_service],
    inegi_service = Provide[ApplicationContainer.inegi_data_service],
    migration_prediction_service = Provide[ApplicationContainer.migration_prediction_service]
):
    """
    Genera predicciones de ejemplo para unos pocos estados representativos.
    
    Esta función:
    1. Carga datos de CONAPO e INEGI
    2. Entrena el modelo
    3. Genera predicciones para estados seleccionados
    """
    logger = LoggerFactory.get_composite_logger(__name__)
    logger.info("Iniciando generación de predicciones de ejemplo...")
    
    # Cargar datos
    logger.info("Cargando datos...")
    conapo_data = conapo_service.get_processed_data()
    inegi_data = inegi_service.get_processed_data()
    logger.info(f"Datos cargados: CONAPO ({conapo_data.shape[0]} filas), INEGI ({inegi_data.shape[0]} filas)")
    
    # Entrenar modelo
    logger.info("Entrenando modelo...")
    history = migration_prediction_service.train_model(
        temporal_data=conapo_data,
        static_data=inegi_data,
        epochs=10  # Reducido para ejemplo
    )
    logger.info(f"Entrenamiento completado con {len(history['loss'])} épocas")
    
    # Seleccionar estados representativos
    states = StateSelectionService.get_representative_states()
    logger.info(f"Generando predicciones para {len(states)} estados: {', '.join(states)}")
    
    # Generar predicciones
    successful_predictions = []
    for state in states:
        try:
            logger.info(f"Prediciendo para {state}...")
            prediction = migration_prediction_service.predict_migration(
                request=PredictionRequestDTO(
                    state=state,
                    future_years=5,
                    monte_carlo_samples=50,  # Reducido para ejemplo
                    confidence_level=0.95
                ),
                temporal_data=conapo_data,
                static_data=inegi_data
            )
            successful_predictions.append(state)
            logger.info(f"✅ Predicción completada para {state}")
        except Exception as e:
            logger.error(f"❌ Error al predecir para {state}: {str(e)}")
    
    logger.info("=== RESUMEN ===")
    logger.info(f"Predicciones exitosas: {len(successful_predictions)} de {len(states)}")
    if successful_predictions:
        logger.info(f"Estados con predicciones: {', '.join(successful_predictions)}")
    logger.info("===============")
    logger.info("Ahora puedes ejecutar el script export_to_power_bi.py para exportar estos datos a CSV.")

if __name__ == "__main__":
    # Inicializar la aplicación
    container = bootstrap_app()
    
    # Configurar inyección de dependencias
    container.wire(modules=[__name__])
    
    # Generar predicciones
    generate_example_predictions()
