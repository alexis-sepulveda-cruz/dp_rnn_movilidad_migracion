"""
Script para exportar datos para análisis en Power BI.

Este script extrae datos históricos y predicciones, los combina y exporta
en un formato optimizado para visualización y análisis en Power BI.
"""
from dependency_injector.wiring import inject, Provide

from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


@inject
def export_data_for_power_bi(
    conapo_data_service = Provide[ApplicationContainer.conapo_data_service],
    power_bi_exporter = Provide[ApplicationContainer.power_bi_exporter]
):
    """
    Exporta datos consolidados para análisis en Power BI.
    
    Esta función:
    1. Carga datos históricos de CONAPO
    2. Extrae predicciones desde el repositorio
    3. Combina ambos conjuntos de datos
    4. Agrega columnas de clasificación para filtrado
    5. Exporta todo a CSV
    """
    logger = LoggerFactory.get_composite_logger(__name__)
    logger.info("Iniciando exportación de datos para Power BI...")
    
    # Cargar datos históricos de CONAPO
    logger.info("Cargando datos históricos de CONAPO...")
    historical_data = conapo_data_service.get_processed_data()
    logger.info(f"Datos históricos cargados: {len(historical_data)} registros")
    
    # Exportar datos combinados
    output_path = power_bi_exporter.export_consolidated_data(historical_data)
    
    # Mostrar información de diagnóstico sobre el archivo generado
    try:
        import pandas as pd
        exported_data = pd.read_csv(output_path)
        total_records = len(exported_data)
        historical_records = len(exported_data[exported_data['Tipo'] == 'Histórico'])
        prediction_records = len(exported_data[exported_data['Tipo'] == 'Predicción'])
        states_count = exported_data['Estado'].nunique()
        
        logger.info("=== RESUMEN DE EXPORTACIÓN ===")
        logger.info(f"Registros totales: {total_records}")
        logger.info(f"Registros históricos: {historical_records}")
        logger.info(f"Registros de predicción: {prediction_records}")
        logger.info(f"Número de estados: {states_count}")
        logger.info("============================")
        
        if prediction_records == 0:
            logger.warning("⚠️ No se encontraron registros de predicción. Verifica que:")
            logger.warning("  1. Has ejecutado predicciones para al menos un estado")
            logger.warning("  2. Las predicciones están guardadas en el directorio configurado")
            logger.warning(f"  3. El repositorio puede acceder a la ruta correcta")
    except Exception as e:
        logger.error(f"Error al analizar el archivo exportado: {str(e)}")
    
    logger.info(f"\n¡Exportación completada!")
    logger.info(f"Los datos están disponibles en: {output_path}")
    logger.info("\nInformación para uso en Power BI:")
    logger.info("- Filtra por 'Tipo' para distinguir entre datos históricos y predicciones")
    logger.info("- Utiliza las columnas 'EsFronterizo', 'EsSureste', etc. para segmentar estados")
    logger.info("- La columna 'Región' permite análisis por zonas geográficas")
    logger.info("- Para visualizar intervalos de confianza, usa 'LímiteInferior' y 'LímiteSuperior'")


if __name__ == "__main__":
    # Inicializar la aplicación
    container = bootstrap_app()
    
    # Configurar inyección de dependencias
    container.wire(modules=[__name__])
    
    # Ejecutar exportación
    export_data_for_power_bi()
