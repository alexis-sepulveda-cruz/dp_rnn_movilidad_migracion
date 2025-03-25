"""
Exportador de datos para Power BI.

Este módulo proporciona funcionalidad para exportar datos históricos
y predicciones en un formato optimizado para visualización en Power BI.
"""
import os
import pandas as pd
from typing import Dict, List, Set, Any

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.prediction_repository_port import PredictionRepositoryPort
from dp_rnn_movilidad_migracion.src.shared.domain.value_objects.mexican_states import MexicanState
from dp_rnn_movilidad_migracion.src.shared.domain.services.state_selection_service import StateSelectionService
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class PowerBIDataExporter:
    """
    Exportador de datos para Power BI.
    
    Esta clase consolida datos históricos y predicciones en un formato
    adecuado para análisis en Power BI, incluyendo columnas de clasificación
    para filtrado avanzado.
    """
    
    def __init__(
        self, 
        prediction_repository: PredictionRepositoryPort, 
        output_dir: str
    ):
        """
        Inicializa el exportador de datos.
        
        Args:
            prediction_repository: Repositorio para acceder a predicciones guardadas
            output_dir: Directorio donde se guardarán los archivos exportados
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.prediction_repository = prediction_repository
        self.output_dir = output_dir
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear clasificaciones de estados
        self._create_state_classifications()
        
        self.logger.info(f"Exportador Power BI inicializado. Directorio: {output_dir}")
        
        # Verificar directorio del repositorio de predicciones
        self.logger.debug(f"Repositorio de predicciones configurado con directorio: {self.prediction_repository.output_dir}")
        if not os.path.exists(self.prediction_repository.output_dir):
            self.logger.warning(f"⚠️ El directorio del repositorio de predicciones no existe: {self.prediction_repository.output_dir}")
        else:
            prediction_files = [f for f in os.listdir(self.prediction_repository.output_dir) if f.endswith('.csv')]
            self.logger.debug(f"Archivos de predicción encontrados: {len(prediction_files)}")
            if prediction_files:
                self.logger.debug(f"Ejemplo de archivo: {prediction_files[0]}")
    
    def _create_state_classifications(self):
        """Crea diccionarios de clasificación de estados para filtrado rápido."""
        # Estados fronterizos del norte
        self.border_states = set(MexicanState.get_border_states())
        
        # Estados del sureste
        self.southeastern_states = set(MexicanState.get_southeastern_states())
        
        # Estados para comparación económica
        self.economic_comparison_states = set(StateSelectionService.get_economic_comparison_states())
        
        # Estados en corredores migratorios
        self.migration_corridor_states = set(StateSelectionService.get_migration_corridor_states())
        
        # Regiones
        self.regions = StateSelectionService.group_states_by_region()
        self.state_to_region = {}
        for region, states in self.regions.items():
            for state in states:
                self.state_to_region[state] = region
    
    def export_consolidated_data(self, historical_data: pd.DataFrame) -> str:
        """
        Exporta datos históricos y predicciones a un único archivo CSV.
        
        Args:
            historical_data: DataFrame con datos históricos de CONAPO
            
        Returns:
            Ruta al archivo CSV generado
        """
        self.logger.info("Iniciando exportación de datos consolidados")
        
        # 1. Preparar datos históricos
        historical_df = self._prepare_historical_data(historical_data)
        self.logger.info(f"Datos históricos preparados: {len(historical_df)} registros")
        
        # 2. Obtener todas las predicciones disponibles
        predictions_df = self._prepare_prediction_data()
        
        if len(predictions_df) == 0:
            self.logger.warning("⚠️ No se encontraron datos de predicción para exportar")
            self.logger.info("Verificando archivos en el directorio del repositorio...")
            
            repo_dir = self.prediction_repository.output_dir
            if os.path.exists(repo_dir):
                files = os.listdir(repo_dir)
                prediction_files = [f for f in files if '_prediccion.csv' in f]
                metadata_files = [f for f in files if '_metadata.json' in f]
                
                self.logger.info(f"Archivos CSV de predicción encontrados: {len(prediction_files)}")
                if prediction_files:
                    self.logger.info(f"Ejemplos: {prediction_files[:3]}")
                
                self.logger.info(f"Archivos JSON de metadatos encontrados: {len(metadata_files)}")
                if metadata_files:
                    self.logger.info(f"Ejemplos: {metadata_files[:3]}")
                    
                # Intentar cargar un archivo de ejemplo para diagnosticar
                if prediction_files:
                    try:
                        example_file = os.path.join(repo_dir, prediction_files[0])
                        example_df = pd.read_csv(example_file)
                        self.logger.info(f"Contenido de ejemplo: {example_df.shape}")
                        self.logger.info(f"Columnas: {example_df.columns.tolist()}")
                    except Exception as e:
                        self.logger.error(f"Error al leer archivo de ejemplo: {str(e)}")
            else:
                self.logger.error(f"El directorio del repositorio no existe: {repo_dir}")
        else:
            self.logger.info(f"Datos de predicciones preparados: {len(predictions_df)} registros")
        
        # 3. Concatenar datos históricos y predicciones
        combined_df = pd.concat([historical_df, predictions_df], ignore_index=True)
        
        # 4. Agregar columnas de clasificación
        enriched_df = self._add_classification_columns(combined_df)
        self.logger.info(f"Datos consolidados y enriquecidos: {len(enriched_df)} registros")
        
        # 5. Guardar en CSV
        output_path = os.path.join(self.output_dir, "powerbi_migration_data.csv")
        enriched_df.to_csv(output_path, index=False)
        self.logger.info(f"Datos exportados a: {output_path}")
        
        return output_path
    
    def _prepare_historical_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos históricos, filtrando y agregando el campo 'Tipo'.
        
        Args:
            historical_data: DataFrame con datos históricos de CONAPO
            
        Returns:
            DataFrame procesado con datos históricos
        """
        # Seleccionar solo las columnas necesarias y filtrar por años
        df = historical_data[['ENTIDAD', 'AÑO', 'CRE_NAT']].copy()
        
        # Renombrar columnas para consistencia
        df = df.rename(columns={
            'ENTIDAD': 'Estado',
            'AÑO': 'Año',
            'CRE_NAT': 'ValorCRE_NAT'
        })
        
        # Agregar columna 'Tipo'
        df['Tipo'] = 'Histórico'
        
        # Agregar columnas nulas para consistencia con datos de predicción
        df['LímiteInferior'] = None
        df['LímiteSuperior'] = None
        df['DesviaciónEstándar'] = None
        df['ScoreConfiabilidad'] = None
        
        return df
    
    def _prepare_prediction_data(self) -> pd.DataFrame:
        """
        Prepara los datos de predicción, extrayéndolos del repositorio.
        
        Returns:
            DataFrame con datos de predicciones
        """
        # Obtener todos los estados
        states = MexicanState.get_all_states()
        
        prediction_rows = []
        loaded_states = []
        failed_states = []
        
        # Intentar cargar predicciones para cada estado
        for state in states:
            try:
                self.logger.debug(f"Intentando cargar predicción para {state}...")
                prediction = self.prediction_repository.get_prediction(state)
                
                # Crear filas para cada año de predicción
                for i, year in enumerate(prediction.years):
                    prediction_rows.append({
                        'Estado': state,
                        'Año': year,
                        'ValorCRE_NAT': prediction.values[i],
                        'Tipo': 'Predicción',
                        'LímiteInferior': prediction.lower_bounds[i],
                        'LímiteSuperior': prediction.upper_bounds[i],
                        'DesviaciónEstándar': prediction.std_devs[i],
                        'ScoreConfiabilidad': prediction.uncertainty_metrics.reliability_score
                    })
                
                loaded_states.append(state)
                self.logger.debug(f"✅ Predicciones cargadas para {state}: {len(prediction.years)} años")
                
            except FileNotFoundError:
                failed_states.append(state)
                self.logger.warning(f"❌ No se encontraron predicciones para {state}")
            except Exception as e:
                failed_states.append(state)
                self.logger.error(f"❌ Error al cargar predicciones para {state}: {str(e)}")
        
        self.logger.info(f"Estados con predicciones cargadas ({len(loaded_states)}): {', '.join(loaded_states) if loaded_states else 'Ninguno'}")
        self.logger.info(f"Estados sin predicciones ({len(failed_states)}): {len(failed_states)} estados")
        
        # Crear DataFrame
        if not prediction_rows:
            self.logger.warning("⚠️ No se encontraron predicciones para ningún estado")
            return pd.DataFrame(columns=[
                'Estado', 'Año', 'ValorCRE_NAT', 'Tipo', 'LímiteInferior', 
                'LímiteSuperior', 'DesviaciónEstándar', 'ScoreConfiabilidad'
            ])
        
        return pd.DataFrame(prediction_rows)
    
    def _add_classification_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega columnas de clasificación para filtrado en Power BI.
        
        Args:
            df: DataFrame con datos históricos y predicciones
            
        Returns:
            DataFrame enriquecido con columnas de clasificación
        """
        # Clasificar estados fronterizos
        df['EsFronterizo'] = df['Estado'].apply(lambda x: x in self.border_states)
        
        # Clasificar estados del sureste
        df['EsSureste'] = df['Estado'].apply(lambda x: x in self.southeastern_states)
        
        # Clasificar estados para comparación económica
        df['EsComparaciónEconómica'] = df['Estado'].apply(lambda x: x in self.economic_comparison_states)
        
        # Clasificar estados en corredores migratorios
        df['EsCorredorMigratorio'] = df['Estado'].apply(lambda x: x in self.migration_corridor_states)
        
        # Asignar región
        df['Región'] = df['Estado'].apply(lambda x: self.state_to_region.get(x, "Sin clasificar"))
        
        return df
