"""
Servicio de dominio para selección de estados.

Este servicio proporciona operaciones relacionadas con la selección
y agrupación de estados mexicanos para análisis.
"""
from typing import List, Dict, Set
from dp_rnn_movilidad_migracion.src.shared.domain.value_objects.mexican_states import MexicanState


class StateSelectionService:
    """
    Servicio de dominio para selección y agrupación de estados.
    
    Proporciona métodos para seleccionar conjuntos representativos de estados
    para diferentes tipos de análisis comparativos.
    """
    
    @staticmethod
    def get_representative_states() -> List[str]:
        """
        Obtiene un conjunto de estados representativos para análisis generales.
        
        Selecciona estados con diferentes características demográficas,
        económicas y geográficas para proporcionar una visión general
        del comportamiento a nivel nacional.
        
        Returns:
            Lista con nombres de estados representativos
        """
        representative_states = [
            MexicanState.CIUDAD_DE_MEXICO.value,
            MexicanState.JALISCO.value,
            MexicanState.NUEVO_LEON.value,
            MexicanState.CHIAPAS.value
        ]
        return representative_states
    
    @staticmethod
    def get_economic_comparison_states() -> List[str]:
        """
        Obtiene estados para comparación económica.
        
        Selecciona estados con diferentes niveles de desarrollo económico,
        útiles para analizar el impacto de factores económicos.
        
        Returns:
            Lista con nombres de estados para comparación económica
        """
        return [
            MexicanState.CIUDAD_DE_MEXICO.value,
            MexicanState.NUEVO_LEON.value,
            MexicanState.CHIAPAS.value,
            MexicanState.GUERRERO.value,
            MexicanState.OAXACA.value
        ]
    
    @staticmethod
    def get_migration_corridor_states() -> List[str]:
        """
        Obtiene estados que forman parte de corredores migratorios importantes.
        
        Returns:
            Lista con nombres de estados en corredores migratorios
        """
        return [
            MexicanState.CHIAPAS.value,
            MexicanState.TABASCO.value,
            MexicanState.VERACRUZ.value,
            MexicanState.CIUDAD_DE_MEXICO.value,
            MexicanState.BAJA_CALIFORNIA.value,
            MexicanState.SONORA.value,
            MexicanState.TAMAULIPAS.value
        ]
    
    @staticmethod
    def group_states_by_region() -> Dict[str, List[str]]:
        """
        Agrupa estados por región geográfica.
        
        Returns:
            Diccionario con regiones como claves y listas de estados como valores
        """
        return {
            "Norte": [
                MexicanState.BAJA_CALIFORNIA.value,
                MexicanState.BAJA_CALIFORNIA_SUR.value,
                MexicanState.SONORA.value,
                MexicanState.CHIHUAHUA.value,
                MexicanState.COAHUILA.value,
                MexicanState.NUEVO_LEON.value,
                MexicanState.TAMAULIPAS.value,
                MexicanState.DURANGO.value,
                MexicanState.SINALOA.value,
                MexicanState.ZACATECAS.value
            ],
            "Centro": [
                MexicanState.AGUASCALIENTES.value,
                MexicanState.COLIMA.value,
                MexicanState.GUANAJUATO.value,
                MexicanState.HIDALGO.value,
                MexicanState.JALISCO.value,
                MexicanState.MEXICO.value,
                MexicanState.MICHOACAN.value,
                MexicanState.MORELOS.value,
                MexicanState.NAYARIT.value,
                MexicanState.QUERETARO.value,
                MexicanState.SAN_LUIS_POTOSI.value,
                MexicanState.TLAXCALA.value,
                MexicanState.CIUDAD_DE_MEXICO.value
            ],
            "Sur-Sureste": [
                MexicanState.CAMPECHE.value,
                MexicanState.CHIAPAS.value,
                MexicanState.GUERRERO.value,
                MexicanState.OAXACA.value,
                MexicanState.PUEBLA.value,
                MexicanState.QUINTANA_ROO.value,
                MexicanState.TABASCO.value,
                MexicanState.VERACRUZ.value,
                MexicanState.YUCATAN.value
            ]
        }
