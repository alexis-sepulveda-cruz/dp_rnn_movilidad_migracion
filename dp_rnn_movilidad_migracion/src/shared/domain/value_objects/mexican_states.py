"""
Value Object que representa los estados de México.

Este módulo proporciona una enumeración de estados mexicanos, permitiendo
acceder a ellos de manera consistente y tipada a través de la aplicación.
"""
from enum import Enum
from typing import List, Optional


class MexicanState(Enum):
    """
    Enumeración que representa los estados de México.
    
    Permite acceder a los nombres de estados de manera consistente,
    evitando errores de tipo o discrepancias en la representación textual.
    """
    # Estado especial para el total nacional
    REPUBLICA_MEXICANA = "República Mexicana"
    
    # Estados en orden alfabético
    AGUASCALIENTES = "Aguascalientes"
    BAJA_CALIFORNIA = "Baja California"
    BAJA_CALIFORNIA_SUR = "Baja California Sur"
    CAMPECHE = "Campeche"
    COAHUILA = "Coahuila"
    COLIMA = "Colima"
    CHIAPAS = "Chiapas"
    CHIHUAHUA = "Chihuahua"
    CIUDAD_DE_MEXICO = "Ciudad de México"
    DURANGO = "Durango"
    GUANAJUATO = "Guanajuato"
    GUERRERO = "Guerrero"
    HIDALGO = "Hidalgo"
    JALISCO = "Jalisco"
    MEXICO = "México"
    MICHOACAN = "Michoacán"
    MORELOS = "Morelos"
    NAYARIT = "Nayarit"
    NUEVO_LEON = "Nuevo León"
    OAXACA = "Oaxaca"
    PUEBLA = "Puebla"
    QUERETARO = "Querétaro"
    QUINTANA_ROO = "Quintana Roo"
    SAN_LUIS_POTOSI = "San Luis Potosí"
    SINALOA = "Sinaloa"
    SONORA = "Sonora"
    TABASCO = "Tabasco"
    TAMAULIPAS = "Tamaulipas"
    TLAXCALA = "Tlaxcala"
    VERACRUZ = "Veracruz"
    YUCATAN = "Yucatán"
    ZACATECAS = "Zacatecas"
    
    @classmethod
    def get_all_states(cls) -> List[str]:
        """
        Obtiene los nombres de todos los estados (excluyendo República Mexicana).
        
        Returns:
            Lista con los nombres de todos los estados mexicanos.
        """
        return [state.value for state in cls if state != cls.REPUBLICA_MEXICANA]
    
    @classmethod
    def get_border_states(cls) -> List[str]:
        """
        Obtiene los estados fronterizos del norte.
        
        Returns:
            Lista con los nombres de los estados fronterizos.
        """
        border_states = [
            cls.BAJA_CALIFORNIA, 
            cls.SONORA, 
            cls.CHIHUAHUA, 
            cls.COAHUILA, 
            cls.NUEVO_LEON, 
            cls.TAMAULIPAS
        ]
        return [state.value for state in border_states]
    
    @classmethod
    def get_southeastern_states(cls) -> List[str]:
        """
        Obtiene los estados del sureste mexicano.
        
        Returns:
            Lista con los nombres de los estados del sureste.
        """
        southeastern_states = [
            cls.CAMPECHE,
            cls.CHIAPAS,
            cls.OAXACA,
            cls.QUINTANA_ROO,
            cls.TABASCO,
            cls.VERACRUZ,
            cls.YUCATAN
        ]
        return [state.value for state in southeastern_states]
    
    @classmethod
    def get_state_by_name(cls, name: str) -> Optional['MexicanState']:
        """
        Obtiene un estado a partir de su nombre.
        
        Args:
            name: Nombre del estado a buscar
            
        Returns:
            Objeto MexicanState correspondiente o None si no se encuentra
        """
        for state in cls:
            if state.value.lower() == name.lower():
                return state
        return None
    
    @classmethod
    def is_valid_state(cls, name: str) -> bool:
        """
        Verifica si un nombre corresponde a un estado válido.
        
        Args:
            name: Nombre del estado a verificar
            
        Returns:
            True si es un estado válido, False en caso contrario
        """
        return cls.get_state_by_name(name) is not None
