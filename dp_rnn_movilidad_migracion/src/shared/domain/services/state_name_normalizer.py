"""
Servicio de normalización de nombres de estados mexicanos.

Este servicio proporciona funcionalidad para normalizar y mapear
diferentes variantes de nombres de estados mexicanos.
"""
from typing import Dict, Optional
from dp_rnn_movilidad_migracion.src.shared.domain.value_objects.mexican_states import MexicanState

class StateNameNormalizer:
    """
    Servicio de dominio para normalizar nombres de estados mexicanos.
    
    Proporciona funcionalidad para mapear entre nombres oficiales completos,
    nombres cortos, y otras variantes de los nombres de estados mexicanos.
    """
    
    # Mapeo de nombres oficiales a valores enum
    _OFFICIAL_TO_ENUM = {
        "Aguascalientes": MexicanState.AGUASCALIENTES,
        "Baja California": MexicanState.BAJA_CALIFORNIA,
        "Baja California Sur": MexicanState.BAJA_CALIFORNIA_SUR,
        "Campeche": MexicanState.CAMPECHE,
        "Coahuila de Zaragoza": MexicanState.COAHUILA,
        "Colima": MexicanState.COLIMA,
        "Chiapas": MexicanState.CHIAPAS,
        "Chihuahua": MexicanState.CHIHUAHUA,
        "Ciudad de México": MexicanState.CIUDAD_DE_MEXICO,
        "Durango": MexicanState.DURANGO,
        "Guanajuato": MexicanState.GUANAJUATO,
        "Guerrero": MexicanState.GUERRERO,
        "Hidalgo": MexicanState.HIDALGO,
        "Jalisco": MexicanState.JALISCO,
        "México": MexicanState.MEXICO,
        "Michoacán de Ocampo": MexicanState.MICHOACAN,
        "Morelos": MexicanState.MORELOS,
        "Nayarit": MexicanState.NAYARIT,
        "Nuevo León": MexicanState.NUEVO_LEON,
        "Oaxaca": MexicanState.OAXACA,
        "Puebla": MexicanState.PUEBLA,
        "Querétaro": MexicanState.QUERETARO,
        "Quintana Roo": MexicanState.QUINTANA_ROO,
        "San Luis Potosí": MexicanState.SAN_LUIS_POTOSI,
        "Sinaloa": MexicanState.SINALOA,
        "Sonora": MexicanState.SONORA,
        "Tabasco": MexicanState.TABASCO,
        "Tamaulipas": MexicanState.TAMAULIPAS,
        "Tlaxcala": MexicanState.TLAXCALA,
        "Veracruz de Ignacio de la Llave": MexicanState.VERACRUZ,
        "Yucatán": MexicanState.YUCATAN,
        "Zacatecas": MexicanState.ZACATECAS,
        "República Mexicana": MexicanState.REPUBLICA_MEXICANA
    }
    
    # Mapeo inverso
    _ENUM_TO_OFFICIAL = {enum: name for name, enum in _OFFICIAL_TO_ENUM.items()}
    
    # Alias adicionales para mejorar el matching
    _ALIASES = {
        # Variantes comunes
        "Coahuila": "Coahuila de Zaragoza",
        "Michoacán": "Michoacán de Ocampo",
        "Veracruz": "Veracruz de Ignacio de la Llave",
        "Ciudad de Mexico": "Ciudad de México",
        "Mexico": "México",
        "Distrito Federal": "Ciudad de México",
        "Queretaro": "Querétaro",
        "San Luis Potosi": "San Luis Potosí",
        "Yucatan": "Yucatán",
        "Nuevo Leon": "Nuevo León",
        # Abreviaturas y aliases
        "CDMX": "Ciudad de México",
        "EdoMex": "México",
    }
    
    @classmethod
    def to_enum(cls, state_name: str) -> Optional[MexicanState]:
        """
        Convierte un nombre de estado (oficial o variante) a su enumeración equivalente.
        
        Args:
            state_name: Nombre del estado en cualquier variante
            
        Returns:
            Enumeración MexicanState correspondiente o None si no se encuentra
        """
        # Buscar directamente si es un nombre oficial
        if state_name in cls._OFFICIAL_TO_ENUM:
            return cls._OFFICIAL_TO_ENUM[state_name]
        
        # Normalizar a nombre oficial si es un alias
        if state_name in cls._ALIASES:
            official_name = cls._ALIASES[state_name]
            return cls._OFFICIAL_TO_ENUM.get(official_name)
        
        # Intento de matching de subcadena para nombres parciales
        for official_name, enum in cls._OFFICIAL_TO_ENUM.items():
            if state_name in official_name or official_name in state_name:
                return enum
        
        # Búsqueda case-insensitive como último recurso
        state_name_lower = state_name.lower()
        for official_name, enum in cls._OFFICIAL_TO_ENUM.items():
            if state_name_lower == official_name.lower():
                return enum
        
        return None
    
    @classmethod
    def to_short_name(cls, state_name: str) -> Optional[str]:
        """
        Convierte un nombre de estado a su forma corta estándar.
        
        Args:
            state_name: Nombre del estado en cualquier variante
            
        Returns:
            Nombre corto estándar o None si no se encuentra
        """
        enum = cls.to_enum(state_name)
        if enum:
            return enum.value
        return None
    
    @classmethod
    def to_official_name(cls, state_name: str) -> Optional[str]:
        """
        Convierte un nombre de estado a su nombre oficial completo.
        
        Args:
            state_name: Nombre del estado en cualquier variante
            
        Returns:
            Nombre oficial completo o None si no se encuentra
        """
        enum = cls.to_enum(state_name)
        if enum:
            return cls._ENUM_TO_OFFICIAL[enum]
        return None
    
    @classmethod
    def get_all_official_names(cls) -> Dict[str, str]:
        """
        Obtiene un diccionario de todos los nombres oficiales y sus versiones cortas.
        
        Returns:
            Diccionario con {nombre_oficial: nombre_corto}
        """
        return {official: enum.value for official, enum in cls._OFFICIAL_TO_ENUM.items()}
