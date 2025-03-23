"""
Esquema de datos para las fuentes de CONAPO.

Este módulo define las características temporales y demográficas
proporcionadas por CONAPO para el procesamiento de datos.
"""
from typing import List

# Variables temporales de CONAPO
TEMPORAL_FEATURES: List[str] = [
    # Variables demográficas básicas
    'POB_MIT_AÑO',  # Población total
    'EDAD_MED',  # Edad mediana
    'HOM_MIT_AÑO',  # Hombres a mitad de año
    'MUJ_MIT_AÑO',  # Mujeres a mitad de año

    # Indicadores de desarrollo y dinámica poblacional
    'CRE_NAT',  # Crecimiento natural
    'IND_ENV',  # Índice de envejecimiento
    'RAZ_DEP',  # Razón de dependencia

    # Indicadores de dinámica poblacional
    'T_CRE_NAT',  # Tasa de crecimiento natural
    'T_BRU_MOR',  # Tasa bruta de mortalidad
    'T_BRU_NAT',  # Tasa bruta de natalidad

    # Variables de población por edad
    'POB_15_49',  # Población en edad reproductiva
    'POB_30_64'  # Población adulta
]

# Variables derivadas específicas de datos CONAPO
CONAPO_DERIVED_FEATURES: List[str] = [
    'T_CRE_NAT',  # Tasa de crecimiento natural
    'T_BRU_MOR',  # Tasa bruta de mortalidad
    'T_BRU_NAT'  # Tasa bruta de natalidad
]

# Variables objetivo para modelos predictivos basados en datos CONAPO
CONAPO_TARGET_VARIABLES: List[str] = [
    'CRE_NAT',  # Crecimiento natural como variable objetivo principal
    'T_CRE_NAT'  # Tasa de crecimiento natural como variable objetivo secundaria
]