"""
Esquema de datos para las fuentes de INEGI.

Este módulo define las características numéricas y categóricas
proporcionadas por INEGI para el procesamiento de datos.
"""
import numpy as np
from typing import Dict, List, Any, Union

# Variables numéricas básicas del INEGI
NUMERIC_FEATURES: List[str] = [
    'PEA', 'PDESOCUP', 'PE_INAC',           # Económicos básicos
    'P15YM_AN', 'P3YM_HLI', 'PHOG_IND',     # Sociales
    'VPH_NDEAED', 'VPH_S_ELEC', 'VPH_AGUAFV', 'VPH_NODREN',  # Vivienda
    'PDER_IMSS', 'PDER_ISTE',               # Servicios de salud
    'GRAPROES',                             # Educación
]

# Características categóricas del INEGI y su procesamiento
CATEGORICAL_FEATURES: Dict[str, Union[Dict[str, Any], List[str]]] = {
    'binary': {
        'C_CONFLICTO1': {1: 1, 2: 0, 9: np.nan},
        'C_CONFLICTO4': {7: 1, 8: 0, 9: np.nan},
        'C_DIS_TRANS': {1: 1, 2: 0, 9: np.nan},
        'C_ALUMBRADO': {1: 1, 2: 0, 9: np.nan}
    },
    'ordinal': {
        'C_RECUBCOB': {
            'mapping': {
                1: 5,    # todas las calles -> mejor valor (5)
                2: 4,    # mayoría de las calles -> (4)
                3: 3,    # mitad de las calles -> (3)
                4: 2,    # pocas calles -> (2)
                5: 1,    # ninguna calle -> peor valor (1)
                9: np.nan    # No especificado -> NaN
            }
        }
    },
    'nominal_prefixes': {
        'ACT_PRIN_': 'C_ACT_PRIN',
        'LTRABAJO_': 'C_LTRABAJO',
        'PROBLEMA_': 'C_PROBLEMA'
    }
}

# Variables derivadas específicas de INEGI
INEGI_DERIVED_FEATURES: Dict[str, List[str]] = {
    'tasas': [
        'TASA_DESEMPLEO',
        'TASA_INACTIVIDAD',
        'TASA_ANALFABETISMO',
        'TASA_INDIGENA',
        'TASA_CARENCIA_SERVICIOS'
    ],
    'indices': [
        'INDICE_INFRAESTRUCTURA',
        'INDICE_CONFLICTOS'
    ]
}

# Lista plana de características categóricas
CATEGORICAL_FEATURES_FLAT: List[str] = (
    list(CATEGORICAL_FEATURES['binary'].keys()) +
    list(CATEGORICAL_FEATURES['ordinal'].keys())
)

# Lista completa de características estáticas de INEGI
INEGI_STATIC_FEATURES: List[str] = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES_FLAT +
    INEGI_DERIVED_FEATURES['tasas'] +
    INEGI_DERIVED_FEATURES['indices']
)