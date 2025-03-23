import pandas as pd
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory

"""
DTO para solicitudes de predicción de migración.

Este módulo contiene la clase PredictionRequestDTO que encapsula
los parámetros necesarios para realizar una solicitud de predicción.
"""


class PredictionRequestDTO:
    """
    Objeto de transferencia de datos (DTO) para solicitudes de predicción de migración.
    
    Esta clase encapsula todos los parámetros necesarios para solicitar una
    predicción de migración, proporcionando una estructura coherente
    para la comunicación entre la capa de presentación y la capa de aplicación.
    
    Attributes:
        state (str): Nombre o identificador del estado/entidad para 
            el cual se solicita la predicción.
            
        future_years (int): Número de años futuros para los que se
            requiere la predicción. Por defecto es 5.
            
        monte_carlo_samples (int): Número de muestras Monte Carlo a utilizar
            para estimar la incertidumbre de la predicción. Por defecto es 100.
            
        target_variable (str): Variable objetivo a predecir (por ejemplo,
            'CRE_NAT' para crecimiento natural). Por defecto es 'CRE_NAT'.
            
        confidence_level (float): Nivel de confianza para los intervalos
            de predicción, expresado como un valor entre 0 y 1. Por defecto es 0.95
            para intervalos de confianza del 95%.
    """

    def __init__(
        self, 
        state: str,
        future_years: int = 5,
        monte_carlo_samples: int = 100,
        target_variable: str = 'CRE_NAT',
        confidence_level: float = 0.95
    ):
        """
        Inicializa un nuevo objeto de solicitud de predicción.
        
        Args:
            state: Nombre o identificador del estado/entidad
            future_years: Número de años futuros a predecir
            monte_carlo_samples: Número de muestras Monte Carlo
            target_variable: Variable objetivo a predecir
            confidence_level: Nivel de confianza para intervalos
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.state = state
        self.future_years = future_years
        self.monte_carlo_samples = monte_carlo_samples
        self.target_variable = target_variable
        self.confidence_level = confidence_level
        
        self.logger.info(f"Solicitud de predicción creada para {state}, "
                        f"años futuros: {future_years}, "
                        f"variable: {target_variable}")
        self.logger.debug(f"Parámetros avanzados - muestras MC: {monte_carlo_samples}, "
                         f"nivel de confianza: {confidence_level}")
    
    def validate(self) -> bool:
        """
        Valida que los parámetros de solicitud sean coherentes.
        
        Verifica que los valores estén dentro de rangos aceptables
        para realizar una predicción válida.
        
        Returns:
            bool: True si todos los parámetros son válidos, False en caso contrario
        """
        self.logger.debug(f"Validando solicitud de predicción para {self.state}")
        
        if not self.state or not isinstance(self.state, str):
            self.logger.error(f"Validación fallida: Estado inválido '{self.state}'")
            return False
        
        if not isinstance(self.future_years, int) or self.future_years < 1 or self.future_years > 50:
            self.logger.error(f"Validación fallida: Años futuros inválidos {self.future_years}")
            return False
            
        if not isinstance(self.monte_carlo_samples, int) or self.monte_carlo_samples < 50:
            self.logger.error(f"Validación fallida: Muestras Monte Carlo inválidas {self.monte_carlo_samples}")
            return False
            
        if not isinstance(self.confidence_level, float) or self.confidence_level <= 0 or self.confidence_level >= 1:
            self.logger.error(f"Validación fallida: Nivel de confianza inválido {self.confidence_level}")
            return False
        
        self.logger.info(f"Solicitud de predicción validada correctamente para {self.state}")
        return True