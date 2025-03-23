"""
Puerto para la generación de predicciones.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List


class PredictionGeneratorPort(ABC):
    """
    Interfaz para generadores de predicciones.
    
    Abstrae la lógica de generación de predicciones, permitiendo diferentes
    implementaciones (Monte Carlo, otros métodos probabilísticos, etc.).
    """
    
    @abstractmethod
    def generate_predictions(
        self, 
        model: any, 
        initial_sequence: np.ndarray, 
        future_steps: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Genera predicciones para pasos futuros con estimación de incertidumbre.
        
        Args:
            model: Modelo entrenado para realizar las predicciones
            initial_sequence: Secuencia inicial para comenzar la predicción
            future_steps: Número de pasos futuros a predecir
            num_samples: Número de muestras para estimar incertidumbre
            
        Returns:
            Tupla con (matriz de trayectorias [samples, steps], datos de incertidumbre)
        """
        pass
