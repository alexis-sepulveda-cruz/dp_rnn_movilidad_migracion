"""
Generador de predicciones usando simulación Monte Carlo.
"""
import numpy as np
from typing import Tuple, List

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.prediction_generator_port import PredictionGeneratorPort
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory


class MonteCarloPredictor(PredictionGeneratorPort):
    """
    Implementación de PredictionGeneratorPort usando simulación Monte Carlo.
    
    Genera predicciones mediante simulación Monte Carlo, permitiendo
    estimar la incertidumbre asociada con cada predicción futura.
    """
    
    def __init__(self, cv_threshold: float = 0.2, batch_size: int = 20):
        """
        Inicializa el generador de predicciones Monte Carlo.
        
        Args:
            cv_threshold: Umbral de coeficiente de variación para regulación
            batch_size: Tamaño de lote para predicciones por batches
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.cv_threshold = cv_threshold
        self.batch_size = batch_size
        
    def generate_predictions(
        self, 
        model: any, 
        initial_sequence: np.ndarray, 
        future_steps: int,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Genera predicciones usando simulación Monte Carlo.
        
        Args:
            model: Modelo entrenado para realizar predicciones
            initial_sequence: Secuencia inicial para comenzar la predicción
            future_steps: Número de pasos futuros a predecir
            num_samples: Número de muestras Monte Carlo
            
        Returns:
            Tupla con (matriz de trayectorias [samples, steps], datos de incertidumbre)
        """
        self.logger.info(f"Generando {future_steps} predicciones con {num_samples} muestras Monte Carlo")
        
        # Almacenar todas las trayectorias Monte Carlo
        all_trajectories = np.zeros((num_samples, future_steps))
        current_sequence = initial_sequence.copy()
        uncertainty_data = []
        
        # Ajustar tamaño de batch si es necesario
        batch_size = min(self.batch_size, num_samples)
        
        for step in range(future_steps):
            self.logger.debug(f"Prediciendo paso {step+1} de {future_steps}")
            
            # Realizar múltiples predicciones en lotes
            predictions_batches = []
            
            for i in range(0, num_samples, batch_size):
                batch_end = min(i + batch_size, num_samples)
                batch_size_actual = batch_end - i
                sequence_batch = np.repeat([current_sequence], batch_size_actual, axis=0)
                
                # batch_predictions = model.predict(sequence_batch, verbose=0) (Determinístico)
                # CORRECCIÓN CRÍTICA: Usar llamada directa al modelo con training=True
                # para mantener el comportamiento estocástico del Dropout necesario para Monte Carlo
                batch_predictions = model(sequence_batch, training=True).numpy() # (Estocástico)
                predictions_batches.append(batch_predictions)
            
            # Combinar predicciones de todos los lotes
            predictions = np.concatenate(predictions_batches).flatten()
            
            # Monitorear varianza para estabilidad
            pred_std = predictions.std()
            pred_mean = predictions.mean()
            cv = (pred_std / abs(pred_mean)) if abs(pred_mean) > 1e-6 else 0
            
            uncertainty_data.append({
                'step': step,
                'mean': float(pred_mean),
                'std': float(pred_std),
                'cv': float(cv)
            })
            
            # Control de varianza para predicciones futuras
            if cv > self.cv_threshold:
                self.logger.warning(f"Varianza alta detectada en paso {step+1}, aplicando corrección")
                scaled_std = abs(pred_mean) * self.cv_threshold
                predictions = np.random.normal(pred_mean, scaled_std, num_samples)
            
            # Almacenar predicciones para este paso
            all_trajectories[:, step] = predictions
            
            # Actualizar secuencia para el siguiente paso
            current_sequence = self._update_sequence(current_sequence, pred_mean)
        
        return all_trajectories, uncertainty_data
    
    def _update_sequence(self, current_sequence: np.ndarray, new_value: float) -> np.ndarray:
        """
        Actualiza la secuencia desplazándola y añadiendo el nuevo valor.
        
        Args:
            current_sequence: Secuencia actual
            new_value: Nuevo valor a añadir
            
        Returns:
            Secuencia actualizada
        """
        # Crear copia para evitar modificar el original
        updated_sequence = current_sequence.copy()
        
        # Desplazar valores (eliminar el más antiguo)
        updated_sequence = np.roll(updated_sequence, -1, axis=0)
        
        # Añadir nuevo valor en la última posición
        feature_count = updated_sequence.shape[1]
        if feature_count > 1:
            # Si hay múltiples características, mantener las estáticas
            static_features = updated_sequence[-1, 1:]
            updated_sequence[-1] = np.concatenate([[new_value], static_features])
        else:
            updated_sequence[-1] = new_value
            
        return updated_sequence
