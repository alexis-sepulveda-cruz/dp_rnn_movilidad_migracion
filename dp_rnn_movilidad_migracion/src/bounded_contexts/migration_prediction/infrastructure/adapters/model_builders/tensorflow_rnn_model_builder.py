import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.domain.ports.model_builder_port import ModelBuilderPort
from dp_rnn_movilidad_migracion.src.shared.infrastructure.factories.logger_factory import LoggerFactory

class TensorflowRNNModelBuilder(ModelBuilderPort):
    """Implementación con TensorFlow de ModelBuilderPort para construir modelos RNN."""

    def __init__(self, random_seed: int = 42):
        """
        Inicializa el constructor de modelos RNN.
        
        Args:
            random_seed: Semilla para reproducibilidad
        """
        self.logger = LoggerFactory.get_composite_logger(__name__)
        self.random_seed = random_seed
        
        # Establecer semillas para reproducibilidad
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)
        
        # Activar determinismo si está disponible
        try:
            tf.config.experimental.enable_op_determinism()
        except:
            self.logger.warning("No se pudo activar determinismo en TensorFlow")
        
        self.logger.info("Constructor de modelos TensorFlow RNN inicializado")

    def build_model(self, input_shape: tuple) -> tf.keras.Model:
        """
        Construye un modelo RNN con arquitectura LSTM.
        
        Args:
            input_shape: Forma de los datos de entrada
            
        Returns:
            Modelo de Keras compilado
        """
        self.logger.info(f"Construyendo modelo con input_shape: {input_shape}")
        
        # Limpiar sesión de Keras para evitar efectos secundarios
        tf.keras.backend.clear_session()
        
        # Crear el modelo usando Sequential con Input explícito
        model = Sequential([
            # Usar Input como primera capa
            Input(shape=input_shape),
            
            # Primera capa LSTM
            LSTM(128, return_sequences=True, 
                kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            BatchNormalization(),
            Dropout(0.15),

            # Segunda capa LSTM
            LSTM(64),
            BatchNormalization(),
            Dropout(0.15),

            # Capas densas
            Dense(32, activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.02)),
            Dense(1)
        ])

        # Configurar optimizador
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0003,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        # Compilar modelo
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Agregar método necesario para obtener la longitud de secuencia
        # (útil para las predicciones)
        def get_sequence_length():
            return input_shape[0]
        
        model.get_sequence_length = get_sequence_length
        
        self.logger.info(f"Modelo construido: {model.summary()}")
        return model