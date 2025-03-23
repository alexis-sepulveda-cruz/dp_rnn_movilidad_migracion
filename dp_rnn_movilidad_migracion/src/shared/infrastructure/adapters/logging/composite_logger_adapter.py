from typing import Any, Dict, List, Tuple

from dp_rnn_movilidad_migracion.src.shared.domain.ports.logger_port import LoggerPort


class CompositeLoggerAdapter(LoggerPort):
    """
    Adaptador compuesto que implementa el puerto LoggerPort delegando
    las operaciones de logging a múltiples adaptadores subyacentes.
    
    Este adaptador implementa el patrón de diseño Composite, permitiendo
    tratar a un grupo de objetos de manera uniforme.
    
    Siguiendo el principio Open/Closed (O de SOLID), esta clase está
    abierta para extensión pero cerrada para modificación, ya que se pueden
    añadir nuevos loggers sin cambiar el código existente.
    """
    
    def __init__(self, *loggers: LoggerPort):
        """
        Inicializa el adaptador compuesto con una lista de loggers.
        
        Args:
            *loggers: Instancias de LoggerPort a las que se delegarán las operaciones
        """
        self.loggers: List[LoggerPort] = list(loggers)
        self.context: Dict[str, Any] = {}
    
    def add_logger(self, logger: LoggerPort) -> None:
        """
        Añade un nuevo logger al adaptador compuesto.
        
        Args:
            logger: Instancia de LoggerPort a añadir
        """
        self.loggers.append(logger)
    
    def remove_logger(self, logger: LoggerPort) -> bool:
        """
        Elimina un logger del adaptador compuesto.
        
        Args:
            logger: Instancia de LoggerPort a eliminar
            
        Returns:
            True si el logger fue eliminado, False si no se encontró
        """
        if logger in self.loggers:
            self.loggers.remove(logger)
            return True
        return False
    
    def with_context(self, **context_data) -> 'LoggerPort':
        """
        Crea una nueva instancia del logger compuesto con contexto adicional
        que será incluido en todos los mensajes y propagado a todos los loggers.
        
        Args:
            **context_data: Datos de contexto a incluir en los logs
            
        Returns:
            Una nueva instancia de logger compuesto con el contexto incorporado
        """
        # Creamos adaptadores con contexto para cada logger interno
        contextualized_loggers = [
            logger.with_context(**context_data) for logger in self.loggers
        ]
        
        # Creamos un nuevo adaptador compuesto con los loggers contextualizados
        new_composite = CompositeLoggerAdapter(*contextualized_loggers)
        
        # Guardamos también el contexto en el adaptador compuesto para referencia
        new_composite.context = {**self.context, **context_data}
        
        return new_composite
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel DEBUG en todos los loggers.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        self._log_to_all('debug', message, *args, **kwargs)
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel INFO en todos los loggers.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        self._log_to_all('info', message, *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel WARNING en todos los loggers.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        self._log_to_all('warning', message, *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel ERROR en todos los loggers.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        self._log_to_all('error', message, *args, **kwargs)
    
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel CRITICAL en todos los loggers.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        self._log_to_all('critical', message, *args, **kwargs)
    
    def _log_to_all(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Método interno para registrar un mensaje en todos los loggers.
        
        Args:
            level: Nivel de log ('debug', 'info', etc.)
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        # Capturamos excepciones individuales para evitar que un logger que falle
        # impida que los demás registren el mensaje
        exceptions: List[Tuple[LoggerPort, Exception]] = []
        
        for logger in self.loggers:
            try:
                # Usamos getattr para obtener dinámicamente el método correspondiente al nivel
                log_method = getattr(logger, level)
                log_method(message, *args, **kwargs)
            except Exception as e:
                # Capturamos la excepción para manejarla después
                exceptions.append((logger, e))
        
        # Si hubo excepciones, las manejamos (en este caso, solo las mostramos)
        if exceptions:
            for logger, exception in exceptions:
                # En un entorno real, podrías querer registrar esto en algún lado
                # o implementar un mecanismo de reintentos
                print(f"Error logging to {logger.__class__.__name__}: {exception}")