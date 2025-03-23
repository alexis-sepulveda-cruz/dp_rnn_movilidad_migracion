from abc import ABC, abstractmethod
from typing import Any


class LoggerPort(ABC):
    """
    Puerto (interfaz) para el sistema de logging.
    Define los contratos que cualquier implementación de logging debe cumplir.
    """

    @abstractmethod
    def with_context(self, **context_data) -> 'LoggerPort':
        """
        Crea una nueva instancia del logger con contexto adicional
        que será incluido en todos los mensajes.
        
        Args:
            **context_data: Datos de contexto a incluir en los logs
            
        Returns:
            Una nueva instancia de logger con el contexto incorporado
        """
        pass
    
    @abstractmethod
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel DEBUG.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        pass
    
    @abstractmethod
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel INFO.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        pass
    
    @abstractmethod
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel WARNING.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        pass
    
    @abstractmethod
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel ERROR.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        pass
    
    @abstractmethod
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel CRITICAL.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        pass
