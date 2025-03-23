import logging
import inspect
import os
from typing import Any, Dict, Optional

from dp_rnn_movilidad_migracion.src.shared.domain.ports.logger_port import LoggerPort


class ConsoleLoggerAdapter(LoggerPort):
    """
    Adaptador de infraestructura que implementa el puerto LoggerPort
    para registrar logs en la consola.
    
    Siguiendo el principio de Single Responsibility (S de SOLID), esta clase
    tiene la única responsabilidad de manejar el logging a consola.
    """
    
    def __init__(
        self, 
        name: str, 
        log_level: int = logging.INFO,
        colored_output: bool = True,
        show_timestamps: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el adaptador de logging para consola.
        
        Args:
            name: Nombre del logger, generalmente el nombre del módulo
            log_level: Nivel de logging (DEBUG, INFO, etc.)
            colored_output: Si True, usa colores ANSI en la salida
            show_timestamps: Si True, muestra timestamps en cada mensaje
            context: Datos de contexto a incluir en cada log
        """
        self.name = name
        self.colored_output = colored_output
        self.show_timestamps = show_timestamps
        self.context = context or {}
        
        # Configurar el logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Evitar duplicación de handlers
        if not self.logger.handlers:
            # Crear handler para consola
            console_handler = logging.StreamHandler()
            
            # Formato configurable
            if show_timestamps:
                fmt = "%(asctime)s [%(levelname)s] %(message)s"
                datefmt = "%Y-%m-%d %H:%M:%S"
            else:
                fmt = "[%(levelname)s] %(message)s"
                datefmt = None
                
            formatter = logging.Formatter(fmt, datefmt=datefmt)
            console_handler.setFormatter(formatter)
            
            if colored_output:
                self._apply_colored_formatting(console_handler)
            
            # Agregar handler al logger
            self.logger.addHandler(console_handler)
    
    def _apply_colored_formatting(self, handler):
        """
        Aplica formato de colores al handler de consola.
        
        Args:
            handler: El handler de consola al que aplicar colores
        """
        # No implementar la lógica de colores directamente para evitar dependencias
        # externas. Esto se puede implementar con bibliotecas como colorama o
        # con códigos ANSI directamente si se prefiere.
        try:
            # Intentar importar colorlog si está disponible
            import colorlog
            
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            handler.setFormatter(color_formatter)
        except ImportError:
            # Si colorlog no está disponible, usar códigos ANSI básicos
            class ColorFormatter(logging.Formatter):
                COLORS = {
                    'DEBUG': '\033[36m',     # Cyan
                    'INFO': '\033[32m',      # Green
                    'WARNING': '\033[33m',   # Yellow
                    'ERROR': '\033[31m',     # Red
                    'CRITICAL': '\033[41m',  # Red background
                    'RESET': '\033[0m'       # Reset
                }
                
                def format(self, record):
                    levelname = record.levelname
                    color = self.COLORS.get(levelname, self.COLORS['RESET'])
                    record.levelname = f"{color}{levelname}{self.COLORS['RESET']}"
                    return super().format(record)
            
            if self.show_timestamps:
                fmt = "%(asctime)s [%(levelname)s] %(message)s"
                datefmt = "%Y-%m-%d %H:%M:%S"
            else:
                fmt = "[%(levelname)s] %(message)s"
                datefmt = None
                
            color_formatter = ColorFormatter(fmt, datefmt=datefmt)
            handler.setFormatter(color_formatter)
    
    def _get_caller_info(self) -> str:
        """
        Obtiene información sobre el llamador (clase y método).
        
        Returns:
            Cadena con información del llamador
        """
        stack = inspect.stack()
        # El índice 2 corresponde al llamador del método de logging
        # (0 es este método, 1 es el método de logging como debug/info)
        caller = stack[4]
        filename = os.path.basename(caller.filename)
        lineno = caller.lineno
        function = caller.function
        return f"{filename}:{function}:{lineno}"
    
    def with_context(self, **context_data) -> 'LoggerPort':
        """
        Crea una nueva instancia del logger con contexto adicional
        que será incluido en todos los mensajes.
        
        Args:
            **context_data: Datos de contexto a incluir en los logs
            
        Returns:
            Una nueva instancia de logger con el contexto incorporado
        """
        new_context = {**self.context, **context_data}
        return ConsoleLoggerAdapter(
            name=self.name,
            log_level=self.logger.level,
            colored_output=self.colored_output,
            show_timestamps=self.show_timestamps,
            context=new_context
        )
    
    def _format_with_context(self, message: str) -> str:
        """
        Formatea el mensaje con el contexto actual.
        
        Args:
            message: Mensaje original
            
        Returns:
            Mensaje formateado con contexto
        """
        if not self.context:
            return message
            
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{message} [{context_str}]"
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel DEBUG.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.debug(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel INFO.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.info(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel WARNING.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.warning(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel ERROR.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.error(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel CRITICAL.
        
        Args:
            message: Mensaje a registrar
            *args: Argumentos adicionales para formatear el mensaje
            **kwargs: Argumentos adicionales con nombre
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.critical(f"[{caller_info}] {formatted_message}", *args, **kwargs)