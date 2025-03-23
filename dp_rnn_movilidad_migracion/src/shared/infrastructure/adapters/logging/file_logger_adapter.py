import os
import logging
import threading  # Agregado, faltaba esta importación
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import inspect
from typing import Any, Dict, Optional

from dp_rnn_movilidad_migracion.src.shared.domain.ports.logger_port import LoggerPort


class FileLoggerAdapter(LoggerPort):
    """
    Adaptador de infraestructura que implementa el puerto LoggerPort
    para registrar logs en archivos con rotación diaria.
    
    Siguiendo el principio de Single Responsibility (S de SOLID), esta clase
    tiene la única responsabilidad de manejar el logging a archivos.
    
    Attributes:
        name (str): Nombre del logger, generalmente el nombre del módulo.
        log_dir (str): Directorio donde se guardarán los archivos de log.
        project_name (str): Nombre del proyecto para el archivo de log.
        context (Dict[str, Any]): Datos de contexto a incluir en cada log.
        logger (logging.Logger): Instancia del logger de Python.
    """
    
    def __init__(
        self, 
        name: str, 
        log_dir: str = "logs", 
        project_name: str = "dp_rnn_movilidad_migracion",
        log_to_console: bool = False,
        log_level: int = logging.INFO,
        rotation_type: str = "time",  # Opciones: "size", "time", "both"
        max_bytes: int = 10_485_760,  # 10MB por archivo
        backup_count: int = 10,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el adaptador de logging.
        
        Args:
            name (str): Nombre del logger, generalmente el nombre del módulo.
            log_dir (str): Directorio donde se guardarán los archivos de log.
            project_name (str): Nombre del proyecto para el archivo de log.
            log_to_console (bool): Si debe enviar logs también a la consola.
            log_level (int): Nivel mínimo de logs a registrar (logging.INFO, logging.DEBUG, etc).
            rotation_type (str): Tipo de rotación para los archivos. Opciones: "size", "time", "both".
            max_bytes (int): Tamaño máximo en bytes para los archivos rotados por tamaño.
            backup_count (int): Número máximo de archivos de respaldo a mantener.
            context (Optional[Dict[str, Any]]): Datos de contexto a incluir en cada log.
        
        Returns:
            None
            
        Example:
            >>> logger = FileLoggerAdapter(
            ...     name="mi_modulo",
            ...     log_dir="mi_app/logs",
            ...     project_name="mi_proyecto"
            ... )
            >>> logger.info("Aplicación iniciada")
        """
        self.name = name
        self.log_dir = log_dir
        self.project_name = project_name
        self.context = context or {}
        
        # Crear directorio de logs si no existe
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configurar el logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Evitar duplicación de handlers
        if not self.logger.handlers:
            # Definir el formato de log
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            log_file = os.path.join(self.log_dir, f"{self.project_name}.log")
            
            # Elegir el tipo de rotación según la configuración
            if rotation_type == "size":
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            elif rotation_type == "time":
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when="midnight",
                    interval=1,
                    backupCount=backup_count
                )
            elif rotation_type == "both":
                # Usar ambos tipos de rotación si específicamente se solicita
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when="midnight",
                    interval=1,
                    backupCount=backup_count
                )
                
                size_log_file = os.path.join(self.log_dir, f"{self.project_name}_size.log")
                size_handler = RotatingFileHandler(
                    size_log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                size_handler.setFormatter(formatter)
                self.logger.addHandler(size_handler)
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler condicional
            if log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

    def _get_caller_info(self) -> str:
        """
        Obtiene información sobre el llamador (clase y método).
        
        Returns:
            str: Cadena con información del llamador en formato "archivo:función:línea".
            
        Note:
            Este método es privado y se usa internamente para enriquecer los mensajes de log
            con información contextual sobre desde dónde se realizó la llamada.
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
        Crea una nueva instancia del logger con contexto adicional.
        
        Este método permite crear un logger derivado que incluye información
        de contexto adicional en cada mensaje de log.
        
        Args:
            **context_data: Pares clave-valor con información contextual a añadir.
            
        Returns:
            LoggerPort: Una nueva instancia de logger con el contexto combinado.
            
        Example:
            >>> logger = FileLoggerAdapter(name="base")
            >>> user_logger = logger.with_context(user_id="123", session="abc")
            >>> user_logger.info("Acción completada")  # Incluirá el user_id y session
        """
        new_context = {**self.context, **context_data}
        return FileLoggerAdapter(
            name=self.name,
            log_dir=self.log_dir,
            project_name=self.project_name,
            log_level=self.logger.level,
            log_to_console=any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers),
            context=new_context
        )

    def _format_with_context(self, message: str) -> str:
        """
        Formatea un mensaje de log incluyendo la información de contexto.
        
        Args:
            message (str): Mensaje original a formatear.
            
        Returns:
            str: Mensaje formateado con contexto añadido si existe.
            
        Note:
            Este método es privado y se usa internamente para añadir
            información contextual a los mensajes de log.
        """
        if not self.context:
            return message
            
        context_str = " ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{message} [{context_str}]"
    
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel DEBUG.
        
        Args:
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> logger.debug("Valor de la variable: %s", valor)
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.debug(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel INFO.
        
        Args:
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> logger.info("Proceso completado en %d segundos", tiempo)
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.info(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel WARNING.
        
        Args:
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> logger.warning("Recurso %s cerca del límite", recurso)
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.warning(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel ERROR.
        
        Args:
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> try:
            ...     # código que puede fallar
            ... except Exception as e:
            ...     logger.error("Error al procesar datos: %s", str(e))
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.error(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje con nivel CRITICAL.
        
        Args:
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> logger.critical("Sistema en estado crítico: %s", estado)
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.critical(f"[{caller_info}] {formatted_message}", *args, **kwargs)
    
    def async_log(self, level: int, message: str, *args: Any, **kwargs: Any) -> None:
        """
        Registra un mensaje de forma asíncrona.
        
        Este método permite registrar logs sin bloquear el hilo principal,
        útil para operaciones de logging intensivas o en situaciones
        donde el rendimiento es crítico.
        
        Args:
            level (int): Nivel de log (logging.DEBUG, logging.INFO, etc).
            message (str): Mensaje a registrar.
            *args: Argumentos adicionales para formatear el mensaje.
            **kwargs: Argumentos adicionales con nombre.
            
        Returns:
            None
            
        Example:
            >>> logger.async_log(logging.INFO, "Operación completada")
        """
        threading.Thread(
            target=self._log_message,
            args=(level, message, args, kwargs),
            daemon=True
        ).start()
        
    def _log_message(self, level: int, message: str, args: tuple, kwargs: dict) -> None:
        """
        Método interno para realizar el logging real.
        
        Este método es utilizado por async_log para realizar el
        registro efectivo del mensaje desde un hilo separado.
        
        Args:
            level (int): Nivel de log (logging.DEBUG, logging.INFO, etc).
            message (str): Mensaje a registrar.
            args (tuple): Argumentos para formatear el mensaje.
            kwargs (dict): Argumentos con nombre adicionales.
            
        Returns:
            None
        """
        caller_info = self._get_caller_info()
        formatted_message = self._format_with_context(message)
        self.logger.log(level, f"[{caller_info}] {formatted_message}", *args, **kwargs)