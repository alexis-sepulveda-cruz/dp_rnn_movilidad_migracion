from typing import Dict, Optional, Type
import os

from dp_rnn_movilidad_migracion.src.shared.domain.ports.logger_port import LoggerPort
from dp_rnn_movilidad_migracion.src.shared.infrastructure.adapters.logging.file_logger_adapter import FileLoggerAdapter
from dp_rnn_movilidad_migracion.src.shared.infrastructure.adapters.logging.console_logger_adapter import ConsoleLoggerAdapter
from dp_rnn_movilidad_migracion.src.shared.infrastructure.adapters.logging.composite_logger_adapter import CompositeLoggerAdapter

class LoggerFactory:
    """
    Fábrica para crear y gestionar instancias de loggers en la aplicación.
    
    Esta clase implementa el patrón Factory para la creación de loggers,
    permitiendo configurar y reutilizar instancias entre diferentes
    componentes de la aplicación.
    
    Attributes:
        _instances (Dict[str, LoggerPort]): Caché de instancias de loggers.
        _project_name (str): Nombre del proyecto utilizado en los archivos de log.
        _log_dir (str): Directorio donde se almacenarán los archivos de log.
        _default_adapter_class (Type[LoggerPort]): Clase adaptadora por defecto.
        _environment (str): Entorno de ejecución (development, testing, production).
    """
    
    _instances: Dict[str, LoggerPort] = {}
    _project_name: str = "dp_rnn_movilidad_migracion"
    _log_dir: str = "logs"
    _default_adapter_class: Type[LoggerPort] = FileLoggerAdapter
    _environment: str = os.getenv("APP_ENV", "development")
    
    @classmethod
    def configure(
        cls, 
        project_name: str, 
        log_dir: Optional[str] = None,
        adapter_class: Optional[Type[LoggerPort]] = None,
        environment: Optional[str] = None
    ) -> None:
        """
        Configura los parámetros globales de la fábrica de loggers.
        
        Esta configuración afecta a todas las instancias futuras de loggers
        que se crearán a través de esta fábrica.
        
        Args:
            project_name (str): Nombre del proyecto para los archivos de log.
            log_dir (Optional[str]): Directorio donde se guardarán los logs.
                Si es None, se mantiene el valor actual.
            adapter_class (Optional[Type[LoggerPort]]): Clase adaptadora por defecto.
                Si es None, se mantiene el valor actual.
            environment (Optional[str]): Entorno de ejecución (development, testing, production).
                Si es None, se mantiene el valor actual.
        
        Returns:
            None
            
        Example:
            >>> LoggerFactory.configure(
            ...     project_name="mi_aplicacion",
            ...     log_dir="/var/log/mi_app",
            ...     environment="production"
            ... )
        """
        cls._project_name = project_name
        if log_dir:
            cls._log_dir = log_dir
        if adapter_class:
            cls._default_adapter_class = adapter_class
        if environment:
            cls._environment = environment
    
    @classmethod
    def get_logger(
        cls, 
        name: str, 
        adapter_class: Optional[Type[LoggerPort]] = None
    ) -> LoggerPort:
        """
        Obtiene o crea una instancia de logger con la configuración especificada.
        
        Este método implementa un patrón Singleton por cada combinación única
        de nombre y clase adaptadora, garantizando la reutilización de instancias.
        
        Args:
            name (str): Nombre identificativo del logger, generalmente el nombre del módulo.
            adapter_class (Optional[Type[LoggerPort]]): Clase adaptadora a utilizar.
                Si es None, se utiliza la clase por defecto configurada en la fábrica.
        
        Returns:
            LoggerPort: Instancia del logger configurado.
            
        Example:
            >>> logger = LoggerFactory.get_logger("auth.service")
            >>> logger.info("Usuario autenticado")
        """
        logger_key = f"{name}:{adapter_class.__name__ if adapter_class else cls._default_adapter_class.__name__}"
        
        if logger_key not in cls._instances:
            adapter = adapter_class or cls._default_adapter_class
            
            # Configuración basada en entorno
            log_level = cls._get_log_level_for_environment()
            
            cls._instances[logger_key] = adapter(
                name=name,
                log_dir=cls._log_dir,
                project_name=cls._project_name,
                log_level=log_level
            )
        return cls._instances[logger_key]
    
    @classmethod
    def _get_log_level_for_environment(cls) -> int:
        """
        Determina el nivel de log apropiado basado en el entorno de ejecución.
        
        Returns:
            int: Constante de nivel de logging (logging.DEBUG, logging.INFO, etc.)
        
        Note:
            Este método es privado y se usa internamente para ajustar
            automáticamente los niveles de log según el entorno.
            - development: DEBUG
            - testing: INFO
            - production: WARNING
        """
        import logging
        env_log_levels = {
            "development": logging.DEBUG,
            "testing": logging.INFO,
            "production": logging.WARNING
        }
        return env_log_levels.get(cls._environment, logging.INFO)
        
    @classmethod
    def clear_instances(cls) -> None:
        """
        Limpia todas las instancias de loggers almacenadas.
        
        Este método es útil principalmente para testing y para
        situaciones donde se necesita reconfigurrar completamente
        el sistema de logging.
        
        Returns:
            None
            
        Example:
            >>> LoggerFactory.clear_instances()
            >>> # Todas las instancias posteriores serán nuevas
        """
        cls._instances.clear()

    @classmethod
    def get_console_logger(cls, module_name: str) -> LoggerPort:
        """
        Obtiene un logger que solo escribe en la consola.
        
        Args:
            module_name (str): Nombre del módulo para identificar el logger.
        
        Returns:
            LoggerPort: Instancia de logger configurado para escribir en consola.
            
        Example:
            >>> console_logger = LoggerFactory.get_console_logger("ui.component")
            >>> console_logger.info("Componente inicializado")
        """
        name = f"console.{module_name}"
        return ConsoleLoggerAdapter(name=name, colored_output=True)

    @classmethod
    def get_file_logger(cls, module_name: str) -> LoggerPort:
        """
        Obtiene un logger que solo escribe en archivo.
        
        Args:
            module_name (str): Nombre del módulo para identificar el logger.
        
        Returns:
            LoggerPort: Instancia de logger configurado para escribir en archivo.
            
        Example:
            >>> file_logger = LoggerFactory.get_file_logger("data.processor")
            >>> file_logger.info("Procesamiento iniciado")
        """
        name = f"file.{module_name}"
        return FileLoggerAdapter(
            name=name,
            log_dir=cls._log_dir,
            project_name=cls._project_name,
            log_to_console=False
        )

    @classmethod
    def get_composite_logger(cls, module_name: str) -> LoggerPort:
        """
        Obtiene un logger compuesto que escribe tanto en archivo como en consola.
        
        Este método crea un logger que utiliza múltiples adaptadores, permitiendo
        el registro simultáneo en diferentes destinos.
        
        Args:
            module_name (str): Nombre del módulo para identificar el logger.
        
        Returns:
            LoggerPort: Instancia de composite logger que escribe en varios destinos.
            
        Example:
            >>> logger = LoggerFactory.get_composite_logger("api.controller")
            >>> logger.info("Solicitud procesada")  # Se registra en consola y archivo
        """
        file_logger = cls.get_file_logger(f"{module_name}.inner")
        console_logger = cls.get_console_logger(f"{module_name}.inner")
        return CompositeLoggerAdapter(file_logger, console_logger)