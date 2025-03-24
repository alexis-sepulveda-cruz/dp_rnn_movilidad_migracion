"""
bootstrap.py - Inicialización simplificada de la aplicación
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer

def load_environment():
    """Carga las variables de entorno desde el archivo .env."""
    env_paths = [
        Path('.env'),
        Path(__file__).parents[5] / '.env',
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"Variables de entorno cargadas desde: {env_path}")
            break
    
    # Asegurar que la variable BASE_PATH está disponible
    if 'BASE_PATH' not in os.environ:
        # Fallback a la ruta por defecto si no se encontró en .env
        os.environ['BASE_PATH'] = os.path.expanduser("~/Documents/Maestria_Ciencia_Datos/Tesis/dp_rnn_movilidad_migracion")
        print(f"BASE_PATH no encontrado en .env, usando valor por defecto: {os.environ['BASE_PATH']}")

def bootstrap_app():
    """Inicializa la aplicación y sus dependencias."""
    # Cargar variables de entorno
    load_environment()
    
    # Obtener rutas desde variables de entorno
    base_path = os.environ.get('BASE_PATH')
    
    # Crear contenedor de aplicación
    container = ApplicationContainer()
    
    # Configurar rutas desde variables de entorno de manera explícita
    container.config.paths.base.from_value(base_path)
    container.config.paths.conapo.from_value(os.environ.get('CONAPO_PATH', f"{base_path}/assets/dataset/conapo"))
    container.config.paths.inegi.from_value(os.environ.get('INEGI_PATH', f"{base_path}/assets/dataset/inegi"))
    container.config.paths.output.from_value(os.environ.get('OUTPUT_DIR', 'graficos'))

    # Configurar archivos
    container.config.files.conapo.from_env("CONAPO_FILE", default="5_Indicadores_demográficos_proyecciones.xlsx")
    container.config.files.inegi.from_env("INEGI_FILE", default="RESLOC_NACCSV2020.csv")

    # Configurar parámetros de datos
    container.config.data.years.start.from_env("START_YEAR", as_=int, default=1970)
    container.config.data.years.end.from_env("END_YEAR", as_=int, default=2019)
    container.config.data.sequence_length.from_env("SEQUENCE_LENGTH", as_=int, default=5)
    container.config.data.random_seed.from_env("RANDOM_SEED", as_=int, default=42)

    # Configurar el wiring
    containers_to_wire = [
        "dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.conapo_data_service",
        "dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.inegi_data_service",
    ]

    # Inicializar recursos después de la configuración
    container.init_resources()

    for module in containers_to_wire:
        container.wire(modules=[module])

    return container