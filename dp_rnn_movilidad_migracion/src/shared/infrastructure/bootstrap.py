"""
bootstrap.py - Inicialización simplificada de la aplicación
"""
from pathlib import Path
from dotenv import load_dotenv
from dp_rnn_movilidad_migracion.src.shared.infrastructure.di.application_container import ApplicationContainer

def load_environment():
    """Carga las variables de entorno."""
    env_paths = [
        Path('.env'),
        Path(__file__).parents[5] / '.env',
    ]

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break

def bootstrap_app():
    """Inicializa la aplicación y sus dependencias."""
    # Cargar variables de entorno
    load_environment()

    # Crear contenedor de aplicación
    container = ApplicationContainer()

    # Cargar configuración desde variables de entorno
    container.config.paths.base.from_env("BASE_PATH")
    container.config.paths.conapo.from_env("CONAPO_PATH")
    container.config.paths.inegi.from_env("INEGI_PATH")

    container.config.files.conapo.from_env("CONAPO_FILE")
    container.config.files.inegi.from_env("INEGI_FILE")

    container.config.data.years.start.from_env("START_YEAR", as_=int)
    container.config.data.years.end.from_env("END_YEAR", as_=int)
    container.config.data.sequence_length.from_env("SEQUENCE_LENGTH", as_=int)
    container.config.data.random_seed.from_env("RANDOM_SEED", as_=int)

    # Configurar el wiring
    containers_to_wire = [
        "dp_rnn_movilidad_migracion.src.bounded_contexts.data.application.services.conapo_data_service"
    ]

    # Inicializar recursos
    container.init_resources()

    for module in containers_to_wire:
        container.wire(modules=[module])

    return container