# DP-RNN Movilidad y Migración

[![Estado del Proyecto: En Desarrollo](https://img.shields.io/badge/Estado-En%20Desarrollo-yellow)](https://github.com/yourusername/dp_rnn_movilidad_migracion)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Arquitectura Hexagonal](https://img.shields.io/badge/Arquitectura-Hexagonal-lightgrey)](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))

## 📋 Descripción

Sistema de análisis y predicción de patrones de movilidad y migración en México utilizando redes neuronales recurrentes (RNN) con consideraciones de privacidad diferencial. Proporciona capacidades de modelado temporal avanzado para comprender dinámicas demográficas, incorporando análisis de incertidumbre mediante simulación Monte Carlo.

## 🎯 Objetivo

El proyecto busca proporcionar herramientas para el análisis de tendencias migratorias en México, permitiendo predecir el crecimiento natural de población en los estados mexicanos a partir de series temporales históricas (CONAPO) y datos socioeconómicos estáticos (INEGI).

## 🧠 Enfoque Técnico

- **Modelo Híbrido Temporal-Estático**: Combina series temporales demográficas con características socioeconómicas estáticas para mejorar precisión predictiva
- **Redes LSTM Multicapa**: Captura dependencias a largo plazo en patrones migratorios
- **Simulación Monte Carlo**: Cuantifica incertidumbre a través de distribuciones predictivas
- **Análisis de Fiabilidad**: Métricas detalladas para evaluar confiabilidad de predicciones (CV, scores de fiabilidad)

## 🏗️ Arquitectura

El proyecto implementa una **arquitectura hexagonal** (ports & adapters) siguiendo principios de **Domain-Driven Design (DDD)**:

```txt
bounded_contexts/
├── data/                      # Contexto acotado para gestión de datos
│   ├── domain/                # Entidades y puertos del dominio
│   ├── application/           # Servicios de aplicación (orquestación)
│   └── infrastructure/        # Repositorios y adaptadores externos
└── migration_prediction/      # Contexto acotado para predicción
    ├── domain/                # Entidades y puertos del dominio
    ├── application/           # Servicios de aplicación
    └── infrastructure/        # Adaptadores (modelos RNN, visualizadores)
```

## 🔧 Tecnologías

- **TensorFlow/Keras**: Implementación del modelo RNN
- **NumPy/Pandas**: Manipulación y procesamiento de datos
- **Scikit-learn**: Normalización y evaluación de modelos
- **Matplotlib/Seaborn**: Visualización avanzada de resultados
- **Dependency Injector**: Inyección de dependencias para aplicación modular

## 📊 Características

- **Predicción Multi-paso**: Pronósticos de 5 años de evolución demográfica
- **Análisis de Incertidumbre**: Intervalos de confianza y coeficientes de variación
- **Visualizaciones Comparativas**: Comparación entre estados por regiones
- **Normalización Automatizada**: Preprocesamiento adaptativo según magnitudes de datos
- **Métricas de Fiabilidad**: Scores que facilitan interpretación de confiabilidad

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/yourusername/dp_rnn_movilidad_migracion.git
cd dp_rnn_movilidad_migracion

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con las rutas correspondientes
```

## ⚙️ Configuración

El sistema utiliza un archivo `.env` para configuración:

```bash
# Rutas base
BASE_PATH=/ruta/completa/al/proyecto
CONAPO_PATH=${BASE_PATH}/assets/dataset/conapo
INEGI_PATH=${BASE_PATH}/assets/dataset/inegi
OUTPUT_DIR=graficos

# Configuración de datos
START_YEAR=1970
END_YEAR=2019
SEQUENCE_LENGTH=5
RANDOM_SEED=42

# Archivos de datos
CONAPO_FILE=5_Indicadores_demográficos_proyecciones.xlsx
INEGI_FILE=RESLOC_NACCSV2020.csv
```

## 📈 Ejemplo de Uso

```python
from dp_rnn_movilidad_migracion.src.shared.infrastructure.bootstrap import bootstrap_app
from dp_rnn_movilidad_migracion.src.shared.domain.value_objects.mexican_states import MexicanState

# Inicializar la aplicación
container = bootstrap_app()

# Obtener servicios
conapo_service = container.conapo_data_service()
inegi_service = container.inegi_data_service()
prediction_service = container.migration_prediction_service()

# Cargar datos
conapo_data = conapo_service.get_processed_data()
inegi_data = inegi_service.get_processed_data()

# Entrenar modelo
history = prediction_service.train_model(
    temporal_data=conapo_data,
    static_data=inegi_data,
    visualize_history=True
)

# Predecir para un estado específico
from dp_rnn_movilidad_migracion.src.bounded_contexts.migration_prediction.application.dto.prediction_request_dto import PredictionRequestDTO

prediction = prediction_service.predict_migration(
    request=PredictionRequestDTO(
        state="Jalisco",
        future_years=5,
        monte_carlo_samples=500,
        confidence_level=0.95,
        visualize=True
    ),
    temporal_data=conapo_data,
    static_data=inegi_data
)
```

📁 Estructura del Proyecto

```txt
dp_rnn_movilidad_migracion/
├── assets/                    # Datos de entrada (no incluidos en repo)
│   └── dataset/
│       ├── conapo/            # Datos históricos demográficos
│       └── inegi/             # Datos del censo
├── graficos/                  # Directorio de salida para visualizaciones
├── logs/                      # Logs de la aplicación
├── resultados/                # Resultados de predicciones
├── dp_rnn_movilidad_migracion/
│   └── src/
│       ├── bounded_contexts/  # Contextos acotados (DDD)
│       └── shared/            # Infraestructura compartida
├── tests/                     # Pruebas unitarias e integración
├── .env                       # Variables de entorno
├── main.py                    # Punto de entrada principal
└── requirements.txt           # Dependencias
```

## 🔍 Detalles Técnicos
