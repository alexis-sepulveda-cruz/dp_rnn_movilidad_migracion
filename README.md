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

