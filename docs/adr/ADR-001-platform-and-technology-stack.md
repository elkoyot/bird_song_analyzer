# ADR-001: Platform and Technology Stack

Status: Accepted
Date: 2026-02-19
Deciders: @owner

## Context

Необходимо определить технологический стек для мобильного приложения Bird Song Analyzer. Приложение должно записывать аудио, запускать ML-модель на устройстве, хранить наблюдения локально и предоставлять удобный UI. Разработка ведётся одним человеком (solo-dev).

Рассматривались варианты:
- Kotlin + Jetpack Compose (нативный Android)
- Flutter / React Native (кроссплатформа)
- KMP + Compose Multiplatform (мультиплатформа)

## Decision

Нативный Android-стек:

- **Язык:** Kotlin
- **UI:** Jetpack Compose + Material Design 3
- **DI:** Hilt
- **Навигация:** Jetpack Navigation Compose (type-safe routes через Kotlin serialization)
- **Локальная БД:** Room (SQLite)
- **Preferences:** DataStore
- **Сеть (будущее):** Retrofit + OkHttp
- **Изображения:** Coil
- **Async:** Kotlin Coroutines + Flow
- **ML:** TensorFlow Lite
- **Аудио:** AudioRecord API
- **Min SDK:** Android 11 (API 30)
- **Target SDK:** Android 14 (API 34)

## Consequences

### Positive
- Полный доступ к AudioRecord API и фоновым сервисам (критично для аудио-ловушки)
- Нативная производительность TFLite без мостов
- Jetpack Compose — современный декларативный UI, меньше boilerplate чем XML
- Hilt — стандарт DI для Android, тесная интеграция с Jetpack
- Одна платформа = меньше сложности для solo-dev

### Negative
- Нет iOS (пока). При необходимости iOS — потребуется отдельная разработка или миграция на KMP
- Ограничение по аудитории (~70% рынка смартфонов)

### Neutral
- Material Design 3 с dynamic colors поддерживает светлую и тёмную тему из коробки
- Jetpack Navigation Compose — официальное решение Google, хорошо документировано

## Alternatives Considered

- **Flutter** — кроссплатформа, но TFLite интеграция через FFI, ограниченный доступ к AudioRecord, сложнее Foreground Services
- **React Native** — аналогичные проблемы с нативными API, плюс JS bridge overhead для ML
- **KMP** — перспективный, но экосистема менее зрелая, DI/навигация сложнее. Можно мигрировать позже если нужен iOS
