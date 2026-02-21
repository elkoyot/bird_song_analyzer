# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bird Song Analyzer — Android-приложение для определения птиц по голосу с использованием ML. Data-First стратегия: приложение как инструмент сбора структурированных акустических данных.

**Статус:** ML-pipeline реализован и протестирован (100% recall на benchmark 44 вида). Бэкенд пока не разрабатывается.

**Лицензия:** Proprietary (closed source, private repo).

**Документация:** Планирование в `docs/planning/`. Архитектурные решения: `docs/adr/` (ADR-001..010).

## Architecture (ADR-001, ADR-002)

**Паттерн:** MVVM + Clean Architecture, один Gradle-модуль `:app`

**Стек:**
- Kotlin, Jetpack Compose, Material Design 3 (светлая + тёмная тема, dynamic colors)
- DI: Hilt
- Навигация: Jetpack Navigation Compose (type-safe routes)
- Локальная БД: Room (SQLite)
- Preferences: DataStore
- ML: TensorFlow Lite (BirdNET)
- Аудио: AudioRecord API
- Async: Kotlin Coroutines + Flow
- Изображения: Coil
- Min SDK: Android 11 (API 30), Target: Android 14 (API 34)

**Структура пакетов:**
```
com.birdsong.analyzer/
├── data/           # Room DAO, DataStore, repository impl
├── domain/         # Models, UseCases, repository interfaces
├── presentation/   # Compose UI (detection, detail, history, settings, theme, common)
├── ml/             # BirdNET classifier, AudioChunkProcessor, BandpassFilter, DetectionAggregator
├── service/        # Foreground Service для непрерывной записи
└── di/             # Hilt модули
```

## Data Storage (ADR-005)

Только локальное хранение, без бэкенда и синхронизации:
- **Room:** ObservationEntity, SpeciesEntity
- **Internal Storage:** аудиофайлы (OGG Opus)
- **DataStore:** настройки (язык, тема, параметры записи)

## ML Model (ADR-003)

BirdNET V2.4 FP16 через TFLite. Лицензия CC BY-NC-SA 4.0 (некоммерческая) — перед монетизацией нужно получить коммерческую лицензию или обучить свою модель. Pipeline: AudioChunkProcessor (пре-фильтрация, bandpass 80 Гц – 15 кГц, нормализация) → BirdNetV24Classifier (inference + sigmoid + meta-model) → DetectionAggregator (sliding window, подтверждение ≥2 chunk-ов, фильтрация не-птиц). Benchmark: 100% recall на 44 видах.

## Audio Format (ADR-004)

OGG Opus, 44.1/48 kHz, моно, 64-96 kbps. ~0.5-0.7 MB/мин. Максимальная длительность фрагмента: 60 сек.

## MVP Scope (ADR-006, ADR-009, ADR-010)

### Live Detection — главный экран

Непрерывный анализ аудиопотока с живой лентой обнаруженных видов:
- **Start** → непрерывная запись + chunking (3-5 сек) + BirdNET inference
- **Pause** → приостановить анализ (можно прослушать записи)
- **Stop** → завершить сессию, сохранить результаты
- **Reset** → очистить текущий список, продолжить анализ
- Confidence ≥ 80% → вид появляется в ленте + аудиофрагмент сохраняется
- Тап на вид → Detail Screen (название, confidence, GPS, плеер)
- Автопауза анализа при воспроизведении аудио
- Foreground Service для непрерывной записи

### Экраны

| Экран | Тип | Описание |
|-------|-----|----------|
| Live Detection | Tab (Bottom Nav) | Непрерывный анализ + лента обнаружений |
| History | Tab (Bottom Nav) | Сохранённые наблюдения, фильтры, удаление |
| Settings | Tab (Bottom Nav) | Язык (RU/EN), тема, разрешения |
| Detail | Push screen | Детали вида + плеер аудиофрагмента |

**GPS:** опционально (null если недоступен). **Хранилище:** автоудаление старых при 10 000 записей или < 100 MB.

**UI Preview:** Jetpack Compose @Preview (код прототипа = продакшн код).

НЕ в MVP: умная ловушка (VAD-триггер, интеллектуальные фильтры), бэкенд, авторизация, справочник птиц.

## Testing (ADR-007)

Полное тестирование с самого начала:
- JUnit 5 + MockK — unit-тесты (UseCase, ViewModel, Repository)
- Turbine — тестирование Flow
- Room in-memory — DAO queries
- Compose UI Testing — экраны и навигация
- CI/CD: пока локально, GitHub Actions позже

## Performance Requirements

- Распознавание chunk: < 3 секунд
- APK: < 150 MB (включая ML-модель)
- RAM: < 200 MB в активном режиме
