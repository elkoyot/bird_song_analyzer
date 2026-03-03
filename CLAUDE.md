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
- ML: TensorFlow Lite (BirdNET V2.4) + ONNX Runtime (BirdNET V3.0 EUNA)
- Аудио: AudioRecord API
- Async: Kotlin Coroutines + Flow
- Изображения: Coil
- Min SDK: Android 11 (API 30), Target: Android 14 (API 34), Compile SDK: 35

**Структура пакетов:**
```
com.birdsong.analyzer/
├── data/
│   ├── local/          # Room DB (GeoDatabase, GeoDao, GeoSeedLoader)
│   ├── model/          # Room entities (GeoEntity, MlModelEntity, GeoModelEntity)
│   ├── repository/     # GeoRepository
│   └── PreferencesRepository.kt  # DataStore (countryCode, regionCode, activeModel)
├── presentation/
│   ├── navigation/     # Screen routes, NavGraph (BirdSongNavHost)
│   ├── detection/      # Home, LiveDetection, DualDetection, FileAnalysis screens + VMs
│   ├── detail/         # DetailScreen
│   ├── history/        # HistoryScreen
│   ├── location/       # LocationPickerScreen + VM (continent → country → region)
│   ├── settings/       # SettingsScreen + VM
│   └── theme/          # Color, Theme, Typography
├── ml/                 # BirdNET V2.4/V3.0 classifiers, AudioChunkProcessor, pipeline, aggregator
├── service/            # AudioRecorder (@Singleton, не Service)
└── di/                 # AppModule, MlModule (Hilt)
```

## Data Storage (ADR-005)

Только локальное хранение, без бэкенда и синхронизации:
- **Room (GeoDatabase):** GeoEntity (континенты/страны/регионы), MlModelEntity (ML-модели), GeoModelEntity (связь гео↔модель). ObservationEntity/SpeciesEntity — запланированы, но ещё не реализованы
- **Internal Storage:** аудиофайлы (OGG Opus) — запланировано
- **DataStore:** настройки (countryCode, regionCode, activeModel)

## ML Models (ADR-003)

Dual-classifier система:

**BirdNET V2.4** — TFLite FP16, 48 kHz / 3 s, 6521 классов. Лицензия CC BY-NC-SA 4.0 (некоммерческая).
**BirdNET V3.0 EUNA** — ONNX Runtime, 32 kHz / 5 s, 1225 классов (972 Aves). Лицензия CC BY-SA 4.0 (коммерция ОК). Модель не в assets — sideload через `adb push` в `filesDir/models/birdnet_v30/`.

Pipeline: AudioChunkProcessor (FULL: silence/clipping/spectral check, bandpass 80 Гц – 15 кГц, нормализация) → Classifier (logits → sigmoid → MetaProfile geo-filter) → DetectionAggregator (sliding window=8, avg-top-3, подтверждение ≥2 chunk-ов, фильтрация не-птиц). Benchmark V2.4: 100% recall на 44 видах.

## Audio Format (ADR-004)

OGG Opus, 44.1/48 kHz, моно, 64-96 kbps. ~0.5-0.7 MB/мин. Максимальная длительность фрагмента: 60 сек.

## MVP Scope (ADR-006, ADR-009, ADR-010)

### Home Hub — стартовый экран

Плиточная навигация по режимам работы:
- **Live Detection** — крупная плитка, основной use case
- **File Analysis** — компактная плитка, анализ аудиофайлов

### Live Detection

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
| Home | Tab (Bottom Nav) | Хаб с плитками режимов (Live Detection, File Analysis) |
| History | Tab (Bottom Nav) | Сохранённые наблюдения, фильтры, удаление |
| Settings | Tab (Bottom Nav) | Модель, локация, разрешения |
| Live Detection | Push screen | Dual-model анализ (V2.4 + V3.0) + лента обнаружений |
| File Analysis | Push screen | Concurrent анализ файла двумя моделями + timeline |
| Location Picker | Push screen | Выбор региона: континент → страна → регион |
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
