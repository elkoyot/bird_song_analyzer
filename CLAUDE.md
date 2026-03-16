# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bird Song Analyzer (Avalga) — Android-приложение для определения птиц по голосу с использованием ML. Data-First стратегия: приложение как инструмент сбора структурированных акустических данных.

**Статус:** ML-pipeline реализован и протестирован (100% recall на benchmark 44 вида). Справочник видов реализован (11,500 видов). Бэкенд пока не разрабатывается.

**Лицензия:** Proprietary (closed source, private repo).

**Документация:**
- Техническая: `docs/architecture.md` — полная документация (ML, БД, UI, pipeline)
- Планирование: `docs/planning/`
- ADR: `docs/adr/` (ADR-001..010)

## Architecture (ADR-001, ADR-002)

**Паттерн:** MVVM + Clean Architecture, один Gradle-модуль `:app`

**Стек:**
- Kotlin, Jetpack Compose, Material Design 3 (светлая + тёмная тема, dynamic colors)
- DI: Hilt
- Навигация: Jetpack Navigation Compose (type-safe routes)
- Локальная БД: Room (две БД: ReferenceDatabase + UserDatabase)
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
│   ├── local/          # ReferenceDatabase (read-only), UserDatabase, DAOs
│   ├── model/          # Room entities (Geo, Species, Taxon, FileAnalysis, Translation)
│   ├── repository/     # GeoRepository, SpeciesRepository, FileAnalysisRepository
│   └── PreferencesRepository.kt  # DataStore (countryCode, regionCode, activeModel)
├── presentation/
│   ├── navigation/     # Screen routes, NavGraph (BirdSongNavHost)
│   ├── detection/      # Home, DualDetection, FileAnalysis screens + VMs
│   ├── detail/         # DetailScreen
│   ├── history/        # HistoryScreen + VM
│   ├── location/       # LocationPickerScreen + VM
│   ├── settings/       # SettingsScreen + VM
│   ├── splash/         # SplashScreen, PermissionScreen
│   └── theme/          # Color (HubColors), Theme, Typography
├── ml/                 # Classifiers, AudioChunkProcessor, Pipeline, ModelMap, MetaProfile
├── service/            # AudioRecorder, AudioPlaybackManager
└── di/                 # AppModule, MlModule (Hilt)
```

## Data Storage (ADR-005)

Две базы данных + DataStore:

**ReferenceDatabase** (`reference.db`) — read-only, `createFromAsset`:
- `geo_entity` — континенты/страны/регионы + bbox
- `ml_model`, `geo_model` — ML-модели и связь с регионами
- `taxon_order` (~56), `taxon_family` (~259) — таксономическая иерархия
- `species` (~11,500) — виды (scientific_name PK, familyId FK, genus, iucnStatus)
- `species_name` (~22,000) — локализованные названия (en + ru)
- `taxonomy_synonym` (~30,000) — устаревшие лат. имена → актуальные
- `species_country` (~130,000) — вид ↔ страна
- `translation` (~718) — переводы отрядов, семейств, континентов, регионов

**UserDatabase** (`user.db`) — read-write:
- `file_analysis` — сохранённые анализы файлов
- `file_detection` — детекции в анализах

**DataStore:** countryCode, regionCode, activeModel.

**Генерация reference.db:** `dictionary/build_reference_db.py` (Python, запускается один раз).

## ML Models (ADR-003)

Dual-classifier система:

**BirdNET V2.4** — TFLite FP16, 48 kHz / 3 s, 6521 классов. Лицензия CC BY-NC-SA 4.0.
**BirdNET V3.0 EUNA** — ONNX Runtime, 32 kHz / 5 s, 1225 классов (972 Aves + 253 не-птицы). CC BY-SA 4.0. Sideload в `filesDir/models/birdnet_v30/`.

**Pipeline:**
AudioChunkProcessor (silence/clipping/spectral → bandpass 80 Гц–15 кГц)
→ Classifier (inference → sigmoid → MetaProfile geo-filter)
→ ModelMap (labelIndex → resolved scientificName + taxonClass)
→ DetectionAggregator (sliding window=8, avg-top-3, confirmation ≥2)

**ModelMap** (`model_map.csv` per model): маппинг labelIndex → actulName + taxonClass. Резолвит устаревшие имена, фильтрует шум. `enabledClasses` параметр позволяет включать не-птичьи таксоны.

## Audio Format (ADR-004)

OGG Opus, 44.1/48 kHz, моно, 64-96 kbps. ~0.5-0.7 MB/мин. Max: 60 сек.

## MVP Scope (ADR-006, ADR-009, ADR-010)

### Навигация

SplashRoute → PermissionRoute → MainGraph (3 таба: Слушать/Инфо/Профиль).
Push-экраны: DualDetection, FileAnalysis, History, Detail, LocationPicker, Settings.

### Экраны

| Экран | Тип | Описание |
|-------|-----|----------|
| Home | Tab (Bottom Nav) | Хаб с плитками режимов (Live, File, Ловушка*, Экспедиция*) |
| Info | Tab (Bottom Nav) | Информация |
| Profile | Tab (Bottom Nav) | Профиль |
| DualDetection | Push | Dual V2.4+V3.0 live анализ + лента обнаружений |
| FileAnalysis | Push | Parallel анализ файла + spectrogram + timeline |
| History | Push | Сохранённые анализы, swipe-to-delete |
| Detail | Push | Карточка вида + плеер |
| LocationPicker | Push | Континент → страна → регион |
| Settings | Push | Модель, локация |

*помечены "скоро"

### HubColors (тёмная тема)

Bg=#050C18, BgEl2=#162842, Accent=#E8A020, Green=#3DBA7E, Blue=#4BA3C7, Purple=#9B7FE8, Red=#E05050, Yellow=#E8C020.

## Testing (ADR-007)

- JUnit 5 + MockK — unit-тесты
- Turbine — Flow
- Room in-memory — DAO
- Compose UI Testing — экраны
- CI/CD: пока локально

## Performance Requirements

- UI: 16 ms frame budget, тяжёлые операции на IO/Default
- Chunk inference: < 3 сек
- APK: < 150 MB
- RAM: < 200 MB
