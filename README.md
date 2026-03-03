# Bird Song Analyzer

Android-приложение для определения птиц по голосу с использованием ML. Data-First стратегия: приложение как инструмент сбора структурированных акустических данных.

**Статус:** ML-pipeline реализован и протестирован (100% recall на benchmark 44 вида). UI-скелет всех экранов готов. Бэкенд не разрабатывается.

**Лицензия:** Proprietary (closed source). См. [ADR-008](./docs/adr/ADR-008-project-licensing.md).

---

## Архитектура

- **Паттерн:** MVVM + Clean Architecture, один Gradle-модуль `:app`
- **Язык:** Kotlin
- **UI:** Jetpack Compose + Material Design 3 (светлая/тёмная тема, dynamic colors)
- **DI:** Hilt
- **Навигация:** Jetpack Navigation Compose (type-safe routes)
- **БД:** Room (SQLite) — гео-данные
- **Preferences:** DataStore
- **ML:** TensorFlow Lite (BirdNET V2.4) + ONNX Runtime (BirdNET V3.0 EUNA)
- **Аудио:** AudioRecord API (VOICE_RECOGNITION source)
- **Async:** Kotlin Coroutines + Flow
- **Min SDK:** Android 11 (API 30), Target: Android 14 (API 34)

---

## ML-модели

| Модель | Runtime | Частота | Chunk | Классы | Лицензия |
|--------|---------|---------|-------|--------|----------|
| BirdNET V2.4 | TFLite FP16 | 48 kHz | 3 s | 6521 | CC BY-NC-SA 4.0 |
| BirdNET V3.0 EUNA | ONNX FP32 | 32 kHz | 5 s | 1225 (972 Aves) | CC BY-SA 4.0 |

Pipeline: AudioChunkProcessor (bandpass 80 Hz - 15 kHz, нормализация) -> Classifier (sigmoid + MetaProfile geo-filter) -> DetectionAggregator (sliding window, confirmation).

---

## Экраны

| Экран | Тип | Описание |
|-------|-----|----------|
| Home | Tab (Bottom Nav) | Хаб с плитками: Live Detection + File Analysis |
| History | Tab (Bottom Nav) | Сохранённые наблюдения |
| Settings | Tab (Bottom Nav) | Модель, локация, разрешения |
| Live Detection | Push screen | Dual-model анализ + лента обнаружений |
| File Analysis | Push screen | Concurrent анализ файла + timeline |
| Location Picker | Push screen | Выбор региона: континент -> страна -> регион |
| Detail | Push screen | Детали вида + плеер аудиофрагмента |

---

## Сборка

- **Среда:** Windows + Android Studio (WSL не поддерживает Android build-tools)
- **ML-тесты:** только на реальном устройстве (эмулятор не передаёт аудио)
- **V3.0 модель:** не в assets, sideload: `adb push birdnet_v30_euna.onnx /data/data/com.birdsong.analyzer/files/models/birdnet_v30/`

---

## Документация

### Архитектурные решения (ADR):

| ADR | Решение |
|-----|---------|
| [ADR-001](./docs/adr/ADR-001-platform-and-technology-stack.md) | Платформа и технологический стек |
| [ADR-002](./docs/adr/ADR-002-application-architecture.md) | Архитектура приложения |
| [ADR-003](./docs/adr/ADR-003-ml-model-selection.md) | ML-модели (BirdNET V2.4 + V3.0 EUNA) |
| [ADR-004](./docs/adr/ADR-004-audio-recording-format.md) | Формат аудиозаписи (OGG Opus) |
| [ADR-005](./docs/adr/ADR-005-data-storage-strategy.md) | Стратегия хранения данных (local-only) |
| [ADR-006](./docs/adr/ADR-006-mvp-scope.md) | Скоуп MVP |
| [ADR-007](./docs/adr/ADR-007-testing-strategy.md) | Стратегия тестирования |
| [ADR-008](./docs/adr/ADR-008-project-licensing.md) | Лицензия проекта |
| [ADR-009](./docs/adr/ADR-009-ux-behavior-and-user-flows.md) | UX-поведение и пользовательские потоки |
| [ADR-010](./docs/adr/ADR-010-live-detection-screen.md) | Экран Live Detection |

### Планирование:

| Документ | Описание |
|----------|----------|
| [REQUIREMENTS.md](./docs/planning/REQUIREMENTS.md) | Требования (pre-ADR, содержит устаревшие backend-разделы) |
| [STRATEGY.md](./docs/planning/STRATEGY.md) | Data-First стратегия (pre-ADR) |
| [COMPETITIVE_ANALYSIS.md](./docs/planning/COMPETITIVE_ANALYSIS.md) | Анализ конкурентов |
| [DATA_MODEL.md](./docs/planning/DATA_MODEL.md) | Серверная модель данных (pre-ADR, не актуальна для MVP) |
| [ROADMAP.md](./docs/planning/ROADMAP.md) | План разработки (pre-ADR) |

### Техническая документация:

| Документ | Описание |
|----------|----------|
| [AUDIO_ANALYSIS_PIPELINE.md](./docs/planning/AUDIO_ANALYSIS_PIPELINE.md) | Детальное описание ML-pipeline |
| [birdnet-v24-model-reference.md](./docs/birdnet-v24-model-reference.md) | Справка по модели BirdNET V2.4 |
| [SAMPLE_BENCHMARK.md](./docs/benchmarks/SAMPLE_BENCHMARK.md) | Benchmark на 16-мин файле (44 вида) |
| [STANDARD_BENCHMARK.md](./docs/benchmarks/STANDARD_BENCHMARK.md) | Mass benchmark на одновидовых файлах |

---

**Дата последнего обновления:** 2026-03-04
