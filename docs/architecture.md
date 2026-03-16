# Bird Song Analyzer — Техническая документация

## 1. Обзор проекта

**Bird Song Analyzer (Avalga)** — Android-приложение для определения птиц по голосу с использованием ML. Data-First стратегия: приложение как инструмент сбора структурированных акустических данных.

**Стек:** Kotlin, Jetpack Compose, Material Design 3, Hilt, Room, TensorFlow Lite, ONNX Runtime, Coroutines + Flow.

**Min SDK:** Android 11 (API 30), Target: Android 14 (API 34).

**Архитектура:** MVVM + Clean Architecture, один Gradle-модуль `:app`.

```
com.birdsong.analyzer/
├── data/           # Хранение: Room, DataStore, репозитории
├── presentation/   # UI: экраны (Compose), ViewModels, тема
├── ml/             # ML: классификаторы, pipeline, preprocessing
├── service/        # Аудио: запись, воспроизведение
└── di/             # Hilt DI модули
```

---

## 2. ML-слой: распознавание птиц

### 2.1. Архитектура dual-classifier

Приложение использует две модели BirdNET параллельно:

| Параметр | BirdNET V2.4 | BirdNET V3.0 EUNA |
|----------|-------------|-------------------|
| Runtime | TensorFlow Lite (FP16) | ONNX Runtime (FP32) |
| Sample rate | 48 kHz | 32 kHz |
| Chunk duration | 3 сек (144,000 samples) | 5 сек (160,000 samples) |
| Классы | 6,521 (все виды мира) | 1,225 (972 Aves + 253 не-птицы) |
| Таксоны | Только птицы | Aves, Insecta, Mammalia, Amphibia, Squamata |
| Geo-фильтр | Встроенная meta-model | Через MetaProfile V2.4 |
| Лицензия | CC BY-NC-SA 4.0 | CC BY-SA 4.0 |
| Расположение | `assets/birdnet/v24/` | `filesDir/models/birdnet_v30/` (sideload) |

### 2.2. Загрузка моделей

**V2.4** загружается при старте через Hilt DI (`MlModule`):
1. `audio-model-fp16.tflite` → `MappedByteBuffer` (memory-mapped, без копирования)
2. `meta-model.tflite` → `MappedByteBuffer`
3. `labels/ru.txt` → `List<Pair<String, String>>` (scientific + common name)
4. `model_map.csv` → `ModelMap` (labelIndex → scientificName + taxonClass)
5. `BirdNetV24Classifier` создаётся с этими зависимостями

**V3.0** загружается лениво через `ClassifierFactory` при первом обращении:
1. Проверяет наличие `filesDir/models/birdnet_v30/birdnet_v30_euna.onnx`
2. Загружает `labels.csv` (semicolon-delimited) и `model_map.csv`
3. Строит маппинг V3.0 → V2.4 label indices для geo-фильтрации
4. Создаёт `BirdNetV30Classifier` с ONNX сессией

**SplashScreen** прогревает V2.4 в фоне, показывая прогресс-бар.

### 2.3. Pipeline распознавания

```
Аудио поток (48 kHz, mono)
    │
    ▼
AudioChunkProcessor (preprocessing)
    │ Silence check (RMS < 0.001 → skip)
    │ Clipping check (peak > 0.99 && RMS > 0.3 → skip)
    │ Spectral check (Goertzel, 5 полос → reject if >95% energy at 100 Hz or 12 kHz)
    │ Bandpass filter (Butterworth 80 Hz – 15 kHz)
    │ Post-filter silence check
    │
    ▼
BirdClassifier.classify(samples, location?)
    │ V2.4: TFLite inference → logits → sigmoid → scores[6521]
    │ V3.0: ONNX inference → scores[1225] (already sigmoid)
    │
    ▼
Geo-фильтрация
    │ V2.4: meta-model(lat, lon, week) → blended: score × (α + (1-α) × metaScore)
    │ V3.0: MetaProfile V2.4 → penalize if maxScore < 0.03 (×0.1)
    │
    ▼
ModelMap resolution
    │ labelIndex → scientificName (resolved, e.g. "Parus cinereus" → "Parus major")
    │ labelIndex → taxonClass ("Aves", "Mammalia", ...)
    │ Empty scientificName → noise/unknown → skip
    │
    ▼
buildDetections
    │ Filter: confidence ≥ threshold (0.05)
    │ Filter: taxonClass in enabledClasses (default: Aves only)
    │ Sort by confidence descending
    │ Take top-K (10)
    │
    ▼
List<BirdDetection>
    │ scientificName, commonName, confidence, labelIndex, taxonClass
    │
    ▼
DetectionAggregator
    │ Live: sliding window = 8 chunks, confirmation ≥ 2, avg-top-3
    │ File: unlimited window, confirmation ≥ 1, max confidence
    │
    ▼
List<AggregatedDetection> → UI
```

### 2.4. Preprocessing: AudioChunkProcessor

Три режима:
- **FULL** — полный пайплайн (silence → clipping → spectral → bandpass → post-check)
- **LIGHT** — без spectral check (для моделей с встроенным STFT)
- **PASSTHROUGH** — только clipping + нормализация (текущий default)

Bandpass фильтр: 2nd-order Butterworth, cascaded biquad (high-pass 80 Hz + low-pass 15 kHz). Коэффициенты по формулам Audio EQ Cookbook.

### 2.5. Geo-фильтрация: MetaProfile

**MetaProfileBuilder** строит профиль для выбранного региона:
1. Берёт bbox страны/региона из `geo_entity`
2. Генерирует сетку точек с шагом 3° + buffer 2.5°
3. Для каждой точки × 52 недели запускает meta-model V2.4
4. Для каждого вида сохраняет maxScore (максимум по всем точкам и неделям)

**MetaProfile.apply()** — tiered alpha:

| Категория | maxScore | Alpha | Эффект |
|-----------|----------|-------|--------|
| COMMON | ≥ 0.30 | baseAlpha (0.10) | Минимальное подавление |
| IRRUPTIVE | ≥ 0.05 | 0.50 | Умеренное подавление |
| VAGRANT | ≥ 0.01 | 0.25 | Сильное подавление |
| OUTLIER | < 0.01 | 0.02 | Почти полное подавление |

Формула: `score[i] *= alpha + (1 - alpha) × maxScore[i]`

### 2.6. ModelMap: маппинг модели → справочник

CSV-файл для каждой модели (`model_map.csv`):

```csv
labelIndex,modelLabel,scientificName,taxonClass
0,Abroscopus albogularis,Abroscopus albogularis,Aves
347,Parus cinereus,Parus major,Aves
6500,Human vocal,,
```

- `modelLabel` — имя из модели (может быть устаревшим)
- `scientificName` — актуальное имя в справочнике (пустое = шум)
- `taxonClass` — для фильтрации

Генерируется Python-скриптом `dictionary/build_reference_db.py`.

### 2.7. DetectionAggregator

Агрегатор подтверждает виды через скользящее окно:

**Live Detection:**
- Window = 8 последних чанков
- Species подтверждается если ≥ 2 чанка выше threshold (0.5)
- Confidence = среднее top-3 scores в окне
- Виды с нулевым окном очищаются

**File Analysis:**
- Window = ∞ (все чанки)
- Species подтверждается если ≥ 1 чанк выше threshold
- Confidence = max score

### 2.8. Параллельный анализ файлов

`BirdDetectionPipeline.analyzeFileDetailed()`:

```
Producer (IO)          Workers (Default × N)     Collector (caller)
─────────────          ─────────────────────     ──────────────────
AudioFileDecoder       AudioChunkProcessor       Aggregate results
  ↓ chunks             + Classifier.classify     Report progress
  → Channel(8) ──────→ → Channel(8) ──────────→ Build timeline
                       (N=2 workers default)
```

- Одна декодировка файла, resample для V3.0 (48→32 kHz)
- N worker-корутин берут чанки из канала, обрабатывают параллельно
- Результаты собираются в порядке завершения (не гарантирован порядок)
- Async spectrogram: отдельный Channel → SpectrogramComputer

### 2.9. SpectrogramComputer

- STFT: 512 точек FFT, 256 hop, окно Hanning
- 32 mel-like log-spaced частотных бина
- Инкрементальное построение: `addChunk(samples)` → `build()`
- Глобальная нормализация [0, 1]
- Используется для визуализации на экране File Analysis

---

## 3. Аудио-сервисы

### 3.1. AudioRecorder

Singleton, обёртка над Android `AudioRecord` API.

- Source: `VOICE_RECOGNITION` (с AGC) → fallback `UNPROCESSED`
- Default: 48 kHz, 144,000 samples/chunk (3 сек), 50% overlap
- `configure(AudioConfig)` — переключение sample rate/chunk size для другой модели
- `chunksFlow() → Flow<FloatArray>` — перекрывающиеся чанки для inference
- `rawSamplesFlow() → Flow<FloatArray>` — ~100 мс порции для waveform
- `audioLevel: StateFlow<Float>` — RMS уровень (обновляется ~10 раз/сек)
- Запись стартует при подписке на Flow, останавливается при отмене

### 3.2. AudioPlaybackManager

Singleton, обёртка над `MediaPlayer`.

- States: IDLE → PLAYING ↔ PAUSED → IDLE
- `play(uri)`, `resume()`, `pause()`, `seekTo(ms)`, `seekToFraction(0.0..1.0)`
- StateFlows: `state`, `positionMs`, `durationMs`
- Обновление позиции ~10 раз/сек

---

## 4. Хранение данных

### 4.1. Архитектура: две базы данных

```
┌─────────────────────────────────────┐
│         ReferenceDatabase           │  ← Pre-built SQLite (createFromAsset)
│  "reference.db" — read-only        │
│                                     │
│  geo_entity      taxon_order        │
│  ml_model        taxon_family       │
│  geo_model       species            │
│                  species_name       │
│                  species_country    │
│                  taxonomy_synonym   │
│                  translation        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│           UserDatabase              │  ← Обычный Room (read-write)
│  "user.db" — пользовательские      │
│                                     │
│  file_analysis                      │
│  file_detection                     │
│  (future: observation,              │
│   user_species_list)                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│         DataStore                   │
│  "user_prefs" — настройки           │
│                                     │
│  countryCode, regionCode,           │
│  activeModel                        │
└─────────────────────────────────────┘
```

**Принцип разделения:**
- Справочные данные (reference.db) обновляются через замену файла при апдейте APK
- Пользовательские данные (user.db) не затрагиваются при обновлении справочника
- GeoSeedLoader удалён — все справочные данные в pre-built SQLite

### 4.2. Reference Database — Таблицы

#### geo_entity
Иерархия гео-объектов (adjacency list):

| Колонка | Тип | Описание |
|---------|-----|----------|
| code | TEXT PK | ISO 3166-1 alpha-2 (страны) или custom (EUR, RU-MOW) |
| type | TEXT | "continent" / "country" / "region" |
| parent_code | TEXT FK→self | Код родителя (null для континентов) |
| name_ru | TEXT | Название (для стран = ISO код, разрешается через Locale) |
| name_en | TEXT | Название (для стран = ISO код) |
| min_lat, max_lat | REAL | Bounding box для MetaProfile |
| min_lon, max_lon | REAL | Bounding box для MetaProfile |
| buffer_deg | REAL | Буфер вокруг bbox (default 2.5°) |
| sort_order | INT | Порядок сортировки |

**~276 записей:** 6 континентов + ~220 стран + ~50 регионов (РФ, Казахстан).

#### ml_model
Метаданные ML-моделей:

| Колонка | Тип | Описание |
|---------|-----|----------|
| id | TEXT PK | "BirdNET-V2.4-FP16", "BirdNET-V3.0-EUNA" |
| name | TEXT | Человекочитаемое имя |
| runtime | TEXT | "tflite" / "onnx" |
| audio_path | TEXT | Путь к аудио-модели в assets |
| meta_model_path | TEXT? | Путь к meta-модели (null для V3.0) |
| labels_path | TEXT | Путь к labels |
| sample_rate | INT | 48000 / 32000 |
| chunk_seconds | INT | 3 / 5 |
| species_count | INT | 6521 / 1225 |
| is_bundled | BOOL | true = в APK, false = sideload |

#### geo_model
Связь гео-объект ↔ доступная модель:

| Колонка | Тип | Описание |
|---------|-----|----------|
| geo_code | TEXT PK, FK→geo_entity | Код региона |
| model_id | TEXT PK, FK→ml_model | Код модели |
| is_default | BOOL | Модель по умолчанию для региона |

#### taxon_order
Таксономические отряды (~56 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| id | INT PK AUTO | ID отряда |
| latin_name | TEXT UNIQUE | "Passeriformes", "Orthoptera" |
| taxon_class | TEXT | "Aves", "Insecta", "Mammalia", "Amphibia", "Squamata" |

#### taxon_family
Таксономические семейства (~259 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| id | INT PK AUTO | ID семейства |
| latin_name | TEXT UNIQUE | "Paridae", "Corvidae" |
| order_id | INT FK→taxon_order | Ссылка на отряд |

#### species
Виды (~11,500 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| scientific_name | TEXT PK | "Parus major" — актуальное латинское имя |
| family_id | INT FK→taxon_family | Ссылка на семейство |
| genus | TEXT | "Parus" |
| iucn_status | TEXT? | "LC", "VU", "EN", "CR", "NT", NULL |

**Иерархия:** taxon_order → taxon_family → species (через FK).

#### species_name
Локализованные названия видов (~22,000 записей для en+ru):

| Колонка | Тип | Описание |
|---------|-----|----------|
| scientific_name | TEXT PK, FK→species | Латинское имя |
| lang | TEXT PK | "ru", "en", "de", ... |
| name | TEXT | "Большая синица", "Great Tit" |

Добавление нового языка = INSERT строк, без миграции схемы.

#### taxonomy_synonym
Устаревшие латинские имена и русские синонимы (~30,000 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| synonym | TEXT PK | "Parus cinereus" или "Обыкновенная зарянка" |
| scientific_name | TEXT FK→species | Актуальное имя: "Parus major" |
| type | TEXT | "old_latin" / "synonym_ru" |

**Применение:** модель BirdNET возвращает `Parus cinereus` → synonym lookup → `Parus major` → справочник.

#### species_country
Связь вид ↔ страна (~130,000 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| scientific_name | TEXT PK, FK→species | Латинское имя |
| country_code | TEXT PK | ISO код страны (ссылается на geo_entity) |

**Применение:** фильтрация справочника по стране, региональные паки (будущее).

#### translation
Переводы мелких сущностей (~718 записей):

| Колонка | Тип | Описание |
|---------|-----|----------|
| entity_type | TEXT PK | "order", "family", "continent", "region" |
| entity_key | TEXT PK | "Passeriformes", "Paridae", "EUR" |
| lang | TEXT PK | "ru", "en" |
| name | TEXT | "Воробьинообразные", "Синицевые", "Европа" |

### 4.3. User Database — Таблицы

#### file_analysis
Сохранённые анализы файлов:

| Колонка | Тип | Описание |
|---------|-----|----------|
| id | TEXT PK | UUID |
| file_name | TEXT | Имя файла |
| file_uri | TEXT | Content URI |
| duration_sec | REAL | Длительность аудио |
| file_size_bytes | INT | Размер файла |
| region_code | TEXT? | Код региона на момент анализа |
| region_label | TEXT? | Отображаемое имя региона |
| v30_available | BOOL | Была ли доступна V3.0 |
| waveform_data | BLOB? | Сжатая waveform для preview |
| created_at | INT | Timestamp создания |
| species_count | INT | Количество обнаруженных видов |
| analysis_duration_ms | INT | Время анализа в мс |

#### file_detection
Детекции в анализах:

| Колонка | Тип | Описание |
|---------|-----|----------|
| id | TEXT PK | UUID |
| analysis_id | TEXT FK→file_analysis | Ссылка на анализ |
| scientific_name | TEXT | Латинское имя (снимок на момент детекции) |
| common_name | TEXT | Обиходное имя |
| start_time_sec | REAL | Начало фрагмента в файле |
| end_time_sec | REAL | Конец фрагмента |
| v24_confidence | REAL? | Уверенность V2.4 |
| v30_confidence | REAL? | Уверенность V3.0 |

### 4.4. DataStore (Preferences)

| Ключ | Тип | Default | Описание |
|------|-----|---------|----------|
| countryCode | String | "BY" | Выбранная страна |
| regionCode | String? | null | Выбранный регион |
| activeModel | String | "birdnet_v24" | Активная модель |

### 4.5. Репозитории

| Репозиторий | Источник | Назначение |
|-------------|----------|------------|
| GeoRepository | GeoDao + PreferencesRepository | Гео-навигация, текущий выбор |
| SpeciesRepository | SpeciesDao | Справочник: карточка, поиск, фильтры |
| FileAnalysisRepository | FileAnalysisDao + UserDatabase | Сохранение/загрузка анализов |
| PreferencesRepository | DataStore | Настройки пользователя |

---

## 5. UI / Presentation

### 5.1. Навигация

```
SplashRoute → PermissionRoute → MainGraph
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                      HomeRoute  InfoRoute  ProfileRoute
                      (Tab: 🎙)  (Tab: ℹ)  (Tab: 👤)
                         │
              ┌──────────┼──────────┐
              │                     │
    DualDetectionRoute      FileAnalysisRoute
    (Push: live анализ)     (Push: анализ файла)
              │                     │
         DetailRoute          HistoryRoute
         (Push: карточка)     (Push: история)

    LocationPickerRoute     SettingsRoute
    (Push: выбор региона)   (Push: настройки)
```

**3 таба Bottom Navigation:** Слушать (Home/Mic), Инфо (Info), Профиль (Profile).

### 5.2. Экраны

#### SplashScreen
- Тёмная тема (#050C18), полигональный логотип Avalga
- Прогресс-бар: фазы 0→1→2 (логотип → имя → прогресс)
- Фоновая загрузка BirdClassifier, прогресс capped at 88% до завершения

#### PermissionScreen
- Запрос `RECORD_AUDIO`
- Два состояния: request / denied (с кнопкой в настройки)

#### HomeScreen (Mode Hub)
- 4 карточки-плитки:
  - **Live Detection** (зелёная, активная) — непрерывный анализ с микрофона
  - **File Analysis** (синяя, активная) — анализ аудиофайла
  - **Ловушка** (фиолетовая, "скоро") — VAD-триггер
  - **Экспедиция** (жёлтая, "скоро") — групповые записи
- Цвета из `HubColors`

#### DualDetectionScreen (Live Detection)
- Dual V2.4 + V3.0 inference в реальном времени
- **Компоненты:**
  - WaveBars — анимированные частотные полоски
  - RadarCanvas — круговая визуализация
  - RecordButton (96dp) — старт/стоп
  - BirdListItem — карточка вида с ConfBar
  - SessionCompleteBanner — save/discard
- **States:** IDLE → RECORDING ↔ PAUSED → IDLE
- GPS + MetaProfile для geo-фильтрации
- Family shadowing: подавление дублей из одного семейства

#### FileAnalysisScreen
- **Фазы:** IDLE → READY → ANALYZING ↔ PAUSED → DONE / ERROR
- **Split StateFlows** (оптимизация recomposition):
  - `coreState` — phase, fileName, fileDuration (редко меняется)
  - `progressState` — progress %, elapsed, model progress (часто)
  - `spectrogramState` — columns, birdMarkers (по мере вычисления)
  - `timelineState` — timeline birds, species summaries
  - `playbackUiState` — isPlaying, position
- **Layout:** DropZone (IDLE) → FileCard + Spectrogram + Controls + Timeline
- FAB → HistoryRoute

#### HistoryScreen
- Список `FileAnalysisSummary` (newest first)
- Карточки: fileName, duration, speciesCount, date
- Тап → FileAnalysisRoute(analysisId)
- Swipe to delete

#### LocationPickerScreen
- Мультишаговый: Континенты → Страны → Регионы
- Breadcrumb-навигация
- Locale-aware сортировка названий

#### SettingsScreen
- Выбор модели (V2.4 / V3.0 если доступна)
- Текущий регион (тап → LocationPicker)

#### DetailScreen
- Карточка вида: commonName, scientificName, confidence V24/V30
- Плеер аудиофрагмента

### 5.3. Тема

**HubColors** (тёмная тема):

| Токен | Цвет | Использование |
|-------|------|--------------|
| Bg | #050C18 | Фон экранов |
| BgEl2 | #162842 | Фон карточек |
| Accent | #E8A020 | Акцентный (жёлтый) |
| Green | #3DBA7E | Live Detection, success |
| Blue | #4BA3C7 | File Analysis |
| Purple | #9B7FE8 | Ловушка |
| Red | #E05050 | Ошибки, удаление |
| Yellow | #E8C020 | Экспедиция |

Material Design 3: светлая + тёмная тема, dynamic colors.

---

## 6. Dependency Injection

### AppModule

| Провайдер | Тип | Описание |
|-----------|-----|----------|
| provideDataStore | DataStore<Preferences> | "user_prefs" |
| provideReferenceDatabase | ReferenceDatabase | createFromAsset("db/reference.db") |
| provideUserDatabase | UserDatabase | "user.db" |
| provideGeoDao | GeoDao | из ReferenceDatabase |
| provideSpeciesDao | SpeciesDao | из ReferenceDatabase |
| provideFileAnalysisDao | FileAnalysisDao | из UserDatabase |

### MlModule

| Провайдер | Тип | Описание |
|-----------|-----|----------|
| provideAudioChunkProcessor | AudioChunkProcessor | PASSTHROUGH mode |
| provideAudioModel | MappedByteBuffer | V2.4 audio-model-fp16.tflite |
| provideMetaModel | MappedByteBuffer | V2.4 meta-model.tflite |
| provideLabels | List<Pair<String, String>> | V2.4 labels/ru.txt |
| provideV24ModelMap | ModelMap | V2.4 model_map.csv |
| provideBirdClassifier | BirdClassifier | BirdNetV24Classifier |
| provideFamilyTaxonomy | FamilyTaxonomy | genus_families.json |
| provideMetaProfileBuilder | MetaProfileBuilder | Для построения MetaProfile |

---

## 7. Assets

```
app/src/main/assets/
├── birdnet/
│   ├── v24/
│   │   ├── audio-model-fp16.tflite    # Аудио-модель V2.4 (TFLite FP16)
│   │   ├── meta-model.tflite           # Мета-модель (geo + temporal)
│   │   ├── labels/
│   │   │   ├── ru.txt                  # Labels: "ScientificName_CommonName"
│   │   │   └── en_us.txt
│   │   ├── model_map.csv              # labelIndex → scientificName + taxonClass
│   │   ├── genus_families.json         # genus → family lookup
│   │   └── sample.wav                  # Тестовый аудио-семпл
│   └── v30/
│       ├── labels.csv                  # id;sci_name;com_name;gbif;class;order
│       └── model_map.csv              # labelIndex → scientificName + taxonClass
│       # birdnet_v30_euna.onnx — sideload via adb push, НЕ в assets
│
├── db/
│   └── reference.db                    # Pre-built SQLite (16 МБ, ~5 МБ в APK)
│
└── geo/
    └── seed.json                       # (legacy, данные теперь в reference.db)
```

---

## 8. Генерация данных

### Python-скрипт: `dictionary/build_reference_db.py`

Запускается один раз при подготовке релиза. Генерирует все справочные данные.

**Входные данные:**

| Источник | Описание |
|----------|----------|
| `dictionary/bird/bird_data_collector/output/summary.csv` | 11,250 видов: имя, отряд, семейство, род, IUCN |
| `dictionary/bird/bird_data_collector/output/synonyms.csv` | 31,000+ синонимов (old_latin + synonym_ru) |
| `dictionary/bird/bird_data_collector/output/names_all_languages.csv` | 330,000 переводов на ~70 языков |
| `dictionary/countries_bird/*.json` | 250 файлов: виды по странам |
| `assets/geo/seed.json` | Гео-данные (континенты, страны, регионы, bbox) |
| `assets/birdnet/v24/labels/en_us.txt` | V2.4 labels (6,521) |
| `assets/birdnet/v30/labels.csv` | V3.0 labels (1,225) |

**Выходные данные:**

| Файл | Размер | Описание |
|------|--------|----------|
| `assets/db/reference.db` | ~16 МБ → ~5 МБ в APK | Полная справочная БД |
| `assets/birdnet/v24/model_map.csv` | ~250 КБ | Маппинг V2.4 labels → справочник |
| `assets/birdnet/v30/model_map.csv` | ~50 КБ | Маппинг V3.0 labels → справочник |
| `dictionary/report.txt` | — | Отчёт о качестве данных |

**Обработка грязных данных:**
- 13 записей с загрязнёнными `name_lat` (заголовки страниц dibird.com) автоматически очищаются через regex
- Русские имена берутся из `summary.csv` (не из `names_all_languages.csv`, где нет языка "Русский")
- 252 не-птичьих вида из V3.0 автоматически добавляются в справочник

---

## 9. Performance

| Метрика | Значение |
|---------|----------|
| UI frame budget | 16 мс (тяжёлые операции на IO/Default) |
| Chunk inference V2.4 | < 500 мс на среднем устройстве |
| Chunk inference V3.0 | < 1,5 сек |
| MetaProfile build | 2-3 сек (async, один раз при старте) |
| First launch (reference.db) | < 1 сек (file copy, no parsing) |
| APK size | < 150 МБ (включая V2.4 модель + reference.db) |
| RAM | < 200 МБ в активном режиме |

---

## 10. Будущие расширения (заложены в архитектуру)

| Фича | Что уже подготовлено |
|------|---------------------|
| Справочник видов (UI) | SpeciesDao + SpeciesRepository + reference.db |
| Фильтр по таксонам (не только птицы) | taxonClass в BirdDetection + enabledClasses в classify() |
| Новые языки | INSERT в species_name + translation, без миграций |
| Обновление справочника по сети | Замена reference.db, user.db не затрагивается |
| Фото/позывки (региональные паки) | Структура species_media заложена, пока не реализована |
| Похожие виды | Структура species_similar заложена |
| Лайфлист ("мой список") | UserDatabase + будущая таблица observation |
| Новые ML-модели | Единый формат model_map.csv + BirdClassifier interface |
