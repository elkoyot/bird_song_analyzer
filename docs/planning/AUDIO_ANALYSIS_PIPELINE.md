# Audio Analysis Pipeline

Документация реального pipeline распознавания птиц по аудио в live-режиме.

## Общая схема

```
Микрофон (48 kHz, mono, PCM16)
    │
    ▼
AudioRecorder — непрерывный захват, chunking (3 сек, 50% overlap)
    │
    │  FloatArray[144000] каждые ~1.5 сек
    ▼
AudioChunkProcessor — 6-этапная пре-фильтрация + bandpass + нормализация
    │
    │  FloatArray[144000] или null (skip)
    ▼
BirdNetV24Classifier — TFLite inference + sigmoid + мета-модель (geo-aware)
    │
    │  List<BirdDetection> (вид, confidence 0..1), до 10 видов
    ▼
DetectionAggregator — sliding window, подтверждение ≥2 chunk-ов, фильтрация не-птиц
    │
    │  Map<scientificName, AggregatedDetection>
    ▼
filterDetections() — 2-path фильтр (anchor + aggregator) + family dedup
    │
    │  List<BirdDetection> (финальный набор видов для UI)
    ▼
LiveDetectionScreen — лента обнаруженных видов (prepend, dedup consecutive)
```

---

## 1. Захват аудио — AudioRecorder

**Файл:** `service/AudioRecorder.kt`

| Параметр | Значение |
|----------|----------|
| AudioSource | VOICE_RECOGNITION (приоритет), UNPROCESSED (fallback) |
| Sample rate | 48 000 Hz |
| Формат | PCM 16-bit, mono |
| Chunk | 144 000 сэмплов = 3 секунды |
| Hop | 72 000 сэмплов = 1.5 секунды (50% overlap) |
| Read buffer | ~4 800 сэмплов = 100 мс (SAMPLE_RATE / 10) |

### Как работает chunking

AudioRecorder читает аудио блоками по ~4 800 сэмплов (100 мс) и копирует в аккумулятор. Когда аккумулятор заполняется (144 000 сэмплов = 3 секунды), chunk emit-ится во Flow.

После emit-а вторая половина аккумулятора копируется в начало — это даёт **50% перекрытие** между соседними chunk-ами:

```
Время: 0s        1.5s       3s        4.5s       6s
       |──chunk 1──|
                   |──chunk 2──|
                               |──chunk 3──|
```

Каждый chunk — независимая копия `FloatArray`, нормализованная в `[-1, 1]` (`short / 32768`).

### Почему VOICE_RECOGNITION

`AudioSource.MIC` на многих устройствах применяет аппаратное шумоподавление и AGC, которые могут подавить тихие звуки птиц как «шум». `VOICE_RECOGNITION` отключает эти обработки и обеспечивает более высокий gain. Fallback на `UNPROCESSED` если `VOICE_RECOGNITION` недоступен.

### Audio Level Meter

Параллельно записи обновляется `audioLevel: StateFlow<Float>` ~10 раз в секунду (RMS сигнала). Используется для визуализации уровня аудио в UI.

---

## 2. Пре-обработка — AudioChunkProcessor

**Файл:** `ml/AudioChunkProcessor.kt`

Stateless-процессор, применяемый **до** ML-инференса. Фильтрует тишину, клиппинг и не-птичий шум, применяет bandpass-фильтр и нормализует уровень. Возвращает обработанный chunk или `null` (пропуск).

6 этапов в порядке выполнения:

### 2.1. Silence check

```
RMS < 0.001 → пропуск (тишина)
```

RMS (Root Mean Square) — среднеквадратичная амплитуда. При RMS < 0.001 (~-60 dBFS) chunk содержит только фоновый шум без полезного сигнала.

### 2.2. Clipping check

```
peak > 0.99 AND rms > 0.3 → пропуск (клиппинг)
```

Оба условия должны выполняться: высокий peak сам по себе допустим (одиночный щелчок), но в сочетании с высоким RMS это означает насыщение сигнала (ветер в микрофон, удар).

### 2.3. Spectral check (алгоритм Goertzel)

Быстрая проверка спектрального распределения энергии без полного FFT. Goertzel вычисляет энергию на одной частоте за O(N) — для 4 частот это ~2 мс на chunk.

**Четыре контрольные частоты:**

| Частота | Полоса | Назначение |
|---------|--------|------------|
| 100 Гц | low | Не-птичий шум: моторы, HVAC, 50/60 Гц гул и гармоники |
| 500 Гц | bird-low | Низкочастотные птицы: голуби (~120 Гц + гармоники), совы (300-500 Гц) |
| 3000 Гц | bird-mid | Типичные певчие птицы |
| 12000 Гц | high | Выше диапазона птиц: электроника, насекомые |

**Логика:**
- `totalEnergy = low + birdLow + birdMid + high`
- `lowRatio = low / total`, `highRatio = high / total`
- Если `lowRatio ≥ 0.95` или `highRatio ≥ 0.95` → пропуск (>95% энергии вне диапазона птиц)
- Bird-low и bird-mid считаются «птичьей» энергией и не вызывают пропуск
- Если `totalEnergy < 1e-12` → пропуск — negligible energy, silence check обработает

**Почему 100 Гц, а не 300 Гц для low-band:**
Ранняя версия использовала 300 Гц как границу «не-птичьего» шума. Это отсекало голубей (Columba livia, фундаментальная ~120 Гц с гармониками на 240/480 Гц) и сов (Asio otus, вокализация ~300-500 Гц). Перенос границы на 100 Гц решил проблему: гармоники голубей (480 Гц) попадают в bird-low полосу (500 Гц Goertzel), а совы вообще не попадают в low-band.

### 2.4. Bandpass filter

**Файл:** `ml/BandpassFilter.kt`

Два каскадных biquad-фильтра Баттерворта 2-го порядка:

| Фильтр | Тип | Частота среза | Назначение |
|--------|-----|---------------|------------|
| High-pass | Butterworth 2nd order | **80 Гц** | Убирает инфразвук, 50/60 Гц гул, ветровой шум |
| Low-pass | Butterworth 2nd order | **15 000 Гц** | Убирает электронный шум, ультразвук |

**Коэффициенты:** вычисляются по Audio EQ Cookbook (Robert Bristow-Johnson). Q = 1/√2 для максимально плоской АЧХ Баттерворта.

**Реализация:** Direct Form II Transposed biquad. Состояние фильтра сбрасывается при каждом вызове — между chunk-ами нет переходных процессов (2-й порядок: переходный процесс затухает за <10 сэмплов).

**АЧХ (характерные точки):**

| Частота | Затухание | Примечание |
|---------|-----------|------------|
| 50 Гц | ~-9 дБ | Ниже HP cutoff, сильно ослаблен |
| 80 Гц | -3 дБ | HP cutoff |
| 150 Гц | ~-1 дБ | Хорошо пропускает (голуби, совы) |
| 1-5 кГц | 0 дБ | Полоса пропускания |
| 15 кГц | -3 дБ | LP cutoff |
| 20 кГц | ~-8 дБ | Сильно ослаблен |

**Почему 80 Гц, а не 150 Гц:**
HP cutoff 150 Гц обрезал фундаментальную частоту голубя (~120 Гц), изменяя спектральную сигнатуру настолько, что BirdNET не мог распознать вид. При 80 Гц сигнал на 120 Гц проходит с ~91% амплитуды.

### 2.5. Post-filter silence check

```
postFilterPeak < 0.001 → пропуск
```

Если после bandpass-фильтра сигнал стал почти нулевым — значит вся энергия была за пределами полосы (шум, не птица).

### 2.6. Peak normalization

```
peak в [0.001, 0.9] → gain = 0.9 / peak, каждый сэмпл *= gain (clamp к [-1, 1])
peak > 0.9          → без изменений
```

Компенсирует тихие записи, приводя пиковую амплитуду к 0.9. Громкие сигналы (peak > 0.9) не трогаем — избыточное усиление может привести к клиппингу.

### Параметры AudioChunkProcessor (сводка)

| Параметр | Значение | Константа |
|----------|----------|-----------|
| Порог тишины (RMS) | 0.001 | `SILENCE_RMS_THRESHOLD` |
| Порог клиппинга (peak) | 0.99 | `CLIPPING_PEAK_THRESHOLD` |
| Порог клиппинга (RMS) | 0.3 | `CLIPPING_RMS_THRESHOLD` |
| Порог спектрального отклонения | 95% | `SPECTRAL_REJECT_RATIO` |
| HP cutoff | 80 Гц | `LOW_CUTOFF` |
| LP cutoff | 15 000 Гц | `HIGH_CUTOFF` |
| Целевой peak | 0.9 | `NORM_TARGET` |
| Пост-фильтр тишина | 0.001 | `POST_FILTER_SILENCE_THRESHOLD` |

### Диагностика

AudioChunkProcessor ведёт счётчики: `totalChunks`, `passedChunks`, `silenceRejects`, `clippingRejects`, `spectralRejects`, `postFilterRejects`. Доступны через `statsLine()`.

---

## 3. Классификация — BirdNetV24Classifier

**Файл:** `ml/BirdNetV24Classifier.kt`

### Вход

- `FloatArray[144 000]` — 3 секунды обработанного audio, `[-1, 1]` (после AudioChunkProcessor)
- Опционально: `LocationMeta` (GPS + неделя года) для per-chunk мета-модели
- Опционально: `MetaProfile` (предвычисленный региональный профиль) — приоритетнее LocationMeta

### Шаги обработки

#### 3.1. Audio model (TFLite)

BirdNET V2.4 FP16 — свёрточная нейросеть:

```
FloatArray[1, 144000] → audio-model-fp16.tflite → FloatArray[1, 6522]
```

- **Вход:** 3 секунды audio (48 kHz, mono, float32)
- Модель внутри вычисляет mel-спектрограмму, затем CNN извлекает признаки
- **Выход:** 6 522 logit-а (по одному на каждый вид/класс в labels)

Logit-ы — это сырые значения нейросети, НЕ вероятности. Они могут быть отрицательными.

#### 3.2. Sigmoid

Logit-ы конвертируются в вероятности через sigmoid:

```
probability = 1 / (1 + exp(-logit))
```

| Logit | Sigmoid | Интерпретация |
|-------|---------|---------------|
| -5.0 | 0.007 | шум |
| -2.0 | 0.119 | маловероятно |
| 0.0 | 0.500 | 50/50 |
| 2.0 | 0.881 | вероятно |
| 5.0 | 0.993 | почти точно |

#### 3.3. Мета-модель (geo-aware фильтрация)

После sigmoid применяется географическая и сезонная коррекция. Есть два пути (взаимоисключающие, MetaProfile приоритетнее):

**Путь А: MetaProfile (основной)**

Если `metaProfile != null` — используется предвычисленный региональный профиль (см. раздел 5). Применяется tiered alpha:

```kotlin
for (i in scores.indices) {
    val m = maxScores[i]
    val effectiveAlpha = when {
        m >= 0.30  → baseAlpha (0.10)  // обычный вид
        m >= 0.05  → 0.50               // инвазионный
        m >= 0.01  → 0.25               // редкий залётный
        else       → 0.02               // континентальный выброс
    }
    scores[i] *= effectiveAlpha + (1 - effectiveAlpha) * m
}
```

**Путь Б: Per-chunk LocationMeta (GPS)**

Если MetaProfile нет, но есть GPS-координаты — мета-модель запускается «на лету» для конкретной точки и диапазона недель:

```
[latitude, longitude, weekOfYear] → meta-model.tflite → FloatArray[6522]
```

Blending: `scores[i] *= alpha + (1 - alpha) * rawMeta[i]`, где `alpha = 0.10`.

**Логика:** Оба пути выполняют одну функцию — подавляют виды, невозможные в данной местности/сезоне, и усиливают вероятные. MetaProfile работает лучше, т.к. учитывает весь регион (не одну точку) и весь год (с tiered alpha).

#### 3.4. Фильтрация

Виды с confidence ≥ 0.1 возвращаются из классификатора (до 10 лучших — `topK`). Низкий порог 0.1 сохраняет кандидатов для агрегации.

### Выход

```kotlin
List<BirdDetection>(
    scientificName = "Parus major",
    commonName = "Большая синица",
    confidence = 0.987,     // после sigmoid + мета
    labelIndex = 3847,
)
```

### Параметры BirdNetV24Classifier

| Константа | Значение | Описание |
|-----------|----------|----------|
| `DEFAULT_THRESHOLD` | 0.1 | Минимальный confidence для выдачи |
| `DEFAULT_TOP_K` | 10 | Максимум видов на chunk |
| `DEFAULT_NUM_THREADS` | 2 | Потоки TFLite |
| `DEFAULT_META_ALPHA` | 0.10 | Базовый вес аудио-модели при blending |

---

## 4. Агрегация — DetectionAggregator

**Файл:** `ml/DetectionAggregator.kt`

Управляет скользящим окном обнаружений, подтверждением видов и фильтрацией не-птичьих меток.

### 4.1. Не-птичьи метки

Модель BirdNET включает классы для не-птичьих звуков. Они отфильтровываются до агрегации:

```kotlin
NON_BIRD_LABELS = setOf(
    "Engine", "Environmental", "Fireworks", "Gun",
    "Human vocal", "Noise", "Power tools", "Siren",
    "Apis mellifera",
)
```

Фильтрация происходит в `DetectionAggregator` при вызове `addChunkResults()`.

### 4.2. Sliding window (live detection)

Для каждого вида хранится `ArrayDeque<Float>` — последние N confidence-значений.

```
Window = 8 chunks (~12 секунд с 50% overlap)
```

При каждом новом chunk-е:
1. Для обнаруженных видов — добавляется confidence
2. Для ранее отслеживаемых, но не обнаруженных — добавляется 0.0
3. Если `deque.size > windowSize` — старейшее значение удаляется
4. Виды с окном из нулей удаляются (только в windowed-режиме)

### 4.3. Подтверждение

Вид считается **подтверждённым**, если:

```
(количество scores >= threshold в окне) >= confirmationCount
```

В live-режиме: `threshold = 0.10`, `confirmationCount = 2`.

Это означает: вид должен появиться **минимум в 2 chunk-ах** с confidence ≥ 10% в окне из 8 chunk-ов. Одиночный chunk с высоким confidence — не достаточно для подтверждения через агрегатор.

### 4.4. Confidence calculation

**Live mode — avg-top-3:**

Из всех значений в окне берутся 3 наибольших, их среднее = итоговый confidence.

Пример: окно `[0.9, 0.8, 0.7, 0.3, 0.2, 0, 0, 0]` → top-3: `[0.9, 0.8, 0.7]` → confidence = 0.80.

### 4.5. Параметры DetectionAggregator (live mode)

| Параметр | Значение по умолчанию | Значение в LiveDetectionVM | Константа |
|----------|-----------------------|---------------------------|-----------|
| Window size | 8 chunks | 8 chunks | `DEFAULT_WINDOW_SIZE` |
| Confirmation count | 2 chunks | 2 chunks | `AGGREGATOR_CONFIRMATION` |
| Threshold | 0.50 | **0.10** | `AGGREGATOR_THRESHOLD` |
| Confidence calc | avg-top-3 | avg-top-3 | `useAvgTop3 = true` |

**Важно:** LiveDetectionViewModel создаёт агрегатор с `threshold = 0.10`, а не с дефолтным `0.50`. Это снижает порог подтверждения, позволяя агрегатору ловить виды, которые не достигают 75% anchor-порога, но стабильно появляются.

---

## 5. MetaProfile — предвычисленный региональный профиль

**Файлы:** `ml/MetaProfile.kt`, `ml/MetaProfileBuilder.kt`, `ml/CountryConfig.kt`

### 5.1. Что такое MetaProfile

MetaProfile хранит **максимальный score мета-модели** для каждого из 6 522 видов по всему региону и за весь год. Вместо того чтобы запрашивать мета-модель для каждой конкретной GPS-точки и недели, один раз вычисляется: «Какова максимальная вероятность встретить этот вид где-либо в моём регионе в любое время года?»

### 5.2. Как строится

При старте сессии `LiveDetectionViewModel`:

1. Загружает `countryCode` и `regionCode` из DataStore (выбранные в Settings)
2. Находит `CountryConfig` с bounding box региона
3. `MetaProfileBuilder` расширяет bbox на `bufferDeg = 2.5°` во все стороны
4. Создаёт сетку точек с шагом `gridStepDeg = 3.0°`
5. Для **каждой точки × каждой недели (1-52)** запускает мета-модель
6. Сохраняет **максимум** per species

**Пример для Беларуси:**
- Bbox ≈ 51–56°N × 23–33°E + buffer → ~35 точек
- 35 × 52 = 1 820 inference
- Время: 2–3 секунды

### 5.3. Tiered alpha

Ключевая идея: **разный вес мета-модели для разных уровней «ожидаемости» вида**.

| Тир | Score мета-модели (m) | Effective Alpha | Пример |
|-----|-----------------------|-----------------|--------|
| **COMMON** | m ≥ 0.30 | 0.10 (baseAlpha) | Большая синица в Минске |
| **IRRUPTIVE** | 0.05 ≤ m < 0.30 | 0.50 | Кедровка (инвазия) |
| **VAGRANT** | 0.01 ≤ m < 0.05 | 0.25 | Редкий залётный вид |
| **OUTLIER** | m < 0.01 | 0.02 | Тукан в Беларуси |

**Формула blending:**

```
scores[i] *= effectiveAlpha + (1 - effectiveAlpha) * m
```

**Как это работает на примерах:**

| Вид | Audio score | Meta score (m) | Тир | Итоговый score |
|-----|------------|----------------|-----|----------------|
| Большая синица | 0.95 | 0.85 | COMMON | 0.95 × (0.10 + 0.90 × 0.85) = 0.95 × 0.865 = **0.82** |
| Кедровка | 0.80 | 0.15 | IRRUPTIVE | 0.80 × (0.50 + 0.50 × 0.15) = 0.80 × 0.575 = **0.46** |
| Редкий залётный | 0.70 | 0.03 | VAGRANT | 0.70 × (0.25 + 0.75 × 0.03) = 0.70 × 0.2725 = **0.19** |
| Тукан | 0.60 | 0.001 | OUTLIER | 0.60 × (0.02 + 0.98 × 0.001) = 0.60 × 0.02098 = **0.013** |

**Эффект:** обычные виды почти не теряют confidence, инвазионные умеренно снижаются, континентальные выбросы практически обнуляются (тукан: 60% → 1.3%).

### 5.4. CountryConfig

**Файл:** `assets/birdnet/v24/countries.json`

Поддерживается ~42 страны (СНГ + Европа). Россия разделена на 8 регионов, Казахстан на 5. Каждая конфигурация включает bounding box и опциональный buffer.

---

## 6. Определение GPS-локации

**Файл:** `presentation/detection/LiveDetectionViewModel.kt` → `resolveLocation()`

При старте сессии:

1. Проверяется разрешение `ACCESS_COARSE_LOCATION`
2. Если есть — берётся последняя известная позиция (GPS_PROVIDER → NETWORK_PROVIDER)
3. Текущая неделя года (ISO) ± 4 недели = `weekRange`
4. При переходе через границу года (week < 5 или > 48) → `weekRange = 1..52` (весь год)

**LocationMeta** используется как fallback, когда MetaProfile недоступен. Если есть и MetaProfile, и LocationMeta — MetaProfile имеет приоритет (проверка в `BirdNetV24Classifier.classify()`).

---

## 7. Финальная фильтрация — 2-path + family dedup

**Файл:** `presentation/detection/LiveDetectionViewModel.kt` → `filterDetections()`

Это самый важный этап — решает, какие виды из 10 кандидатов попадают на экран.

### 7.1. Family Taxonomy

**Файл:** `ml/FamilyTaxonomy.kt`, `assets/birdnet/v24/genus_families.json`

Виды птиц группируются по таксономическим семействам (genus → family). Используется для подавления путаницы модели между близкими видами.

```
Parus major → Paridae
Sylvia borin → Sylviidae
Curruca nisoria → Sylviidae  ← одно семейство с Sylvia
```

### 7.2. Путь 1: Anchor (высокая уверенность)

```
confidence ≥ 0.75 → немедленный вывод на экран
```

Виды с confidence ≥ 75% считаются «якорями» — модель достаточно уверена. Их семейства запоминаются как `anchorFamilies`.

### 7.3. Путь 2: Aggregator-confirmed (подтверждение агрегатором)

Виды, не достигшие 75%, проверяются через агрегатор:

```
вид подтверждён (≥2 chunk-а) И семейство НЕ имеет anchor → вывод на экран
```

Confidence для таких видов берётся из агрегатора (avg-top-3), а не из текущего chunk-а.

**Пример:** Славка-завирушка (Sylvia curruca) c confidence 0.60 — ниже anchor-порога 0.75. Но если она подтверждена агрегатором (появилась в ≥2 chunk-ах) и в текущем chunk-е нет более уверенного Sylviidae — она попадёт на экран.

### 7.4. Подавление по семейству

Если у семейства уже есть anchor (confidence ≥ 0.75), все остальные виды этого семейства **подавляются**, даже если подтверждены агрегатором. Логика: если модель уверена на 80% что это Parus major, то Parus caeruleus с 30% — это модельный шум.

### 7.5. Family dedup

Финальный шаг: из всех кандидатов, прошедших пути 1 и 2, оставляется **только один вид на семейство** — с максимальным confidence.

### 7.6. Сводная таблица решений

| Confidence | Подтверждён агрегатором? | Семейство имеет anchor? | Результат |
|-----------|-------------------------|------------------------|-----------|
| ≥ 0.75 | не важно | этот вид — anchor | **PASS** (anchor) |
| < 0.75 | ДА | НЕТ | **PASS** (aggregator-confirmed) |
| < 0.75 | ДА | ДА | **SUPPRESSED** (family has anchor) |
| < 0.75 | НЕТ | не важно | **SUPPRESSED** (not confirmed) |

---

## 8. Вывод в UI — LiveDetectionScreen

### 8.1. Append-логика

Новые обнаружения добавляются **в начало списка** (newest first). Если последний добавленный вид = предыдущий верхний элемент (consecutive same species):
- Временное окно расширяется: `"00:15 – 00:18"` → `"00:15 – 00:21"`
- Confidence обновляется до максимума из старого и нового

### 8.2. Buffer

```kotlin
.buffer(capacity = 1, onBufferOverflow = DROP_OLDEST)
```

Если inference не успевает за аудио-потоком — старые chunk-и дропаются. Приоритет: свежие данные.

### 8.3. Лимит

Максимум `200` записей в UI-списке.

---

## 9. Подготовка сессии — lifecycle

### 9.1. init {}

При создании ViewModel запускается фоновая задача `buildMetaProfileAsync()`:
- Загружает `countryCode`/`regionCode` из DataStore
- Находит CountryConfig → строит MetaProfile
- Присваивает результат в `pipeline.classifier.metaProfile`

### 9.2. onStart()

```
IDLE/STOPPED → PREPARING → ждём MetaProfile → startDetection() → ANALYZING
```

Состояние `PREPARING` отображается в UI (кнопка заблокирована). Типичная задержка: 2-3 сек.

### 9.3. startDetection()

1. `resolveLocation()` — GPS (если доступен)
2. `aggregator.reset()` — очистка окна
3. Старт таймера, сбора уровня аудио, recording loop

---

## 10. Полная сводка параметров

### Запись

| Параметр | Значение |
|----------|----------|
| Sample Rate | 48 000 Hz |
| Chunk Duration | 3 секунды |
| Samples Per Chunk | 144 000 |
| Hop Size | 72 000 (50% overlap) |
| Audio Source | VOICE_RECOGNITION → UNPROCESSED |

### Пре-обработка

| Параметр | Значение |
|----------|----------|
| Silence RMS threshold | 0.001 |
| Clipping: peak / RMS | 0.99 / 0.3 |
| Spectral reject ratio | 95% |
| Bandpass | 80 Hz – 15 kHz |
| Post-filter silence | 0.001 (peak) |
| Normalization target | 0.9 (peak) |

### Классификация

| Параметр | Значение |
|----------|----------|
| Inference threshold | 0.1 (10%) |
| Top-K | 10 |
| TFLite threads | 2 |
| Meta-alpha (blending) | 0.10 |

### Мета-модель

| Параметр | Значение |
|----------|----------|
| Buffer distance | 2.5° |
| Grid step | 3.0° |
| Week range | 1–52 |
| Tier COMMON | m ≥ 0.30 → alpha 0.10 |
| Tier IRRUPTIVE | m ≥ 0.05 → alpha 0.50 |
| Tier VAGRANT | m ≥ 0.01 → alpha 0.25 |
| Tier OUTLIER | m < 0.01 → alpha 0.02 |

### Агрегация (live)

| Параметр | Значение |
|----------|----------|
| Window size | 8 chunks (~12 сек) |
| Confirmation count | 2 chunks |
| Threshold | 0.10 |
| Confidence calc | avg top-3 |

### Финальная фильтрация

| Параметр | Значение |
|----------|----------|
| Anchor threshold | 0.75 (75%) |
| Family dedup | лучший per family |
| Max UI detections | 200 |

### Локация

| Параметр | Значение |
|----------|----------|
| Week window | ±4 недели |
| Providers | GPS → Network |
| Permission | ACCESS_COARSE_LOCATION |

---

## 11. Путь аудио-chunk-а: пример

Допустим, пользователь в Минске, июнь, запись в парке.

```
1. Микрофон записывает 3 секунды (144k samples)
   → Chunk #17, содержит пение синицы

2. AudioChunkProcessor:
   ✓ RMS = 0.032 (> 0.001, не тишина)
   ✓ Peak = 0.45 (< 0.99, нет клиппинга)
   ✓ Spectral: bird-mid(3kHz) = 65%, low(100Hz) = 5% (< 95%)
   ✓ Bandpass 80Hz–15kHz applied
   ✓ Post-peak = 0.38 (> 0.001)
   ✓ Normalize: gain = 0.9 / 0.38 = 2.37×

3. BirdNetV24Classifier:
   Audio model → logits → sigmoid:
     Parus major: logit=3.5 → sigmoid=0.97
     Parus caeruleus: logit=1.2 → sigmoid=0.77
     Sitta europaea: logit=0.4 → sigmoid=0.60
     (ещё 7 видов с ≥0.1)

   MetaProfile (BY, tiered alpha):
     Parus major: m=0.91 (COMMON) → 0.97 × 0.919 = 0.89
     Parus caeruleus: m=0.72 (COMMON) → 0.77 × 0.748 = 0.58
     Sitta europaea: m=0.65 (COMMON) → 0.60 × 0.685 = 0.41

   → Top-10 detections (≥ 0.1)

4. DetectionAggregator:
   Фильтрует NON_BIRD_LABELS
   Добавляет scores в окна:
     Parus major: [0, 0.85, 0.89, 0, ...] → 2 chunks ≥ 0.10 → CONFIRMED
     Parus caeruleus: [0, 0.58, ...] → пока 1 chunk → NOT CONFIRMED
     Sitta europaea: [0.35, 0.41, ...] → 2 chunks → CONFIRMED

5. filterDetections():
   Parus major: conf 0.89 ≥ 0.75 → ANCHOR (family: Paridae)
   Parus caeruleus: conf 0.58, family Paridae → SUPPRESSED (family has anchor)
   Sitta europaea: conf 0.41, aggregator-confirmed, family Sittidae (no anchor) → PASS
     confidence = avg-top-3 from aggregator

6. UI: [Большая синица 89%, Поползень 38%]
```

---

## 12. Ограничения

- **Близкие виды** — модель может присваивать высокий confidence нескольким близкородственным видам (напр. Горихвостка-чернушка 91% vs Сибирская горихвостка 85%). Family dedup решает это частично.
- **MetaProfile vs GPS** — если выбран неправильный регион в настройках, MetaProfile может подавлять реально присутствующие виды. GPS-метод точнее, но MetaProfile доступен без разрешения на геолокацию.
- **Bandpass и BirdNET** — bandpass-фильтр изменяет спектральный состав сигнала. HP cutoff 80 Гц — компромисс: убирает 50/60 Гц гул, но сохраняет фундаментальные частоты низкоголосых птиц.
- **Goertzel = точечная оценка** — спектральная проверка оценивает энергию на 4 конкретных частотах, а не по всему спектру. Сигналы, не совпадающие с контрольными частотами, проходят проверку.
- **50% overlap** — каждый момент аудио анализируется дважды (в двух соседних chunk-ах). Это обеспечивает лучший recall, но увеличивает нагрузку в 2×.
- **Buffer DROP_OLDEST** — при медленном inference свежий chunk дропает предыдущий. На слабых устройствах могут быть пропуски.

---

## 13. Тестирование

### Unit-тесты

| Файл | Покрытие |
|------|----------|
| `BandpassFilterTest` | АЧХ: passband (1/5 кГц), attenuation (50 Гц, 20 кГц), edge (80 Гц, 15 кГц, 150 Гц), тишина |
| `AudioChunkProcessorTest` | Silence skip, clipping skip, spectral reject (100 Гц), pigeon pass (120+240+480 Гц), owl pass (350 Гц), bird-freq pass (3 кГц), normalization, output stats |
| `DetectionAggregatorTest` | Confirmation logic, sliding window expiry, avg-top-3, max (file mode), non-bird filtering, multi-species, reset, null chunks, adaptive thresholds |
| `BirdNetV24ClassifierUnitTest` | buildDetections: threshold/topK filtering, sorting, empty input |

### Instrumented benchmark

**Файл:** `androidTest/ml/BirdNetBenchmarkTest.kt`

Прогоняет 16-минутный аудиофайл с 44 аннотированными видами через параллельный pipeline с AudioChunkProcessor. Результат: 100% recall на 44 видах.
