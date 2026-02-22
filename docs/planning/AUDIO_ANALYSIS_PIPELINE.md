# Audio Analysis Pipeline

Документация реального pipeline распознавания птиц по аудио.

## Общая схема

```
Микрофон (48 kHz, mono, PCM16)
    │
    ▼
AudioRecorder — непрерывный захват, chunking
    │
    │  FloatArray[144000] каждые ~1.5 сек
    ▼
AudioChunkProcessor — пре-фильтрация + bandpass + нормализация
    │
    │  FloatArray[144000] или null (skip)
    ▼
BirdNetV24Classifier — TFLite inference + sigmoid + meta-model
    │
    │  List<BirdDetection> (вид, confidence 0..1)
    ▼
DetectionAggregator — sliding window, подтверждение, фильтрация не-птиц
    │
    │  List<AggregatedDetection> (вид, confidence, кол-во подтверждений)
    ▼
LiveDetectionViewModel — маппинг в UI-состояние
    │
    │  List<DetectedBirdUi> (вид, confidence %, время)
    ▼
LiveDetectionScreen — лента обнаруженных видов
```

## 1. Захват аудио — AudioRecorder

**Файл:** `service/AudioRecorder.kt`

| Параметр | Значение |
|----------|----------|
| AudioSource | VOICE_RECOGNITION (повышенный gain, без шумоподавления) |
| Sample rate | 48 000 Hz |
| Формат | PCM 16-bit, mono |
| Chunk | 144 000 сэмплов = 3 секунды |
| Hop | 72 000 сэмплов = 1.5 секунды (50% overlap) |

### Как работает chunking

AudioRecorder читает аудио блоками по 4 800 сэмплов (100 мс) и копирует в аккумулятор. Когда аккумулятор заполняется (144 000 сэмплов = 3 секунды), chunk emit-ится во Flow.

После emit-а вторая половина аккумулятора копируется в начало — это даёт **50% перекрытие** между соседними chunk-ами:

```
Время: 0s        1.5s       3s        4.5s       6s
       |──chunk 1──|
                   |──chunk 2──|
                               |──chunk 3──|
```

Каждый chunk — независимая копия `FloatArray`, нормализованная в `[-1, 1]` (`short / 32768`).

### Почему VOICE_RECOGNITION

`AudioSource.MIC` на многих устройствах применяет аппаратное шумоподавление и AGC, которые могут подавить тихие звуки птиц как «шум». `VOICE_RECOGNITION` отключает эти обработки и обеспечивает более высокий gain.

## 2. Пре-обработка — AudioChunkProcessor

**Файл:** `ml/AudioChunkProcessor.kt`

Stateless-процессор, применяемый до ML-инференса. Фильтрует тишину, клиппинг и не-птичий шум, применяет bandpass-фильтр и нормализует уровень. Возвращает обработанный chunk или `null` (пропуск).

### 2.1. Silence check

```
RMS < 0.005 → пропуск (тишина)
```

RMS (Root Mean Square) — среднеквадратичная амплитуда. При RMS < 0.005 (~-46 dBFS) chunk содержит только фоновый шум без полезного сигнала.

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
- Если `lowRatio > 0.80` или `highRatio > 0.80` → пропуск (>80% энергии вне диапазона птиц)
- Bird-low и bird-mid считаются «птичьей» энергией и не вызывают пропуск

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
peak 0.001–0.5 → gain = 0.5 / peak, каждый сэмпл *= gain (clamp к [-1, 1])
peak > 0.5     → без изменений
```

Компенсирует тихие записи, приводя пиковую амплитуду к 0.5. Громкие сигналы (peak > 0.5) не трогаем — избыточное усиление может привести к клиппингу.

### Параметры AudioChunkProcessor

| Параметр | Значение | Константа |
|----------|----------|-----------|
| Порог тишины (RMS) | 0.005 | `SILENCE_RMS_THRESHOLD` |
| Порог клиппинга (peak) | 0.99 | `CLIPPING_PEAK_THRESHOLD` |
| Порог клиппинга (RMS) | 0.3 | `CLIPPING_RMS_THRESHOLD` |
| Порог спектрального отклонения | 80% | `SPECTRAL_REJECT_RATIO` |
| HP cutoff | 80 Гц | `LOW_CUTOFF` |
| LP cutoff | 15 000 Гц | `HIGH_CUTOFF` |
| Целевой peak | 0.5 | `NORM_TARGET` |
| Пост-фильтр тишина | 0.001 | `POST_FILTER_SILENCE_THRESHOLD` |

### Эффект пре-обработки (benchmark 44 вида, 639 chunk-ов)

| Метрика | Без процессора | С процессором |
|---------|---------------|---------------|
| Найдено видов | 43/44 (97.7%) | 44/44 (100%) |
| Пропущено chunk-ов | 0 | 74 (11%) |
| Ложных видов (≥0.5) | 22 | 19 |
| Время инференса | ~2:12 | ~2:34 |

Процессор пропускает ~11% chunk-ов (тишина, шум), при этом повышая recall до 100%.

## 3. Классификация — BirdNetV24Classifier

**Файл:** `ml/BirdNetV24Classifier.kt`

### Вход

- `FloatArray[144 000]` — 3 секунды обработанного audio, `[-1, 1]` (после AudioChunkProcessor)
- Опционально: `LocationMeta` (GPS + неделя года) для мета-модели

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

#### 3.3. Meta-model (опционально)

Если передан `LocationMeta` (GPS-координаты + неделя года), мета-модель фильтрует виды по географической и временной вероятности:

```
[latitude, longitude, weekOfYear] → meta-model.tflite → FloatArray[6522]
```

Scores audio-модели умножаются на scores мета-модели (element-wise). Это подавляет виды, невозможные в данной местности и сезоне, и усиливает вероятные.

**В текущей реализации GPS не передаётся — мета-модель не используется.**

#### 3.4. Фильтрация

Виды с confidence ≥ 0.1 возвращаются из классификатора (до 10 лучших — `topK`). Низкий порог 0.1 сохраняет кандидатов для агрегации.

### Выход

```kotlin
List<BirdDetection>(
    scientificName = "Parus major",
    commonName = "Большая синица",
    confidence = 0.987,  // после sigmoid
    labelIndex = 3847,
)
```

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

### 4.2. Два режима агрегации

| Параметр | Live Detection | Анализ файла |
|----------|---------------|--------------|
| Размер окна | 8 chunk-ов | Int.MAX (все chunk-и) |
| Подтверждение | ≥ 2 chunk-а в окне | ≥ 2 chunk-а всего |
| Confidence | Среднее top-3 | Max из всех chunk-ов |
| Factory | `forLiveDetection()` | `forFileAnalysis()` |

### 4.3. Sliding window (live detection)

Для каждого вида хранится `ArrayDeque<Float>` — последние N confidence-значений.

```
Window = 8 chunks (~12 секунд с 50% overlap)
```

При каждом новом chunk-е:
1. Для обнаруженных видов — добавляется confidence
2. Для ранее отслеживаемых, но не обнаруженных — добавляется 0.0
3. Если `deque.size > windowSize` — старейшее значение удаляется

### 4.4. Подтверждение

Вид считается подтверждённым, если **≥ 2 из N** значений в окне превышают порог вида (по умолчанию 0.5). Одиночный chunk с высоким confidence — не достаточно для подтверждения. Это предотвращает ложные срабатывания.

### 4.5. Confidence calculation

**Live mode — avg-top-3:**
Из всех значений в окне берутся 3 наибольших, их среднее = итоговый confidence.
Пример: окно `[0.9, 0.8, 0.7, 0.3, 0.2, 0, 0, 0]` → top-3: `[0.9, 0.8, 0.7]` → confidence = 0.8.

**File mode — max:**
Итоговый confidence = максимальный confidence из всех chunk-ов файла.

### 4.6. Адаптивные пороги

Для отдельных видов можно задать порог выше/ниже дефолтного:

```kotlin
aggregator.setThresholdOverride("Parus major", 0.8f)
```

### 4.7. Выход

```kotlin
List<AggregatedDetection>(
    scientificName = "Parus major",
    commonName = "Большая синица",
    confidence = 0.85,       // avg-top-3 или max
    confirmedChunks = 5,     // сколько chunk-ов подтвердили вид
)
```

Список отсортирован по confidence descending.

## 5. Labels

**Файл:** `assets/birdnet/v24/labels/ru.txt` (6 522 строки)

Формат: `Scientific Name_Русское название`

```
Acanthis flammea_Обыкновенная чечётка
Parus major_Большая синица
...
```

Включает не только птиц, но и другие звуки:
- `Engine` — двигатель
- `Fireworks` — фейерверк
- `Gun` — выстрел
- `Human vocal` — голос человека
- `Apis mellifera` — медоносная пчела

Эти классы помогают модели не путать техногенные/природные шумы с птицами. `DetectionAggregator` отфильтровывает их из результатов.

## 6. Пороги

В pipeline используется несколько порогов на разных этапах:

| Этап | Порог | Значение | Назначение |
|------|-------|----------|------------|
| AudioChunkProcessor | Silence RMS | 0.005 | Пропуск тишины |
| AudioChunkProcessor | Clipping peak/RMS | 0.99 / 0.3 | Пропуск насыщенного сигнала |
| AudioChunkProcessor | Spectral reject | 80% | Пропуск не-птичьего шума |
| AudioChunkProcessor | Post-filter silence | 0.001 | Пропуск после bandpass |
| BirdNetV24Classifier | Classifier threshold | 0.1 | Отсечение шума модели |
| DetectionAggregator | Confirmation threshold | 0.5 | Минимальный confidence для подтверждения |
| DetectionAggregator | Confirmation count | ≥ 2 | Минимум chunk-ов для подтверждения |

## 7. Характеристики производительности

| Метрика | Значение |
|---------|----------|
| Первый chunk | ~3 сек после старта записи |
| Последующие chunk-и | каждые ~1.5 сек (50% overlap) |
| Пре-обработка (AudioChunkProcessor) | ~3-5 мс / chunk |
| Inference на chunk (BirdNetV24Classifier) | ~100-300 мс (зависит от устройства) |
| Задержка до первого результата | ~3.5 сек |
| Потребление RAM | модели ~55 MB (audio 25 MB + meta 29 MB) |
| Размер labels | ~200 KB |
| Пропуск chunk-ов (типичный) | ~10-20% (тишина, шум) |

## 8. Ограничения

- **Нет Foreground Service** — при уходе в фон запись может быть остановлена системой
- **Нет GPS** — мета-модель не используется, confidence ниже чем с геолокацией
- **Близкие виды** — модель может присваивать высокий confidence нескольким близкородственным видам (напр. Горихвостка-чернушка 91% vs Сибирская горихвостка 85%) — это нормально для акустически похожих видов
- **Bandpass и BirdNET** — bandpass-фильтр изменяет спектральный состав сигнала. HP cutoff 80 Гц выбран как компромисс: убирает 50/60 Гц гул, но сохраняет фундаментальные частоты низкоголосых птиц (голуби ~120 Гц, совы ~300 Гц)
- **Goertzel = точечная оценка** — спектральная проверка оценивает энергию на 4 конкретных частотах, а не по всему спектру. Сигналы, не совпадающие с контрольными частотами, проходят проверку по fallback (totalEnergy < threshold)

## 9. Тестирование

### Unit-тесты

| Файл | Покрытие |
|------|----------|
| `BandpassFilterTest` | АЧХ: passband (1/5 кГц), attenuation (50 Гц, 20 кГц), edge (80 Гц, 15 кГц, 150 Гц), тишина |
| `AudioChunkProcessorTest` | Silence skip, clipping skip, spectral reject (100 Гц), pigeon pass (120+240+480 Гц), owl pass (350 Гц), bird-freq pass (3 кГц), normalization, output stats |
| `DetectionAggregatorTest` | Confirmation logic, sliding window expiry, avg-top-3, max (file mode), non-bird filtering, multi-species, reset, null chunks, adaptive thresholds |
| `BirdNetV24ClassifierUnitTest` | buildDetections: threshold/topK filtering, sorting, empty input |

### Instrumented benchmark

**Файл:** `androidTest/ml/BirdNetBenchmarkTest.kt`

Прогоняет 16-минутный аудиофайл с 44 аннотированными видами через параллельный pipeline с AudioChunkProcessor. Shared infrastructure (config, data classes, logging, parsing, matching, reporting) вынесена в `BenchmarkTestInfra.kt`.

#### `benchmark_sample1_withProcessor_parallel` — параллельный pipeline

```
Producer (IO):  decodeChunked → AudioChunkProcessor → chunksChannel
Workers (×N):   chunksChannel → BirdNetV24Classifier → resultsChannel
Collector:      resultsChannel → allDetections
```

N воркеров (по умолчанию 2), каждый со своим экземпляром `BirdNetV24Classifier` (2 TFLite-потока на воркер). Перед pipeline — warmup-проход для JIT-компиляции TFLite. Тишина и шум пропускаются AudioChunkProcessor. Результат: 100% recall на 44 видах.

**Что делает каждый классификатор** (`BirdNetV24Classifier`):

| Шаг | Что происходит | Где |
|-----|----------------|-----|
| 1. Audio model | `FloatArray[144000]` → TFLite (CNN) → `FloatArray[6521]` logits | `audioInterpreter.run()` |
| 2. Sigmoid | logit → probability: `1 / (1 + exp(-x))` | `BirdNetV24Classifier.kt:48` |
| 3. Meta model | `[lat, lon, week]` → `FloatArray[6521]` × scores (опционально, GPS не передаётся в benchmark) | `metaInterpreter.run()` |
| 4. Фильтрация | виды с confidence ≥ 0.1, top-10 по убыванию | `buildDetections()` |

Каждый воркер делает **все 4 шага** для своего chunk-а независимо.

**Потокобезопасность:**

| Объект | Sharing | Почему |
|--------|---------|--------|
| `MappedByteBuffer` (audio/meta модели) | Shared между всеми N | READ_ONLY mmap |
| `labels: List<Pair>` | Shared | Immutable |
| `audioInterpreter` / `metaInterpreter` | По одному на воркер | Mutable state внутри TFLite |
| `AudioChunkProcessor` | Только Producer | Bottleneck в inference, не в preprocessing |
| `totalChunks` / `skippedChunks` | `AtomicInteger` | Producer пишет из IO-корутины |
