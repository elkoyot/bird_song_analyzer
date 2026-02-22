# Sample Benchmark — BirdNetBenchmarkTest

Бенчмарк на одном аннотированном аудиофайле (~16 минут, 44 вида птиц). Сравнивает результат ML-пайплайна с эталонными аннотациями.

**Файл теста:** `app/src/androidTest/java/com/birdsong/analyzer/ml/BirdNetBenchmarkTest.kt`

## Назначение

Измеряет **recall** (полноту) ML-пайплайна — какую долю аннотированных видов модель обнаруживает. Два варианта:

| Тест | Описание |
|------|----------|
| `benchmark_sample1` | Без AudioChunkProcessor — чистый inference |
| `benchmark_sample1_withProcessor` | С AudioChunkProcessor — полный pipeline |

Сравнение двух тестов показывает влияние пре-обработки (bandpass, нормализация, фильтрация шума) на качество распознавания.

## Подготовка к запуску

### Входные данные

Файлы в `androidTest/assets/benchmark/1/`:

| Файл | Описание |
|------|----------|
| `1.mp3` | ~16 минут аудио с голосами 44 видов |
| `1.txt` | Аннотации: время + название вида |

### Требования

- **Реальное Android-устройство** (эмулятор не поддерживает аудио-декодирование через MediaCodec)
- BirdNET модели в `app/src/main/assets/birdnet/v24/` (audio-model-fp16.tflite, meta-model.tflite, labels)

### Запуск

Android Studio → правый клик на тесте → Run (на подключённом устройстве).

### Фильтрация логов

```bash
adb logcat -s BENCH:I
```

Тег `BENCH` — все логи бенчмарка с уровнем INFO.

## Технический процесс

```
                    benchmark_sample1              benchmark_sample1_withProcessor
                    ─────────────────              ───────────────────────────────

1. Загрузка модели (audio-model-fp16.tflite + meta-model.tflite + labels)
                    │                              │
2. Парсинг ground truth из 1.txt
                    │                              │
3. Копирование 1.mp3 из assets во temp-файл
                    │                              │
4. AudioFileDecoder.decodeChunked()
   │  chunk = 144 000 samples (3 сек)
   │  hop = 72 000 samples (50% overlap)
   │  Декодирование через MediaExtractor + MediaCodec
                    │                              │
5. —                │              AudioChunkProcessor.process(chunk)
                    │              │  silence/clipping/spectral → null (skip)
                    │              │  bandpass → normalization → Result
                    │                              │
6. BirdNetV24Classifier.classify(chunk)
   │  audio-model → logits[6522]
   │  sigmoid → probabilities[6522]
   │  filter confidence >= 0.1, topK = 10
                    │                              │
7. Сбор TimedDetection(chunkIndex, startTimeSec, endTimeSec, detection)
                    │                              │
8. matchDetections() — сопоставление с ground truth
                    │                              │
9. Вывод результатов
```

### Шаг 1. Загрузка модели

Модели загружаются через `MappedByteBuffer` (memory-mapped файл). Два TFLite interpreter-а:
- **audio-model-fp16.tflite** (~25 MB) — основная модель, принимает аудио, выдаёт logits
- **meta-model.tflite** (~29 MB) — фильтр по геолокации (не используется в бенчмарке, `location = null`)

Labels загружаются из `en_us.txt` (6 522 вида) и `ru.txt` (русские названия).

Порог уверенности классификатора: **0.1** (низкий, чтобы не терять детекции).

### Шаг 2. Парсинг ground truth

Файл `1.txt` — текстовые аннотации, по одному виду на строку:

```
0:05  Зелёная пересмешка / Icterine Warbler (Hippolais icterina)
1:15  Большая синица / Great Tit (Parus major)
...
```

Формат строки: `MM:SS  Русское название / English Name (Scientific name)`

Парсинг: regex `(\d+:\d+)\s+(.+?)\s*/\s*(.+?)\s*\((.+?)\)\s*$`

### Шаг 3. Копирование аудио

MP3 копируется из `androidTest/assets/` во временный файл в `cacheDir` приложения, так как `MediaExtractor` не может работать с потоком assets напрямую.

### Шаг 4. Декодирование

`AudioFileDecoder.decodeChunked()` декодирует MP3 через `MediaExtractor` + `MediaCodec` в PCM float32:

- Формат выхода: int16 PCM (принудительно через `KEY_PCM_ENCODING`)
- Конвертация в float: `short / 32768`
- Downmix в моно (если стерео): среднее каналов
- Ресемплинг (если нужно): линейная интерполяция до 48 kHz

Chunk-и выдаются через sliding window:
```
Chunk 0: [0 .. 144000)      → 0:00 – 0:03
Chunk 1: [72000 .. 216000)   → 0:01 – 0:04
Chunk 2: [144000 .. 288000)  → 0:03 – 0:06
...
```

### Шаг 5. Пре-обработка (только withProcessor)

`AudioChunkProcessor.process(chunk)` применяет pipeline фильтрации:

1. **Silence check:** RMS < 0.005 → skip
2. **Clipping check:** peak > 0.99 AND rms > 0.3 → skip
3. **Spectral check:** Goertzel на 4 частотах, если > 80% энергии вне птичьего диапазона → skip
4. **Bandpass filter:** Butterworth biquad, 80 Гц – 15 кГц
5. **Post-filter silence:** peak < 0.001 → skip
6. **Peak normalization:** peak → 0.5

Подробное описание: [AUDIO_ANALYSIS_PIPELINE.md](../planning/AUDIO_ANALYSIS_PIPELINE.md), раздел 2.

### Шаг 6. Классификация

`BirdNetV24Classifier.classify(chunk)`:

1. Audio-model inference: `FloatArray[144000]` → `FloatArray[6522]` (logits)
2. Sigmoid: `1 / (1 + exp(-logit))` → вероятности [0..1]
3. Фильтрация: confidence >= 0.1, top-10 видов

Подробное описание: [AUDIO_ANALYSIS_PIPELINE.md](../planning/AUDIO_ANALYSIS_PIPELINE.md), раздел 3.

### Шаг 7. Сбор детекций

Каждая детекция оборачивается в `TimedDetection`:

```kotlin
TimedDetection(
    chunkIndex = 42,           // порядковый номер chunk-а
    startTimeSec = 63.0f,      // начало chunk-а в аудиофайле (секунды)
    endTimeSec = 66.0f,        // конец chunk-а
    detection = BirdDetection( // результат классификации
        scientificName = "Parus major",
        commonName = "Great Tit",
        confidence = 0.987f,
        labelIndex = 3847,
    )
)
```

### Шаг 8. Сопоставление с ground truth

Функция `matchDetections()` — для каждой эталонной записи ищет детекции того же вида во временном окне.

**Временное окно:**

```
[gt.timeSeconds - 3с .. gt.timeSeconds + 20с]
```

- **-3 секунды:** chunk overlap — детекция может начаться раньше аннотации из-за перекрытия chunk-ов
- **+20 секунд:** птица может петь 10-20 секунд после начала аннотации

**Сопоставление имён:**

Функция `matchesScientificName()` сравнивает научные имена с учётом:
- Точное совпадение (case-insensitive)
- Совпадение по префиксу (`"Columba"` matches `"Columba livia"`)
- Taxonomy synonyms (см. ниже)

**Лучшая детекция:** из всех совпавших выбирается с максимальным confidence.

## Описание вывода

### Таблица сравнения (printComparisonTable)

```
═══════...══════════
РЕЗУЛЬТАТЫ БЕНЧМАРКА [БЕЗ AudioChunkProcessor]: sample/1
═══════...══════════

#   │ Время │ Ожидаемый вид                                 │ Обнаруженный вид                              │ Уверенн. │ Время обн. │ Результат
────...──────────
1   │ 0:05  │ Hippolais icterina (Зелёная пересмешка)       │ Hippolais icterina (Зелёная пересмешка)       │ 0.923    │ 0:04       │ НАЙДЕН
2   │ 1:15  │ Parus major (Большая синица)                   │ —                                             │ —        │ —          │ ПРОПУСК
...
```

| Столбец | Описание |
|---------|----------|
| # | Порядковый номер в эталоне |
| Время | Время в аннотации (MM:SS) |
| Ожидаемый вид | Научное название + русское (из аннотации) |
| Обнаруженный вид | Научное название + русское (из модели) |
| Уверенн. | Максимальный confidence из всех совпавших детекций |
| Время обн. | Время начала chunk-а с лучшей детекцией |
| Результат | `НАЙДЕН` или `ПРОПУСК` |

**Итого:**
- Количество найденных и пропущенных видов с процентами
- **Ложные срабатывания** — виды с confidence >= 0.5, отсутствующие в эталоне

### Анализ порогов уверенности (printThresholdAnalysis)

```
ТОЧНОСТЬ ПРИ РАЗНЫХ ПОРОГАХ УВЕРЕННОСТИ:

  Порог        │ Найдено   │ Не найдено  │ Точность
  ──────────────────────────────────────────────────
  ≥ 0.1        │ 44        │ 0           │ 100.0%
  ≥ 0.2        │ 43        │ 1           │ 97.7%
  ≥ 0.3        │ 43        │ 1           │ 97.7%
  ≥ 0.5        │ 42        │ 2           │ 95.5%
  ≥ 0.6        │ 41        │ 3           │ 93.2%
  ≥ 0.8        │ 38        │ 6           │ 86.4%
```

Показывает, сколько видов из эталона определяются при каждом пороге. Помогает выбрать оптимальный порог для production: чем выше порог — тем меньше ложных, но больше пропущенных.

### Хронология обнаружений (printTimeline)

```
ХРОНОЛОГИЯ ОБНАРУЖЕНИЙ (топ видов в 5-секундных окнах, уверенность >= 0.3):
──────...──────────
  0:00-0:05: Hippolais icterina/Зелёная пересмешка(0.92) ◄◄ ЭТАЛОН: Hippolais icterina (Зелёная пересмешка)
  0:05-0:10: Hippolais icterina/Зелёная пересмешка(0.88)
  ...
```

Временная шкала с 5-секундным шагом. Для каждого окна — top-3 вида с confidence. Маркер `◄◄ ЭТАЛОН:` показывает, какие виды ожидались по аннотации в этом окне.

## Taxonomy synonyms

Научные имена видов меняются при ревизиях таксономии. BirdNET использует свою версию, которая может отличаться от аннотаций.

| Ground truth | Метка BirdNET | Причина |
|---|---|---|
| `Coloeus monedula` | `Corvus monedula` | Галка: перенесена из Coloeus обратно в Corvus |
| `Columba` | `Columba livia` | Голубь: род → полное биномиальное имя |

Функция `matchesScientificName()` также использует сопоставление по префиксу, что покрывает случаи неполного имени.

## Формат ground truth файла

Текстовый файл с аннотациями, по одной записи на строку:

```
MM:SS  Русское название / English Name (Scientific name)
```

**Пример:**
```
0:05  Зелёная пересмешка / Icterine Warbler (Hippolais icterina)
0:27  Обыкновенная горихвостка / Common Redstart (Phoenicurus phoenicurus)
1:00  Зяблик / Eurasian Chaffinch (Fringilla coelebs)
```

**Правила:**
- Время: `MM:SS` (минуты:секунды от начала файла)
- Разделитель названий: ` / `
- Научное имя: в круглых скобках
- Пустые строки игнорируются

## Алгоритм сопоставления

```
Для каждой записи ground truth (gt):
  1. windowStart = gt.timeSeconds - 3
  2. windowEnd = gt.timeSeconds + 20
  3. Найти все детекции, где:
     - startTimeSec >= windowStart AND startTimeSec <= windowEnd
     - matchesScientificName(gt.scientificName, detection.scientificName)
  4. bestDetection = детекция с максимальным confidence
  5. Результат: НАЙДЕН (если bestDetection != null) или ПРОПУСК
```

**matchesScientificName:**
```
gt_normalized = taxonomySynonyms[gt] ?: gt    // подмена синонимов
return gt_normalized == ml                     // точное совпадение
    || ml.startsWith(gt_normalized)            // gt — префикс ml
    || gt_normalized.startsWith(ml)            // ml — префикс gt
```

## Глоссарий

| Термин | Описание |
|--------|----------|
| **chunk** | Фрагмент аудио фиксированного размера (144 000 сэмплов = 3 секунды при 48 kHz), подаваемый в классификатор |
| **hop** | Шаг между началами соседних chunk-ов (72 000 сэмплов = 1.5 секунды). hop < chunk → перекрытие |
| **overlap** | Пересечение соседних chunk-ов: `1 - hop/chunk = 50%`. Каждый момент аудио попадает в 2 chunk-а |
| **confidence** | Вероятность (0..1) принадлежности chunk-а виду, получена из sigmoid-а от logit-а |
| **ground truth** | Эталонные аннотации: время и вид птицы, размеченные экспертом |
| **recall** | Доля эталонных видов, обнаруженных моделью. recall = найдено / всего_в_эталоне |
| **sigmoid** | Функция `1 / (1 + exp(-x))`, переводит logit (сырое значение модели) в вероятность [0..1] |
| **logits** | Сырые выходные значения нейросети до sigmoid-а. Могут быть отрицательными. Не вероятности |
| **bandpass** | Полосовой фильтр, пропускающий частоты в заданном диапазоне (80 Гц – 15 кГц) и подавляющий остальные |
| **Goertzel** | Алгоритм вычисления энергии на одной частоте за O(N) — точечный аналог FFT. Используется для быстрой спектральной проверки |
| **early stop** | Не используется в sample benchmark (в отличие от standard benchmark) — обрабатываются все chunk-и файла |
| **TimedDetection** | Детекция с привязкой к времени: содержит BirdDetection + координаты chunk-а в аудиофайле |
| **MatchResult** | Результат сопоставления одной записи ground truth со всеми детекциями: лучшая детекция + все совпадения |

## Параметры и константы

### BirdClassifier (общие)

| Константа | Значение | Описание |
|-----------|----------|----------|
| `SAMPLE_RATE` | 48 000 Hz | Частота дискретизации |
| `CHUNK_DURATION_SECONDS` | 3 | Длительность chunk-а в секундах |
| `SAMPLES_PER_CHUNK` | 144 000 | Размер chunk-а в сэмплах (48000 * 3) |
| `NON_BIRD_LABELS` | 9 меток | Engine, Environmental, Fireworks, Gun, Human vocal, Noise, Power tools, Siren, Apis mellifera |

### BirdNetV24Classifier

| Константа | Значение | Описание |
|-----------|----------|----------|
| `DEFAULT_THRESHOLD` | 0.1 | Минимальный confidence для выдачи детекции (в бенчмарке передаётся как `confidenceThreshold`) |
| `DEFAULT_TOP_K` | 10 | Максимум видов на один chunk |
| `DEFAULT_NUM_THREADS` | 4 | Потоки TFLite (используется в sample benchmark) |
| `AUDIO_MODEL_PATH` | `birdnet/v24/audio-model-fp16.tflite` | Путь к audio-модели |
| `META_MODEL_PATH` | `birdnet/v24/meta-model.tflite` | Путь к meta-модели |

### AudioChunkProcessor

| Константа | Значение | Описание |
|-----------|----------|----------|
| `SILENCE_RMS_THRESHOLD` | 0.005 | RMS ниже этого → тишина, chunk пропускается |
| `CLIPPING_PEAK_THRESHOLD` | 0.99 | Peak выше этого + высокий RMS → клиппинг |
| `CLIPPING_RMS_THRESHOLD` | 0.3 | RMS-порог для клиппинга (совместно с peak) |
| `SPECTRAL_REJECT_RATIO` | 0.80 | Если > 80% энергии в low/high → не-птичий шум |
| `LOW_CUTOFF` | 80 Гц | Частота среза high-pass |
| `HIGH_CUTOFF` | 15 000 Гц | Частота среза low-pass |
| `NORM_TARGET` | 0.5 | Целевой peak при нормализации |
| `POST_FILTER_SILENCE_THRESHOLD` | 0.001 | Peak после bandpass ниже этого → пропуск |

### AudioFileDecoder.decodeChunked()

| Параметр | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `chunkSize` | `SAMPLES_PER_CHUNK` (144 000) | Размер chunk-а |
| `hopSize` | `chunkSize / 2` (72 000) | Шаг между chunk-ами (50% overlap) |

### Временное окно сопоставления

| Параметр | Значение | Описание |
|----------|----------|----------|
| windowStart | `gt.time - 3 сек` | Начало окна поиска (overlap compensation) |
| windowEnd | `gt.time + 20 сек` | Конец окна (птица может петь до 20 сек) |

### Порог ложных срабатываний

| Параметр | Значение | Описание |
|----------|----------|----------|
| False positive threshold | 0.5 | Виды с confidence >= 0.5, не найденные в эталоне, считаются ложными |
