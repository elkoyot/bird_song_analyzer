# Standard Benchmark — StandardBenchmarkTest

Массовый бенчмарк на сотнях аудиофайлов (по одному виду на файл). Параллельная pipeline-обработка через coroutines.

**Файл теста:** `app/src/androidTest/java/com/birdsong/analyzer/ml/StandardBenchmarkTest.kt`

## Назначение

Измеряет **recall** на представительной выборке видов. Каждый файл содержит запись одного вида — имя вида извлекается из имени файла. Бенчмарк определяет, находит ли модель ожидаемый вид среди детекций.

| Тест | Описание |
|------|----------|
| `benchmark_standard` | Параллельная pipeline-обработка всех файлов из `/data/local/tmp/standard/` |

## Подготовка к запуску

### Загрузка данных на устройство

```bash
adb push standard/ /data/local/tmp/standard/
```

Директория `standard/` содержит MP3-файлы, по одному виду на файл.

### Формат имён файлов

```
Genus species type NNNN.mp3
```

| Компонент | Описание | Пример |
|-----------|----------|--------|
| `Genus species` | Научное биномиальное имя (первые 2 слова) | `Parus major` |
| `type` | Тип записи: song, calls, juv, drum и т.д. | `song` |
| `NNNN` | Числовой идентификатор (4 цифры) | `0001` |

**Примеры:**
```
Parus major song 0001.mp3
Turdus merula calls 0042.mp3
Dendrocopos major drum 0003.mp3
Bubo bubo juv 0001.mp3
```

**Извлечение вида:** берутся первые 2 слова имени файла (без `.mp3`).

**Извлечение типа:** всё после вида и до числового суффикса.

### Требования

- **Реальное Android-устройство**
- BirdNET модели в assets
- MP3-файлы в `/data/local/tmp/standard/` на устройстве

### Запуск

Android Studio → правый клик на тесте → Run.

### Фильтрация логов

```bash
adb logcat -s StandardBenchmark:I
```

Тег `StandardBenchmark` — все логи бенчмарка с уровнем INFO.

## Архитектура pipeline

```
                    ┌─────────────────────────────────────────────┐
                    │             benchmark_standard               │
                    │                                             │
                    │  setUp():                                   │
                    │    Labels (загрузка 1 раз, thread-safe)     │
                    │    Workers[N] (каждый: свой classifier +    │
                    │                        свой processor)      │
                    │    WorkerPool (ArrayBlockingQueue)           │
                    │    WorkerSemaphore (Semaphore[N])            │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │  files.map { file ->                        │
                    │    async(Dispatchers.Default) {             │
                    │      workerSemaphore.withPermit {           │
                    │        worker = workerPool.poll()           │
                    │        processFile(file, worker)            │
                    │      }                                      │
                    │    }                                         │
                    │  }.awaitAll()                                │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │  processFile() — pipeline для одного файла  │
                    │                                             │
                    │  ┌─Producer (Dispatchers.IO)──────────────┐ │
                    │  │ AudioFileDecoder.decodeChunked()       │ │
                    │  │   → AudioChunkProcessor.process()      │ │
                    │  │   → Channel.send(samples)              │ │
                    │  └───────────────┬────────────────────────┘ │
                    │                  │ Channel (buffer=2)        │
                    │  ┌───────────────▼────────────────────────┐ │
                    │  │ Consumer (Dispatchers.Default)         │ │
                    │  │   for (samples in channel):            │ │
                    │  │     classifier.classify(samples)       │ │
                    │  │     if confidence >= 0.80 → early stop │ │
                    │  └────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────┘
```

### Worker pool

Каждый **Worker** содержит:
- Свой `BirdNetV24Classifier` (TFLite interpreter не потокобезопасен)
- Свой `AudioChunkProcessor` (stateless, но создаётся для изоляции)

Воркеры хранятся в `ArrayBlockingQueue` — потокобезопасной очереди. Доступ контролируется `Semaphore`:
- `async` + `workerSemaphore.withPermit` — ограничивает параллелизм до N воркеров
- `workerPool.poll()` / `workerPool.offer()` — берёт/возвращает воркера

### Pipeline внутри файла

Для каждого файла создаётся `Channel<FloatArray>` (буфер 2 элемента):

- **Producer** (`Dispatchers.IO`): декодирует аудио, применяет AudioChunkProcessor, отправляет обработанные chunk-и в Channel
- **Consumer** (`Dispatchers.Default`): получает chunk-и, классифицирует через BirdNetV24Classifier

Producer декодирует chunk N+1, пока Consumer классифицирует chunk N — **decode и classify выполняются параллельно**.

### Early stop

Если confidence для ожидаемого вида достигает **0.80** — обработка файла прекращается:
1. `stopSignal.set(true)` — AtomicBoolean, проверяется Producer-ом
2. `channel.cancel()` — прерывает Consumer
3. Producer ловит `StopDecodingException` и завершается

### Защитные ограничения

| Ограничение | Значение | Описание |
|-------------|----------|----------|
| `MAX_CHUNKS_PER_FILE` | 10 | Максимум chunk-ов на файл (после 10 — stop) |
| `FILE_TIMEOUT_MS` | 15 000 мс | Таймаут на файл (15 секунд) |

## Технический процесс

### 1. setUp — создание воркеров

```
CPU cores = Runtime.getRuntime().availableProcessors()
WORKER_COUNT = (cores / TFLITE_THREADS).coerceIn(2, 6)
```

Для каждого воркера загружается своя копия модели (отдельный TFLite interpreter).

### 2. Для каждого файла — async + semaphore

```kotlin
files.map { file ->
    async(Dispatchers.Default) {
        workerSemaphore.withPermit {
            val worker = workerPool.poll()!!
            try {
                processFile(context, file, worker)
            } finally {
                workerPool.offer(worker)
            }
        }
    }
}.awaitAll()
```

Semaphore гарантирует, что одновременно обрабатывается не более `WORKER_COUNT` файлов.

### 3. Pipeline внутри файла

**Producer** (IO dispatcher):
1. `AudioFileDecoder.decodeChunked()` — декодирование MP3
2. Для каждого chunk-а:
   - Проверка `stopSignal` → `StopDecodingException`
   - Проверка `MAX_CHUNKS_PER_FILE` → `StopDecodingException`
   - Проверка таймаута → `StopDecodingException`
   - `AudioChunkProcessor.process(chunk)` → null (skip) или Result
   - `channel.send(processed.samples)`

**Consumer** (Default dispatcher):
1. `for (samples in channel)` — получает обработанные chunk-и
2. `classifier.classify(samples)` → List<BirdDetection>
3. Обновляет `speciesMaxConf` (максимальный confidence по каждому виду)
4. Проверяет early stop: если ожидаемый вид с confidence >= 0.80 — стоп

### 4. Формирование результата

После обработки всех chunk-ов:
1. Исключаются NON_BIRD_LABELS из детекций
2. Виды ранжируются по максимальному confidence
3. Ищется ожидаемый вид (из имени файла) с учётом taxonomy synonyms
4. Формируется `FileResult` с флагом `detected = true/false`

## Описание вывода

### Таблица результатов (printResultsTable)

```
═══════...═══════
РЕЗУЛЬТАТЫ STANDARD BENCHMARK
═══════...═══════

#   │ Файл                                                   │ Ожидаемый вид (RU)                │ Результат │ Уверен.│ Обнаруженный вид / Ошибка              │ Чанки│ Проп.│ Время
────...──────
1   │ Parus major song 0001.mp3                              │ Parus major (Большая синица)       │ ДА        │ 0.987  │ —                                      │ 3    │ 0    │ 1250мс
2   │ Branta ruficollis song 0001.mp3                        │ Branta ruficollis (Краснозобая...) │ НЕТ В МОД │ —      │ —                                      │ 5    │ 1    │ 2100мс
...
```

| Столбец | Описание |
|---------|----------|
| Файл | Имя MP3-файла |
| Ожидаемый вид (RU) | Научное + русское название (из имени файла) |
| Результат | `ДА` / `НЕТ` / `НЕТ В МОД` (вид отсутствует в модели) |
| Уверен. | Максимальный confidence для ожидаемого вида |
| Обнаруженный вид / Ошибка | При `НЕТ` — top-1 вид (что модель «увидела» вместо ожидаемого) |
| Чанки | Сколько chunk-ов было декодировано |
| Проп. | Сколько chunk-ов пропущено AudioChunkProcessor |
| Время | Wall time обработки файла (мс) |

### Анализ порогов (printThresholdAnalysis)

```
ТОЧНОСТЬ ПРИ РАЗНЫХ ПОРОГАХ УВЕРЕННОСТИ:
  (Только виды, присутствующие в модели: 250 файлов)

  Порог        │ Найдено    │ Не найдено   │ Recall
  ────────────────────────────────────────────────────
  >= 0.01      │ 240        │ 10           │ 96.0%
  >= 0.05      │ 238        │ 12           │ 95.2%
  >= 0.10      │ 235        │ 15           │ 94.0%
  ...
  >= 0.80      │ 200        │ 50           │ 80.0%
  >= 0.90      │ 180        │ 70           │ 72.0%
```

Виды, отсутствующие в модели (`missingFromModel`), **исключаются** из статистики порогов. Пороги проверяются: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9.

### Итого (printSummary)

Содержит несколько секций:

**Общая статистика:**
- Всего файлов, видов в модели, видов не в модели, ошибок декодирования
- Recall (все файлы), recall (только виды в модели)

**Результаты по типу записи:**
```
  Тип                  │ Всего    │ Найдено  │ Recall
  ──────────────────────────────────────────────────────
  song                 │ 150      │ 145      │ 96.7%
  calls                │ 80       │ 72       │ 90.0%
  drum                 │ 10       │ 10       │ 100.0%
  ...
```

**Производительность:**
- CPU ядер, TFLite потоков на interpreter, загрузка модели
- Wall time (общее время теста)
- Суммарный инференс (все потоки)
- Среднее время на файл
- Всего chunk-ов, пропущено процессором

**Уверенность (корректно определённые):**
- Минимум, медиана, среднее, максимум confidence

**Не определённые виды:**
- Список файлов с видами в модели, которые не были обнаружены
- Для каждого — top-1 детекция (что модель «увидела» вместо ожидаемого)

## Taxonomy synonyms

14 маппингов для видов, чьё научное имя в файлах отличается от метки BirdNET:

| Имя в файле | Метка BirdNET | Причина |
|---|---|---|
| `Anas querquedula` | `Spatula querquedula` | Чирок-трескунок: перенесён в Spatula |
| `Anas strepera` | `Mareca strepera` | Серая утка: перенесена в Mareca |
| `Aquila clanga` | `Clanga clanga` | Большой подорлик: перенесён в Clanga |
| `Aquila pomarina` | `Clanga pomarina` | Малый подорлик: перенесён в Clanga |
| `Carduelis chloris` | `Chloris chloris` | Зеленушка: перенесена в Chloris |
| `Coloeus dauuricus` | `Corvus dauuricus` | Даурская галка: Coloeus → Corvus |
| `Coloeus monedula` | `Corvus monedula` | Галка: Coloeus → Corvus |
| `Dendrocopos medius` | `Dendrocoptes medius` | Средний дятел: перенесён в Dendrocoptes |
| `Grus virgo` | `Anthropoides virgo` | Журавль-красавка: Grus → Anthropoides |
| `Porzana parva` | `Zapornia parva` | Малый погоныш: Porzana → Zapornia |
| `Porzana pusilla` | `Zapornia pusilla` | Погоныш-крошка: Porzana → Zapornia |
| `Spilopelia senegalensis` | `Streptopelia senegalensis` | Малая горлица: Spilopelia → Streptopelia |
| `Tachymarptis melba` | `Apus melba` | Белобрюхий стриж: Tachymarptis → Apus |
| `Tetrao tetrix` | `Lyrurus tetrix` | Тетерев: Tetrao → Lyrurus |

Маппинг применяется **до** поиска вида в результатах модели.

## Виды, отсутствующие в модели BirdNET V2.4

18 видов из набора файлов не имеют соответствующей метки в BirdNET V2.4:

| Вид | Русское название |
|-----|-----------------|
| Accipiter brevipes | Европейский тювик |
| Branta ruficollis | Краснозобая казарка |
| Buteo rufinus | Курганник |
| Curruca cantillans | Белоусая славка |
| Falco biarmicus | Средиземноморский сокол |
| Falco cherrug | Балобан |
| Glareola nordmanni | Луговая тиркушка |
| Gypaetus barbatus | Бородач |
| Haliaeetus leucoryphus | Орлан-долгохвост |
| Ichthyaetus ichthyaetus | Черноголовый хохотун |
| Microcarbo pygmeus | Малый баклан |
| Otis tarda | Дрофа |
| Pelecanus crispus | Кудрявый пеликан |
| Pelecanus onocrotalus | Розовый пеликан |
| Phalacrocorax aristotelis | Хохлатый баклан |
| Polysticta stelleri | Сибирская гага |
| Prunella atrogularis | Черногорлая завирушка |
| Vanellus gregarius | Кречётка |

Эти виды помечаются как `НЕТ В МОД` в таблице результатов и **исключаются** из статистики recall.

## Глоссарий

| Термин | Описание |
|--------|----------|
| **pipeline** | Конвейерная обработка: decode и classify выполняются параллельно через Channel |
| **Worker** | Экземпляр с собственным TFLite interpreter и AudioChunkProcessor. Не потокобезопасен — каждый файл обрабатывается одним воркером |
| **Semaphore** | Kotlin coroutine-примитив, ограничивающий параллелизм. `withPermit` блокирует корутину (не поток), пока permit не станет доступен |
| **Channel** | Kotlin coroutine-примитив для передачи данных между корутинами. `Channel<FloatArray>(2)` — буферизованный канал на 2 элемента |
| **early stop** | Прекращение обработки файла при достижении confidence >= 0.80 для ожидаемого вида. Экономит время на длинных записях |
| **wall time** | Реальное время «по часам» от начала до конца теста, включая параллельную обработку |
| **recall** | Доля файлов, в которых ожидаемый вид был обнаружен. `recall = detected / total_testable` |
| **StopDecodingException** | Исключение для прерывания `decodeChunked()` из callback-а (early stop, max chunks, timeout) |
| **ArrayBlockingQueue** | Java concurrent-коллекция для пула воркеров. Thread-safe без дополнительной синхронизации |
| **taxonomy synonyms** | Маппинг устаревших/альтернативных научных имён видов на метки BirdNET. Применяется до сопоставления |
| **matchesName** | Нечёткое сопоставление: точное совпадение или совпадение по префиксу (case-insensitive) |

## Параметры и константы

### Параллелизм

| Константа | Значение | Описание |
|-----------|----------|----------|
| `TFLITE_THREADS` | 2 | Потоки TFLite на каждый interpreter |
| `WORKER_COUNT` | `(CPU_cores / TFLITE_THREADS).coerceIn(2, 6)` | Количество воркеров (динамическое) |
| `PIPELINE_BUFFER` | 2 | Размер буфера Channel между decode и classify |

**Пример расчёта WORKER_COUNT:**
- 8 ядер → `8 / 2 = 4` воркера
- 4 ядра → `4 / 2 = 2` воркера
- 16 ядер → `16 / 2 = 8` → coerced to **6** воркеров

### Защитные ограничения

| Константа | Значение | Описание |
|-----------|----------|----------|
| `FILE_TIMEOUT_MS` | 15 000 мс | Максимальное время на файл |
| `MAX_CHUNKS_PER_FILE` | 10 | Максимум chunk-ов на файл |
| `EARLY_STOP_CONFIDENCE` | 0.80 | Порог для early stop |

### Классификатор

| Константа | Значение | Описание |
|-----------|----------|----------|
| `DEFAULT_THRESHOLD` | 0.1 | Минимальный confidence (передаётся как `confidenceThreshold` в конструктор) |
| `tfliteThreads` | 2 | Потоки TFLite (переопределяет DEFAULT_NUM_THREADS=4) |

### AudioChunkProcessor

Все параметры идентичны sample benchmark — см. [SAMPLE_BENCHMARK.md](SAMPLE_BENCHMARK.md#audiochunkprocessor).

### AudioFileDecoder

| Параметр | Значение | Описание |
|----------|----------|----------|
| `chunkSize` | 144 000 | Размер chunk-а (по умолчанию) |
| `hopSize` | 72 000 | Шаг (50% overlap, по умолчанию) |

### Путь к данным

| Константа | Значение | Описание |
|-----------|----------|----------|
| `STANDARD_DIR` | `/data/local/tmp/standard` | Путь к MP3-файлам на устройстве |

## Оптимизации производительности

### 1. Pipeline decode || classify

Декодирование и классификация выполняются параллельно через `Channel`:

```
Без pipeline:    [decode ch1][classify ch1][decode ch2][classify ch2]...
С pipeline:      [decode ch1][decode ch2  ][decode ch3  ]...
                             [classify ch1][classify ch2]...
```

Producer работает на `Dispatchers.IO` (блокирующее I/O — MediaCodec), Consumer — на `Dispatchers.Default` (CPU-интенсивная работа — TFLite). Буфер Channel = 2 позволяет Producer-у декодировать 2 chunk-а вперёд.

### 2. Dynamic WORKER_COUNT

```kotlin
WORKER_COUNT = (Runtime.getRuntime().availableProcessors() / TFLITE_THREADS).coerceIn(2, 6)
```

- Делим CPU на TFLITE_THREADS: каждый воркер использует 2 потока TFLite, суммарно `WORKER_COUNT * 2` потоков
- Минимум 2: даже на 2-ядерном устройстве — один файл на pipeline
- Максимум 6: ограничение по RAM (каждый воркер ~55 MB TFLite моделей)

### 3. Coroutines vs Executors

`Semaphore.withPermit` **суспендит корутину**, а не блокирует поток. Это позволяет запустить сотни `async` задач с фактическим параллелизмом = WORKER_COUNT, без создания сотен потоков.

### 4. TFLITE_THREADS = 2

В standard benchmark используется 2 потока TFLite (вместо 4 по умолчанию):
- Позволяет большему количеству воркеров работать одновременно
- Общая пропускная способность выше: `N * (inference / 2 threads)` > `(N/2) * (inference / 4 threads)` при N воркерах
- На мобильных CPU с big.LITTLE архитектурой 2 потока эффективнее: меньше overhead от синхронизации

### 5. Early stop

Если вид обнаружен с confidence >= 0.80, дальнейшая обработка файла прекращается. Для файлов длиннее 15 секунд (5+ chunk-ов) это значительная экономия.

Подробное описание ML-пайплайна: [AUDIO_ANALYSIS_PIPELINE.md](../planning/AUDIO_ANALYSIS_PIPELINE.md).
