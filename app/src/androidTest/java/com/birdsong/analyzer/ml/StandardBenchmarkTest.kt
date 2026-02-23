package com.birdsong.analyzer.ml

import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileFilter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Standard Benchmark: прогоняет аудиофайлы (по одному виду на файл)
 * через полный ML-пайплайн и сравнивает результат с ожидаемым видом из имени файла.
 *
 * Оптимизации:
 *   - Pipeline внутри файла: decode+preprocess (IO) || classify (Default) через Channel
 *   - Coroutines + Semaphore: эффективное управление параллелизмом без блокировки потоков
 *   - Динамический WORKER_COUNT на основе доступных CPU-ядер
 *
 * Перед запуском:
 *   adb push standard/ /data/local/tmp/standard/
 *
 * Запуск из Android Studio: правый клик → Run 'benchmark_standard'
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class StandardBenchmarkTest {

    /** Воркер: собственный classifier + processor (не потокобезопасны). */
    private data class Worker(
        val id: Int,
        val classifier: BirdNetV24Classifier,
        val processor: AudioChunkProcessor,
    )

    private lateinit var workers: List<Worker>
    private lateinit var workerPool: ArrayBlockingQueue<Worker>
    private lateinit var workerSemaphore: Semaphore
    private lateinit var labels: List<Pair<String, String>>
    private lateinit var ruLabelMap: Map<String, String>
    private lateinit var sciNameSet: Set<String>
    private var modelLoadTimeMs: Long = 0

    /** Прерывает MediaCodec/MediaExtractor из колбэка decodeChunked. */
    private class StopDecodingException(val reason: String) : RuntimeException(reason)

    // ── Data class ──

    data class FileResult(
        val filename: String,
        val expectedSpecies: String,
        val expectedRussianName: String,
        val detected: Boolean,
        val confidence: Float,
        val detectedSpecies: String,
        val detectedRussianName: String,
        val inferenceTimeMs: Long,
        val chunkCount: Int,
        val skippedChunks: Int,
        val speciesInLabels: Boolean,
        val stopReason: String,   // "confirmed" | "wrong_species" | "max_chunks" | "error"
    )

    // ── Taxonomy synonyms: filename → BirdNET label ──

    private val taxonomySynonyms = mapOf(
        "Anas querquedula" to "Spatula querquedula",
        "Anas strepera" to "Mareca strepera",
        "Aquila clanga" to "Clanga clanga",
        "Aquila pomarina" to "Clanga pomarina",
        "Carduelis chloris" to "Chloris chloris",
        "Coloeus dauuricus" to "Corvus dauuricus",
        "Coloeus monedula" to "Corvus monedula",
        "Dendrocopos medius" to "Dendrocoptes medius",
        "Grus virgo" to "Anthropoides virgo",
        "Porzana parva" to "Zapornia parva",
        "Porzana pusilla" to "Zapornia pusilla",
        "Spilopelia senegalensis" to "Streptopelia senegalensis",
        "Tachymarptis melba" to "Apus melba",
        "Tetrao tetrix" to "Lyrurus tetrix",
    )

    // ── Species absent from BirdNET V2.4 ──

    private val missingFromModel = setOf(
        "Accipiter brevipes", "Branta ruficollis", "Buteo rufinus",
        "Curruca cantillans", "Falco biarmicus", "Falco cherrug",
        "Glareola nordmanni", "Gypaetus barbatus", "Haliaeetus leucoryphus",
        "Ichthyaetus ichthyaetus", "Microcarbo pygmeus", "Otis tarda",
        "Pelecanus crispus", "Pelecanus onocrotalus", "Phalacrocorax aristotelis",
        "Polysticta stelleri", "Prunella atrogularis", "Vanellus gregarius",
    )

    // ── Setup / Teardown ──

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val start = System.currentTimeMillis()

        // Labels загружаем один раз (immutable, thread-safe)
        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/en_us.txt"
        labels = context.assets.open(labelsPath).use { LabelParser.load(it) }

        val ruLabelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        val ruLabels = context.assets.open(ruLabelsPath).use { LabelParser.load(it) }
        ruLabelMap = ruLabels.associate { (sci, ru) -> sci to ru }
        sciNameSet = labels.map { it.first.lowercase() }.toSet()

        log("CPU: ${Runtime.getRuntime().availableProcessors()} ядер, " +
            "воркеров: $WORKER_COUNT, TFLite потоков: $TFLITE_THREADS/interpreter")

        // Модели загружаем один раз — один MappedByteBuffer shared across all workers.
        // TFLite создаёт свою копию весов при Interpreter(buffer), buffer только для чтения.
        val audioModel = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        val metaModel = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)

        workers = (1..WORKER_COUNT).map { id ->
            Worker(
                id = id,
                classifier = BirdNetV24Classifier(
                    audioModel, metaModel, labels,
                    confidenceThreshold = 0.1f,
                    tfliteThreads = TFLITE_THREADS,
                ),
                processor = AudioChunkProcessor(),
            )
        }
        workerPool = ArrayBlockingQueue(WORKER_COUNT)
        workers.forEach { workerPool.put(it) }
        workerSemaphore = Semaphore(WORKER_COUNT)

        // TFLite warmup: первый инференс всегда медленнее из-за JIT-компиляции ops.
        // Прогреваем все классификаторы параллельно — сокращает setUp с N×warmup до 1×warmup.
        log("Warmup: прогрев $WORKER_COUNT классификаторов параллельно...")
        val warmupChunk = FloatArray(BirdClassifier.SAMPLES_PER_CHUNK)
        runBlocking {
            workers.map { worker ->
                async(Dispatchers.Default) {
                    val ms = System.currentTimeMillis()
                    worker.classifier.classify(warmupChunk)
                    log("  [W${worker.id}] прогрет: ${System.currentTimeMillis() - ms} мс")
                }
            }.awaitAll()
        }

        modelLoadTimeMs = System.currentTimeMillis() - start
        log("Модель загружена + прогрета: ${labels.size} видов, $WORKER_COUNT воркеров, $modelLoadTimeMs мс")
    }

    @After
    fun tearDown() {
        workers.forEach { it.classifier.close() }
    }

    // ── Lookup ──

    private fun russianName(scientificName: String): String =
        ruLabelMap[scientificName] ?: ""

    // ── Main benchmark ──

    @Test
    fun benchmark_standard() = runBlocking {
        val testStartMs = System.currentTimeMillis()
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        val standardDir = File(STANDARD_DIR)
        check(standardDir.exists() && standardDir.isDirectory) {
            "Директория $STANDARD_DIR не найдена. Используйте: adb push standard/ $STANDARD_DIR/"
        }

        val mp3Filter = FileFilter { file -> file.extension.equals("mp3", ignoreCase = true) }
        val files = standardDir.listFiles(mp3Filter)?.sortedBy { it.name }
            ?: error("Нет MP3 файлов в $STANDARD_DIR")

        log("═══ STANDARD BENCHMARK: ${files.size} файлов, $WORKER_COUNT потоков (pipeline) ═══")
        log("Директория: $STANDARD_DIR")
        log("")

        val completed = AtomicInteger(0)
        val detected = AtomicInteger(0)
        val total = files.size

        // Параллельная обработка файлов через coroutines + Semaphore
        val results = files.map { file ->
            async(Dispatchers.Default) {
                workerSemaphore.withPermit {
                    val worker = workerPool.poll()!!
                    try {
                        val result = processFile(appContext, file, worker)

                        val done = completed.incrementAndGet()
                        if (result.detected) detected.incrementAndGet()

                        val mark = when {
                            !result.speciesInLabels && result.expectedSpecies in missingFromModel -> "НЕТ В МОД"
                            result.detected -> "ДА %.3f".format(result.confidence)
                            else -> "НЕТ -> ${result.detectedSpecies}"
                        }
                        log("[${done}/${total}] ${file.name} => $mark " +
                            "(${result.chunkCount}ч, ${result.inferenceTimeMs}мс, ${result.stopReason})")

                        if (done % 50 == 0) {
                            val elapsed = System.currentTimeMillis() - testStartMs
                            val avg = elapsed / done
                            val eta = avg * (total - done)
                            log("--- ПРОГРЕСС: $done/$total, найдено: ${detected.get()}, " +
                                "прошло: ${formatDuration(elapsed)}, осталось ~${formatDuration(eta)} ---")
                        }

                        result
                    } finally {
                        workerPool.offer(worker)
                    }
                }
            }
        }.awaitAll()

        val totalTestMs = System.currentTimeMillis() - testStartMs

        printResultsTable(results)
        printThresholdAnalysis(results)
        printSummary(results, totalTestMs)
    }

    // ── Process single file (pipelined: decode || classify) ──

    /**
     * Pipeline-обработка файла:
     * - Producer (Dispatchers.IO): decode + preprocess → Channel
     * - Consumer (Dispatchers.Default): classify из Channel
     *
     * Decode chunk N+1 перекрывается с classify chunk N.
     */
    private suspend fun processFile(
        context: android.content.Context,
        file: File,
        worker: Worker,
    ): FileResult {
        val rawSpecies = extractSpecies(file.name)
        val mappedSpecies = taxonomySynonyms[rawSpecies] ?: rawSpecies
        val inLabels = sciNameSet.contains(mappedSpecies.lowercase())
        val expectedRu = ruLabelMap[mappedSpecies] ?: ruLabelMap[rawSpecies] ?: ""

        val startMs = System.currentTimeMillis()
        val speciesMaxConf = mutableMapOf<String, Float>()

        // Shared state (written by producer, read after join via happens-before)
        var totalChunks = 0
        var skippedChunks = 0
        var producerError: Exception? = null
        val stopSignal = AtomicBoolean(false)

        val channel = Channel<FloatArray>(PIPELINE_BUFFER)
        var stopReason = "max_chunks"

        try {
            coroutineScope {
                // Producer: decode + preprocess на IO dispatcher
                val producer = launch(Dispatchers.IO) {
                    var localTotal = 0
                    var localSkipped = 0
                    try {
                        AudioFileDecoder.decodeChunked(
                            context,
                            Uri.fromFile(file),
                        ) { _, _, chunk ->
                            if (stopSignal.get()) throw StopDecodingException("early_stop")
                            localTotal++
                            if (localTotal > MAX_CHUNKS_PER_FILE) throw StopDecodingException("max_chunks")
                            if (System.currentTimeMillis() - startMs > FILE_TIMEOUT_MS) {
                                throw StopDecodingException("timeout")
                            }

                            val processed = worker.processor.process(chunk)
                            if (processed == null) {
                                localSkipped++
                                return@decodeChunked
                            }

                            try {
                                channel.trySendBlocking(processed.samples).getOrThrow()
                            } catch (_: Exception) {
                                throw StopDecodingException("send_failed")
                            }
                        }
                    } catch (_: StopDecodingException) {
                        // Ожидаемое прерывание
                    } catch (e: Exception) {
                        producerError = e
                    } finally {
                        totalChunks = localTotal
                        skippedChunks = localSkipped
                        channel.close()
                    }
                }

                // Consumer: classify на Default dispatcher (через withContext внутри classify)
                var chunksClassified = 0
                for (samples in channel) {
                    val detections = worker.classifier.classify(samples)
                    chunksClassified++
                    for (d in detections) {
                        val cur = speciesMaxConf[d.scientificName] ?: 0f
                        if (d.confidence > cur) speciesMaxConf[d.scientificName] = d.confidence
                    }

                    // Стоп: ожидаемый вид подтверждён
                    if (speciesMaxConf.any { (sp, conf) ->
                            conf >= EARLY_STOP_CONFIDENCE && matchesName(mappedSpecies, sp)
                        }) {
                        stopReason = "confirmed"
                        stopSignal.set(true)
                        channel.cancel()
                        break
                    }

                    // Стоп: другой вид уверенно найден, ожидаемый не появлялся совсем.
                    // Гарды: минимум MIN_CHUNKS_BEFORE_WRONG_STOP чанков обработано +
                    //        ожидаемый ниже EXPECTED_ABSENT_THRESHOLD (нет даже слабого сигнала).
                    if (chunksClassified >= MIN_CHUNKS_BEFORE_WRONG_STOP) {
                        val expectedMaxConf = speciesMaxConf.entries
                            .filter { (sp, _) -> matchesName(mappedSpecies, sp) }
                            .maxOfOrNull { it.value } ?: 0f
                        val hasWrongSpecies = speciesMaxConf.any { (sp, conf) ->
                            conf >= WRONG_SPECIES_CONFIDENCE
                                && !matchesName(mappedSpecies, sp)
                                && BirdClassifier.NON_BIRD_LABELS.none { nb ->
                                    sp.equals(nb, ignoreCase = true)
                                }
                        }
                        if (hasWrongSpecies && expectedMaxConf < EXPECTED_ABSENT_THRESHOLD) {
                            stopReason = "wrong_species"
                            stopSignal.set(true)
                            channel.cancel()
                            break
                        }
                    }
                }

                // coroutineScope ждёт завершения producer
            }
        } catch (_: kotlinx.coroutines.CancellationException) {
            // channel.cancel() может пробросить CancellationException через coroutineScope
        } catch (e: Exception) {
            return FileResult(
                filename = file.name, expectedSpecies = rawSpecies,
                expectedRussianName = expectedRu, detected = false,
                confidence = 0f, detectedSpecies = "ОШИБКА: ${e.message?.take(60)}",
                detectedRussianName = "",
                inferenceTimeMs = System.currentTimeMillis() - startMs,
                chunkCount = totalChunks, skippedChunks = skippedChunks,
                speciesInLabels = inLabels,
                stopReason = "error",
            )
        }

        val inferenceTimeMs = System.currentTimeMillis() - startMs

        // Ошибка от producer
        if (producerError != null) {
            return FileResult(
                filename = file.name, expectedSpecies = rawSpecies,
                expectedRussianName = expectedRu, detected = false,
                confidence = 0f,
                detectedSpecies = "ОШИБКА: ${producerError!!.message?.take(60)}",
                detectedRussianName = "",
                inferenceTimeMs = inferenceTimeMs,
                chunkCount = totalChunks, skippedChunks = skippedChunks,
                speciesInLabels = inLabels,
                stopReason = "error",
            )
        }

        val ranked = speciesMaxConf
            .filter { (sp, _) ->
                BirdClassifier.NON_BIRD_LABELS.none { nb -> sp.equals(nb, ignoreCase = true) }
            }
            .entries.sortedByDescending { it.value }
            .map { it.key to it.value }

        val expectedMatch = ranked.find { (sp, _) -> matchesName(mappedSpecies, sp) }
        val top = ranked.firstOrNull()

        return if (expectedMatch != null) {
            FileResult(
                filename = file.name, expectedSpecies = rawSpecies,
                expectedRussianName = expectedRu, detected = true,
                confidence = expectedMatch.second,
                detectedSpecies = expectedMatch.first,
                detectedRussianName = russianName(expectedMatch.first),
                inferenceTimeMs = inferenceTimeMs,
                chunkCount = totalChunks, skippedChunks = skippedChunks,
                speciesInLabels = inLabels,
                stopReason = stopReason,
            )
        } else {
            FileResult(
                filename = file.name, expectedSpecies = rawSpecies,
                expectedRussianName = expectedRu, detected = false,
                confidence = top?.second ?: 0f,
                detectedSpecies = top?.first ?: "—",
                detectedRussianName = if (top != null) russianName(top.first) else "",
                inferenceTimeMs = inferenceTimeMs,
                chunkCount = totalChunks, skippedChunks = skippedChunks,
                speciesInLabels = inLabels,
                stopReason = stopReason,
            )
        }
    }

    // ── Output: Results table ──

    private fun printResultsTable(results: List<FileResult>) {
        val sep = "═".repeat(200)
        val thinSep = "─".repeat(200)
        log("")
        log(sep)
        log("РЕЗУЛЬТАТЫ STANDARD BENCHMARK")
        log(sep)
        log("")

        log(String.format(
            "%-4s│ %-55s│ %-35s│ %-10s│ %-8s│ %-40s│ %-6s│ %-5s│ %-8s",
            "#", " Файл", " Ожидаемый вид (RU)", " Результат", " Уверен.",
            " Обнаруженный вид / Ошибка", " Чанки", " Проп.", " Время",
        ))
        log(thinSep)

        for ((index, r) in results.withIndex()) {
            val fileStr = r.filename.take(53)
            val expectedStr = "${r.expectedSpecies} (${r.expectedRussianName})".take(33)

            val resultStr: String
            val confStr: String
            val detectedStr: String

            if (!r.speciesInLabels && r.expectedSpecies in missingFromModel) {
                resultStr = "НЕТ В МОД"
                confStr = "—"
                detectedStr = if (r.detectedSpecies != "—") {
                    val ru = r.detectedRussianName
                    "${r.detectedSpecies}${if (ru.isNotBlank()) " ($ru)" else ""}".take(38)
                } else "—"
            } else if (r.detected) {
                resultStr = "ДА"
                confStr = "%.3f".format(r.confidence)
                detectedStr = "—"
            } else {
                resultStr = "НЕТ"
                confStr = "%.3f".format(r.confidence)
                detectedStr = if (r.detectedSpecies != "—") {
                    val ru = r.detectedRussianName
                    "${r.detectedSpecies}${if (ru.isNotBlank()) " ($ru)" else ""}".take(38)
                } else "—"
            }

            log(String.format(
                "%-4d│ %-55s│ %-35s│ %-10s│ %-8s│ %-40s│ %-6s│ %-5s│ %-8s",
                index + 1, fileStr, expectedStr, resultStr, confStr,
                detectedStr, "${r.chunkCount}", "${r.skippedChunks}",
                "${r.inferenceTimeMs}мс",
            ))
        }
        log(thinSep)
        log("")
    }

    // ── Output: Threshold analysis ──

    private fun printThresholdAnalysis(results: List<FileResult>) {
        val testable = results.filter { it.speciesInLabels || it.expectedSpecies !in missingFromModel }

        log("ТОЧНОСТЬ ПРИ РАЗНЫХ ПОРОГАХ УВЕРЕННОСТИ:")
        log("  (Только виды, присутствующие в модели: ${testable.size} файлов)")
        log("")
        log(String.format(
            "  %-12s │ %-10s │ %-12s │ %-10s",
            "Порог", "Найдено", "Не найдено", "Recall",
        ))
        log("  " + "─".repeat(52))

        for (threshold in listOf(0.01f, 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.6f, 0.8f, 0.9f)) {
            val matched = testable.count { it.detected && it.confidence >= threshold }
            val missed = testable.size - matched
            log(String.format(
                "  %-12s │ %-10d │ %-12d │ %-10s",
                ">= %.2f".format(threshold), matched, missed,
                "%.1f%%".format(matched * 100.0 / testable.size),
            ))
        }
        log("")
    }

    // ── Output: Summary ──

    private fun printSummary(results: List<FileResult>, totalTestMs: Long) {
        val sep  = "═".repeat(80)
        val thin = "─".repeat(80)

        // ── Базовые счётчики ──

        val total        = results.size
        val inModelList  = results.filter { it.speciesInLabels || it.expectedSpecies !in missingFromModel }
        val inModel      = inModelList.size
        val notInModelList = results.filter { !it.speciesInLabels && it.expectedSpecies in missingFromModel }
        val notInModel   = notInModelList.size
        val errors       = results.count { it.detectedSpecies.startsWith("ОШИБКА") }

        val detectedInModel = inModelList.count { it.detected }

        // Определилось что-то, но не тот вид (только среди файлов с видами в модели)
        val wrongDetectionList = inModelList.filter {
            !it.detected && it.detectedSpecies != "—" && !it.detectedSpecies.startsWith("ОШИБКА")
        }
        val wrongDetection = wrongDetectionList.size

        // Вообще ничего не определилось (только среди файлов с видами в модели, без ошибок)
        val trulyMissedList = inModelList.filter {
            !it.detected && it.detectedSpecies == "—"
        }
        val trulyMissed = trulyMissedList.size

        log("")
        log(sep)
        log("ИТОГО")
        log(sep)
        log("")

        // ── Файлы и виды ──

        log("  ФАЙЛЫ И ВИДЫ:")
        log("  $thin")
        log("  Всего файлов:                   $total")
        log("  Видов ЕСТЬ в модели BirdNET:    $inModel  (по ним считается Recall)")
        log("  Видов НЕТ в модели BirdNET:     $notInModel  " +
            "(уникальных: ${notInModelList.map { it.expectedSpecies }.toSet().size})")
        if (errors > 0) log("  Ошибки декодирования:           $errors")
        log("  Воркеры / TFLite потоков:       $WORKER_COUNT / $TFLITE_THREADS")
        log("")

        // ── Результаты определения ──

        log("  ОПРЕДЕЛЕНИЕ  (виды в модели, $inModel файлов):")
        log("  $thin")
        log("  Верно определено:               $detectedInModel / $inModel" +
            "  (%.1f%%)".format(detectedInModel * 100.0 / inModel.coerceAtLeast(1)))
        log("  Определилось НЕ ТО:             $wrongDetection" +
            "  (%.1f%%)".format(wrongDetection * 100.0 / inModel.coerceAtLeast(1)))
        log("  Ничего не определилось:         $trulyMissed" +
            "  (%.1f%%)".format(trulyMissed * 100.0 / inModel.coerceAtLeast(1)))
        log("")

        // ── Причины остановки ──

        log("  ПРИЧИНЫ ОСТАНОВКИ:")
        log("  $thin")
        val stopReasonOrder = listOf("confirmed", "wrong_species", "max_chunks", "error")
        val stopReasonLabels = mapOf(
            "confirmed"     to "Ожидаемый вид найден (confirmed)  ",
            "wrong_species" to "Найден другой вид  (wrong_species)",
            "max_chunks"    to "Лимит чанков ($MAX_CHUNKS_PER_FILE)       (max_chunks)  ",
            "error"         to "Ошибка декодирования   (error)      ",
        )
        val bySR = results.groupBy { it.stopReason }
        for (reason in stopReasonOrder) {
            val count = bySR[reason]?.size ?: 0
            if (count > 0 || reason in listOf("confirmed", "wrong_species", "max_chunks")) {
                log("  ${stopReasonLabels[reason] ?: reason.padEnd(40)}  $count файлов")
            }
        }
        log("")

        // ── По типу записи ──

        val byType = results.groupBy { extractType(it.filename) }
        log("  РЕЗУЛЬТАТЫ ПО ТИПУ ЗАПИСИ:")
        log("  $thin")
        log(String.format("  %-20s │ %-8s │ %-8s │ %-10s", "Тип", "Всего", "Найдено", "Recall"))
        log("  $thin")
        for ((type, typeResults) in byType.entries.sortedByDescending { it.value.size }) {
            val t = typeResults.size
            val d = typeResults.count { it.detected }
            log(String.format("  %-20s │ %-8d │ %-8d │ %-10s",
                type, t, d, "%.1f%%".format(d * 100.0 / t)))
        }
        log("")

        // ── Производительность ──

        val totalInferenceMs = results.sumOf { it.inferenceTimeMs }
        val avgFileMs = if (results.isNotEmpty()) totalInferenceMs / results.size else 0L
        val totalChunks  = results.sumOf { it.chunkCount }
        val totalSkipped = results.sumOf { it.skippedChunks }

        log("  ПРОИЗВОДИТЕЛЬНОСТЬ:")
        log("  $thin")
        log("  CPU ядер:                       ${Runtime.getRuntime().availableProcessors()}")
        log("  TFLite потоков/interpreter:     $TFLITE_THREADS")
        log("  Загрузка + warmup:              ${modelLoadTimeMs} мс")
        log("  Wall time:                      ${formatDuration(totalTestMs)}")
        log("  Суммарный инференс (все потоки):${formatDuration(totalInferenceMs)}")
        log("  Среднее время на файл:          $avgFileMs мс")
        log("  Всего чанков / пропущено:       $totalChunks / $totalSkipped" +
            " (%.1f%%)".format(if (totalChunks > 0) totalSkipped * 100.0 / totalChunks else 0.0))
        log("")

        // ── Уверенность (корректно определённые) ──

        val detectedResults = results.filter { it.detected }
        if (detectedResults.isNotEmpty()) {
            val vals = detectedResults.map { it.confidence }.sorted()
            log("  УВЕРЕННОСТЬ (верно определённые, $detectedInModel файлов):")
            log("  $thin")
            log("  Минимум:   %.3f".format(vals.first()))
            log("  Медиана:   %.3f".format(vals[vals.size / 2]))
            log("  Среднее:   %.3f".format(vals.average()))
            log("  Максимум:  %.3f".format(vals.last()))
            log("")
        }

        // ── Определилось НЕ ТО ──

        if (wrongDetectionList.isNotEmpty()) {
            log("  ОПРЕДЕЛИЛОСЬ НЕ ТО ($wrongDetection файлов)  — модель нашла вид, но не тот:")
            log("  $thin")
            for (r in wrongDetectionList.sortedBy { it.filename }) {
                val expRu = if (r.expectedRussianName.isNotBlank()) " (${r.expectedRussianName})" else ""
                val detRu = if (r.detectedRussianName.isNotBlank()) " (${r.detectedRussianName})" else ""
                log("  * ${r.filename}")
                log("      ожидалось: ${r.expectedSpecies}$expRu")
                log("      найдено:   ${r.detectedSpecies}$detRu  [%.3f]".format(r.confidence))
            }
            log("")
        }

        // ── Ничего не определилось ──

        if (trulyMissedList.isNotEmpty()) {
            log("  НИЧЕГО НЕ ОПРЕДЕЛИЛОСЬ ($trulyMissed файлов):")
            log("  $thin")
            for (r in trulyMissedList.take(50).sortedBy { it.filename }) {
                val ruStr = if (r.expectedRussianName.isNotBlank()) " (${r.expectedRussianName})" else ""
                log("  * ${r.filename}: ${r.expectedSpecies}$ruStr")
            }
            if (trulyMissedList.size > 50) log("  ... и ещё ${trulyMissedList.size - 50} файлов")
            log("")
        }

        // ── Виды не в модели ──

        if (notInModelList.isNotEmpty()) {
            val uniqueSpecies = notInModelList
                .groupBy { it.expectedSpecies }
                .entries.sortedBy { it.key }
            log("  ВИДЫ НЕ В МОДЕЛИ BirdNET (${uniqueSpecies.size} уникальных видов, $notInModel файлов):")
            log("  $thin")
            for ((species, files) in uniqueSpecies) {
                val ruStr = files.first().expectedRussianName.let { if (it.isNotBlank()) " ($it)" else "" }
                log("  * $species$ruStr  — ${files.size} файл(ов)")
            }
            log("")
        }

        log(sep)
        log("  Общее время теста: ${formatDuration(totalTestMs)}")
        log(sep)
        log("")
    }

    // ── Filename parsing ──

    private fun extractSpecies(filename: String): String {
        val words = filename.removeSuffix(".mp3").split(" ")
        return if (words.size >= 2) "${words[0]} ${words[1]}" else words[0]
    }

    private fun extractType(filename: String): String {
        val withoutNum = filename.removeSuffix(".mp3").replace(Regex("\\s+\\d{4}$"), "")
        val words = withoutNum.split(" ")
        val raw = if (words.size > 2) words.drop(2).joinToString(" ") else ""
        return when {
            raw.contains("song") -> "song"
            raw.contains("calls") -> "calls"
            raw == "juv" -> "juv"
            raw == "drum" -> "drum"
            raw == "piping" -> "piping"
            raw == "sounds" -> "sounds"
            raw.isBlank() -> "unknown"
            else -> raw
        }
    }

    // ── Matching ──

    private fun matchesName(expected: String, detected: String): Boolean {
        val exp = expected.lowercase().trim()
        val det = detected.lowercase().trim()
        return exp == det || det.startsWith(exp) || exp.startsWith(det)
    }

    // ── Utilities ──

    private fun formatDuration(ms: Long): String {
        val s = ms / 1000
        return "%d мин %02d сек (%d мс)".format(s / 60, s % 60, ms)
    }

    private fun loadModel(context: android.content.Context, assetPath: String): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    companion object {
        private const val TAG = "StandardBenchmark"
        private const val STANDARD_DIR = "/data/local/tmp/standard"
        private const val FILE_TIMEOUT_MS = 25_000L       // страховочный таймаут (MAX_CHUNKS × ~3с ≈ 18с)
        private const val EARLY_STOP_CONFIDENCE = 0.80f   // ожидаемый вид найден → стоп
        private const val WRONG_SPECIES_CONFIDENCE = 0.89f // чужой вид найден уверенно → стоп
        private const val MIN_CHUNKS_BEFORE_WRONG_STOP = 2 // гард: ожидаемый не успел появиться в 1-м чанке
        private const val EXPECTED_ABSENT_THRESHOLD = 0.1f // гард: у ожидаемого нет даже слабого сигнала
        private const val MAX_CHUNKS_PER_FILE = 6          // было 10: 6 × 3с = 18с достаточно
        private const val PIPELINE_BUFFER = 8  // было 2: малый буфер вызывал частые stall-ы producer-а
        private const val TFLITE_THREADS = 1  // было 2: 1 поток на interpreter → больше воркеров при том же CPU

        // Динамический подсчёт воркеров: cores / tflite_threads, [2..8]
        // TFLITE_THREADS=1: для 8-ядерного устройства → 8 воркеров вместо 4 (было cores/2, cap 6)
        private val WORKER_COUNT = (Runtime.getRuntime().availableProcessors() / TFLITE_THREADS)
            .coerceIn(2, 8)

        private fun log(message: String) { Log.i(TAG, message) }
    }
}
