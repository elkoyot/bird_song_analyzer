package com.birdsong.analyzer.ml

import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Benchmark-тест: прогоняет реальный аудиофайл с аннотированными видами птиц
 * через полный ML-пайплайн и сравнивает результаты с ground truth.
 *
 * Входные данные: androidTest/assets/benchmark/1/
 *   - 1.mp3 — ~16 минут аудио с голосами 44 видов птиц
 *   - 1.txt — аннотации: время + название вида (RU / EN (Scientific))
 *
 * Результат: таблица сравнения + статистика точности.
 *
 * Запуск: Android Studio → правый клик → Run.
 * Работает как на реальном устройстве, так и на виртуальном (эмуляторе),
 * поскольку тест использует аудиофайлы из assets, а не микрофон.
 *
 * ---
 *
 * ## Термины
 *
 * **Chunk (чанк)** — фиксированный фрагмент аудио длиной [BirdClassifier.CHUNK_DURATION_SECONDS]
 * секунд (обычно 3 с). BirdNET принимает на вход ровно столько — не больше и не меньше.
 * Аудиофайл нарезается последовательно без перекрытия через [AudioFileDecoder.decodeChunked].
 *
 * **Порог уверенности (confidence threshold)** — минимальная вероятность (0..1), при которой
 * обнаружение считается валидным. В бенчмарке используется 0.1 («принять всё с вероятностью
 * ≥ 10%»), чтобы собрать сырые данные и затем проанализировать точность при разных порогах.
 * В продакшн-режиме приложение использует порог 0.5 или выше.
 *
 * ## Pipeline декодирования
 *
 * 1. Аудиофайл из assets копируется во временный файл кэша — MediaExtractor требует URI на диске.
 * 2. [AudioFileDecoder.decodeChunked] декодирует MP3 через MediaExtractor + MediaCodec,
 *    нарезает PCM-поток на чанки по [BirdClassifier.CHUNK_DURATION_SECONDS] секунд
 *    и передаёт каждый чанк в callback в виде FloatArray (нормализованные сэмплы −1..1).
 * 3. (Опционально) [AudioChunkProcessor] применяет bandpass-фильтр (80 Гц–15 кГц)
 *    и пиковую нормализацию. Чанки с уровнем сигнала ниже порога RMS пропускаются —
 *    это убирает тишину и снижает количество ложных срабатываний.
 * 4. [BirdNetV24Classifier] запускает TFLite-инференс: аудиомодель выдаёт logits,
 *    к которым применяется sigmoid, затем мета-модель уточняет вероятности по всем видам.
 * 5. Обнаружения сопоставляются с эталонными аннотациями по естественному окну
 *    [GT_time - 3 с, next_GT_time]: каждый вид получает слот до следующей аннотации.
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class BirdNetBenchmarkTest {

    private lateinit var classifier: BirdNetV24Classifier
    private lateinit var labels: List<Pair<String, String>>
    private lateinit var ruLabelMap: Map<String, String> // scientificName → русское название
    private lateinit var audioModelBuffer: MappedByteBuffer
    private lateinit var metaModelBuffer: MappedByteBuffer
    private val audioChunkProcessor = AudioChunkProcessor()

    // ── Data classes ──

    data class GroundTruth(
        val index: Int,
        val timeSeconds: Int,
        val timeFormatted: String,
        val russianName: String,
        val englishName: String,
        val scientificName: String,
    )

    data class TimedDetection(
        val chunkIndex: Int,
        val startTimeSec: Float,
        val endTimeSec: Float,
        val detection: BirdDetection,
    )

    data class MatchResult(
        val groundTruth: GroundTruth,
        val bestDetection: TimedDetection?,
        val allMatches: List<TimedDetection>,
        val windowStartSec: Float,
        val windowEndSec: Float,
    )

    // ── Setup / Teardown ──

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        var t = logBegin("Загрузка аудиомодели", "путь: ${BirdNetV24Classifier.AUDIO_MODEL_PATH}")
        audioModelBuffer = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        logEnd("Загрузка аудиомодели", t)

        t = logBegin("Загрузка мета-модели", "путь: ${BirdNetV24Classifier.META_MODEL_PATH}")
        metaModelBuffer = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)
        logEnd("Загрузка мета-модели", t)

        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/en_us.txt"
        t = logBegin("Загрузка меток EN", "путь: $labelsPath")
        labels = context.assets.open(labelsPath).use { LabelParser.load(it) }
        logEnd("Загрузка меток EN", t, "${labels.size} видов")

        val ruLabelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        t = logBegin("Загрузка меток RU", "путь: $ruLabelsPath")
        val ruLabels = context.assets.open(ruLabelsPath).use { LabelParser.load(it) }
        ruLabelMap = ruLabels.associate { (sci, ru) -> sci to ru }
        logEnd("Загрузка меток RU", t, "${ruLabels.size} видов")

        t = logBegin("Инициализация классификатора", "BirdNetV24Classifier, порог уверенности: 0.1")
        classifier = BirdNetV24Classifier(audioModelBuffer, metaModelBuffer, labels, confidenceThreshold = 0.1f)
        logEnd("Инициализация классификатора", t)
    }

    @After
    fun tearDown() {
        classifier.close()
    }

    // ── Lookup: научное имя → русское название ──

    private fun russianName(scientificName: String): String =
        ruLabelMap[scientificName] ?: ""

    private fun formatSpeciesWithRu(scientificName: String, ruName: String = russianName(scientificName)): String {
        return if (ruName.isNotBlank()) "$scientificName ($ruName)" else scientificName
    }

    // ── Main benchmark test (без AudioChunkProcessor) ──

    @Test
    fun benchmark_sample1() {
        val testStartMs = System.currentTimeMillis()
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        log("═══ БЕНЧМАРК БЕЗ AudioChunkProcessor ═══")
        log("")

        // [1/5] Парсинг эталонных аннотаций
        var t = logBegin("Парсинг эталонных аннотаций", "файл: benchmark/1/1.txt")
        val groundTruth = testContext.assets.open("benchmark/1/1.txt").use { stream ->
            parseGroundTruth(stream.bufferedReader().readText())
        }
        logEnd("Парсинг эталонных аннотаций", t, "${groundTruth.size} видов")
        log("")

        val tempFile = File(appContext.cacheDir, "benchmark_1.mp3")
        try {
            // [2/5] Копирование аудиофайла
            t = logBegin("Копирование аудиофайла", "источник: benchmark/1/1.mp3 → ${tempFile.absolutePath}")
            testContext.assets.open("benchmark/1/1.mp3").use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            logEnd("Копирование аудиофайла", t, "${tempFile.length() / 1024} КБ")
            log("")

            // [3/5] Декодирование + инференс
            val allDetections = mutableListOf<TimedDetection>()
            var chunkCount = 0
            t = logBegin(
                "Декодирование + инференс BirdNET",
                "файл: ${tempFile.name}, chunk: ${BirdClassifier.CHUNK_DURATION_SECONDS} с, порог: 0.1",
            )

            AudioFileDecoder.decodeChunked(
                appContext,
                Uri.fromFile(tempFile),
            ) { chunkIndex, startTimeSec, chunk ->
                val endTimeSec = startTimeSec + BirdClassifier.CHUNK_DURATION_SECONDS
                val detections = runBlocking { classifier.classify(chunk) }
                for (d in detections) {
                    allDetections.add(TimedDetection(chunkIndex, startTimeSec, endTimeSec, d))
                }
                chunkCount++

                if (chunkCount % 50 == 0) {
                    log("  [ПРОГРЕСС] чанк $chunkCount, позиция ${formatTime(startTimeSec.toInt())}, " +
                        "обнаружений: ${allDetections.size}, прошло: ${formatElapsed(System.currentTimeMillis() - t)}")
                }
            }

            if (chunkCount == 0) {
                log("ОШИБКА: decodeChunked не вернул ни одного чанка.")
                return
            }

            val avgMs = (System.currentTimeMillis() - t) / chunkCount
            logEnd(
                "Декодирование + инференс BirdNET", t,
                "$chunkCount чанков, ${allDetections.size} обнаружений, среднее $avgMs мс/чанк",
            )
            log("")

            // [4/5] Сопоставление с эталоном
            val recordingEndSec = allDetections.maxOfOrNull { it.endTimeSec } ?: 0f
            t = logBegin(
                "Сопоставление обнаружений с эталоном",
                "${allDetections.size} обнаружений × ${groundTruth.size} эталонных записей, окно: [GT-3с, след.GT]",
            )
            val matchResults = matchDetections(groundTruth, allDetections, recordingEndSec)
            val hits = matchResults.count { it.bestDetection != null }
            logEnd("Сопоставление обнаружений с эталоном", t, "$hits совпадений из ${matchResults.size}")
            log("")

            // [5/5] Формирование отчёта
            t = logBegin("Формирование отчёта", "режим: БЕЗ AudioChunkProcessor")
            printComparisonTable("БЕЗ AudioChunkProcessor", groundTruth, matchResults, allDetections)
            printThresholdAnalysis(matchResults)
            printTimeline(allDetections, groundTruth)
            logEnd("Формирование отчёта", t)
            log("")

            log("═══ Общее время теста (без процессора): ${formatElapsed(System.currentTimeMillis() - testStartMs)} ═══")
            log("")
        } finally {
            tempFile.delete()
        }
    }

    // ── Benchmark with AudioChunkProcessor ──

    @Test
    fun benchmark_sample1_withProcessor() {
        val testStartMs = System.currentTimeMillis()
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        log("═══ БЕНЧМАРК С AudioChunkProcessor ═══")
        log("")

        // [1/5] Парсинг эталонных аннотаций
        var t = logBegin("Парсинг эталонных аннотаций", "файл: benchmark/1/1.txt")
        val groundTruth = testContext.assets.open("benchmark/1/1.txt").use { stream ->
            parseGroundTruth(stream.bufferedReader().readText())
        }
        logEnd("Парсинг эталонных аннотаций", t, "${groundTruth.size} видов")
        log("")

        val tempFile = File(appContext.cacheDir, "benchmark_1_proc.mp3")
        try {
            // [2/5] Копирование аудиофайла
            t = logBegin("Копирование аудиофайла", "источник: benchmark/1/1.mp3 → ${tempFile.absolutePath}")
            testContext.assets.open("benchmark/1/1.mp3").use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            logEnd("Копирование аудиофайла", t, "${tempFile.length() / 1024} КБ")
            log("")

            // [3/5] Декодирование + AudioChunkProcessor + инференс
            val allDetections = mutableListOf<TimedDetection>()
            var chunkCount = 0
            var skippedChunks = 0
            t = logBegin(
                "Декодирование + AudioChunkProcessor + инференс BirdNET",
                "файл: ${tempFile.name}, chunk: ${BirdClassifier.CHUNK_DURATION_SECONDS} с, порог: 0.1",
            )

            AudioFileDecoder.decodeChunked(
                appContext,
                Uri.fromFile(tempFile),
            ) { chunkIndex, startTimeSec, chunk ->
                val endTimeSec = startTimeSec + BirdClassifier.CHUNK_DURATION_SECONDS
                chunkCount++

                val processed = audioChunkProcessor.process(chunk)
                if (processed == null) {
                    skippedChunks++
                    return@decodeChunked
                }

                val detections = runBlocking { classifier.classify(processed.samples) }
                for (d in detections) {
                    allDetections.add(TimedDetection(chunkIndex, startTimeSec, endTimeSec, d))
                }

                if (chunkCount % 50 == 0) {
                    log("  [ПРОГРЕСС] чанк $chunkCount ($skippedChunks пропущено), " +
                        "позиция ${formatTime(startTimeSec.toInt())}, обнаружений: ${allDetections.size}, " +
                        "прошло: ${formatElapsed(System.currentTimeMillis() - t)}")
                }
            }

            val classifiedChunks = chunkCount - skippedChunks
            if (classifiedChunks == 0) {
                log("ОШИБКА: AudioChunkProcessor пропустил все чанки!")
                return
            }

            val skipPct = skippedChunks * 100 / chunkCount.coerceAtLeast(1)
            val avgMs = (System.currentTimeMillis() - t) / classifiedChunks
            logEnd(
                "Декодирование + AudioChunkProcessor + инференс BirdNET", t,
                "$chunkCount чанков ($skippedChunks пропущено = $skipPct%), $classifiedChunks классифицировано, " +
                    "${allDetections.size} обнаружений, среднее $avgMs мс/чанк",
            )
            log("")

            // [4/5] Сопоставление с эталоном
            val recordingEndSec = allDetections.maxOfOrNull { it.endTimeSec } ?: 0f
            t = logBegin(
                "Сопоставление обнаружений с эталоном",
                "${allDetections.size} обнаружений × ${groundTruth.size} эталонных записей, окно: [GT-3с, след.GT]",
            )
            val matchResults = matchDetections(groundTruth, allDetections, recordingEndSec)
            val hits = matchResults.count { it.bestDetection != null }
            logEnd("Сопоставление обнаружений с эталоном", t, "$hits совпадений из ${matchResults.size}")
            log("")

            // [5/5] Формирование отчёта
            t = logBegin("Формирование отчёта", "режим: С AudioChunkProcessor")
            printComparisonTable("С AudioChunkProcessor", groundTruth, matchResults, allDetections)
            printThresholdAnalysis(matchResults)
            printTimeline(allDetections, groundTruth)
            logEnd("Формирование отчёта", t)
            log("")

            log("═══ Общее время теста (с процессором): ${formatElapsed(System.currentTimeMillis() - testStartMs)} ═══")
            log("")
        } finally {
            tempFile.delete()
        }
    }

    // ── Benchmark with AudioChunkProcessor + parallel pipeline ──

    @Test
    fun benchmark_sample1_withProcessor_parallel() {
        val testStartMs = System.currentTimeMillis()
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        log("═══ БЕНЧМАРК С AudioChunkProcessor (ПАРАЛЛЕЛЬНЫЙ PIPELINE) ═══")
        log("")

        // [1/6] Парсинг эталонных аннотаций
        var t = logBegin("Парсинг эталонных аннотаций", "файл: benchmark/1/1.txt")
        val groundTruth = testContext.assets.open("benchmark/1/1.txt").use { stream ->
            parseGroundTruth(stream.bufferedReader().readText())
        }
        logEnd("Парсинг эталонных аннотаций", t, "${groundTruth.size} видов")
        log("")

        // [2/6] Создание 2 классификаторов (по 2 TFLite-потока каждый)
        t = logBegin("Создание параллельных классификаторов", "2 × BirdNetV24Classifier, tfliteThreads=2")
        val classifiers = (1..4).map { i ->
            BirdNetV24Classifier(
                audioModelBuffer, metaModelBuffer, labels,
                confidenceThreshold = 0.1f,
                tfliteThreads = 2,
            ).also { log("  Классификатор #$i создан") }
        }
        logEnd("Создание параллельных классификаторов", t)
        log("")

        val tempFile = File(appContext.cacheDir, "benchmark_1_par.mp3")
        try {
            // [3/6] Копирование аудиофайла
            t = logBegin("Копирование аудиофайла", "источник: benchmark/1/1.mp3 → ${tempFile.absolutePath}")
            testContext.assets.open("benchmark/1/1.mp3").use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            logEnd("Копирование аудиофайла", t, "${tempFile.length() / 1024} КБ")
            log("")

            // [4/6] Параллельный pipeline: producer → 2 workers → collector
            data class IndexedChunk(
                val chunkIndex: Int,
                val startTimeSec: Float,
                val samples: FloatArray,
            )

            data class IndexedResult(
                val chunkIndex: Int,
                val startTimeSec: Float,
                val endTimeSec: Float,
                val detections: List<BirdDetection>,
            )

            val allDetections = mutableListOf<TimedDetection>()
            var chunkCount = 0
            var skippedChunks = 0

            t = logBegin(
                "Параллельный pipeline (2 воркера)",
                "файл: ${tempFile.name}, chunk: ${BirdClassifier.CHUNK_DURATION_SECONDS} с, порог: 0.1",
            )

            runBlocking {
                val chunksChannel = Channel<IndexedChunk>(capacity = 4)
                val resultsChannel = Channel<IndexedResult>(capacity = 4)

                // Producer: decode → process → send to chunksChannel
                launch(Dispatchers.IO) {
                    try {
                        AudioFileDecoder.decodeChunked(
                            appContext,
                            Uri.fromFile(tempFile),
                        ) { chunkIndex, startTimeSec, chunk ->
                            chunkCount++

                            val processed = audioChunkProcessor.process(chunk)
                            if (processed == null) {
                                skippedChunks++
                                return@decodeChunked
                            }

                            runBlocking {
                                chunksChannel.send(
                                    IndexedChunk(chunkIndex, startTimeSec, processed.samples)
                                )
                            }
                        }
                    } finally {
                        chunksChannel.close()
                    }
                }

                // Workers: read from chunksChannel → classify → send to resultsChannel
                val workers = classifiers.map { clf ->
                    launch(Dispatchers.Default) {
                        for (ic in chunksChannel) {
                            val endTimeSec = ic.startTimeSec + BirdClassifier.CHUNK_DURATION_SECONDS
                            val detections = clf.classify(ic.samples)
                            resultsChannel.send(
                                IndexedResult(ic.chunkIndex, ic.startTimeSec, endTimeSec, detections)
                            )
                        }
                    }
                }

                // Closer: wait for all workers, then close resultsChannel
                launch {
                    workers.forEach { it.join() }
                    resultsChannel.close()
                }

                // Collector: receive results in main coroutine
                for (result in resultsChannel) {
                    for (d in result.detections) {
                        allDetections.add(
                            TimedDetection(result.chunkIndex, result.startTimeSec, result.endTimeSec, d)
                        )
                    }
                }
            }

            // Сортировка по chunkIndex для детерминированного порядка
            allDetections.sortBy { it.chunkIndex }

            val classifiedChunks = chunkCount - skippedChunks
            if (classifiedChunks == 0) {
                log("ОШИБКА: AudioChunkProcessor пропустил все чанки!")
                return
            }

            val skipPct = skippedChunks * 100 / chunkCount.coerceAtLeast(1)
            val avgMs = (System.currentTimeMillis() - t) / classifiedChunks
            logEnd(
                "Параллельный pipeline (2 воркера)", t,
                "$chunkCount чанков ($skippedChunks пропущено = $skipPct%), $classifiedChunks классифицировано, " +
                    "${allDetections.size} обнаружений, среднее $avgMs мс/чанк",
            )
            log("")

            // [5/6] Сопоставление с эталоном
            val recordingEndSec = allDetections.maxOfOrNull { it.endTimeSec } ?: 0f
            t = logBegin(
                "Сопоставление обнаружений с эталоном",
                "${allDetections.size} обнаружений × ${groundTruth.size} эталонных записей, окно: [GT-3с, след.GT]",
            )
            val matchResults = matchDetections(groundTruth, allDetections, recordingEndSec)
            val hits = matchResults.count { it.bestDetection != null }
            logEnd("Сопоставление обнаружений с эталоном", t, "$hits совпадений из ${matchResults.size}")
            log("")

            // [6/6] Формирование отчёта
            t = logBegin("Формирование отчёта", "режим: С AudioChunkProcessor (ПАРАЛЛЕЛЬНЫЙ)")
            printComparisonTable("С AudioChunkProcessor (ПАРАЛЛЕЛЬНЫЙ)", groundTruth, matchResults, allDetections)
            printThresholdAnalysis(matchResults)
            printTimeline(allDetections, groundTruth)
            logEnd("Формирование отчёта", t)
            log("")

            log("═══ Общее время теста (параллельный): ${formatElapsed(System.currentTimeMillis() - testStartMs)} ═══")
            log("")
        } finally {
            classifiers.forEach { it.close() }
            tempFile.delete()
        }
    }

    // ── Ground truth parsing ──

    private fun parseGroundTruth(text: String): List<GroundTruth> {
        val regex = Regex("""(\d+:\d+)\s+(.+?)\s*/\s*(.+?)\s*\((.+?)\)\s*$""")
        return text.lines().mapIndexedNotNull { idx, line ->
            val trimmed = line.trim()
            if (trimmed.isBlank()) return@mapIndexedNotNull null
            val match = regex.find(trimmed) ?: return@mapIndexedNotNull null
            val (time, ruName, enName, sciName) = match.destructured
            val parts = time.split(":")
            val seconds = parts[0].toInt() * 60 + parts[1].toInt()
            GroundTruth(idx + 1, seconds, time, ruName.trim(), enName.trim(), sciName.trim())
        }
    }

    // ── Matching logic ──

    /**
     * For each ground truth entry, find detections of the same species
     * within its natural time window: [gt.time - 3s, next_gt.time].
     *
     * Window rationale:
     *   -3s: chunk boundary may start slightly before annotation time
     *   end = next annotation time: each species gets the full recording slot
     *         until the next annotated species begins, capturing the complete
     *         vocalization period without an arbitrary fixed offset.
     *   For the last entry, the window extends to [recordingEndSec].
     */
    private fun matchDetections(
        groundTruth: List<GroundTruth>,
        detections: List<TimedDetection>,
        recordingEndSec: Float,
    ): List<MatchResult> {
        return groundTruth.mapIndexed { i, gt ->
            val windowStart = (gt.timeSeconds - 3).toFloat().coerceAtLeast(0f)
            val windowEnd = if (i < groundTruth.lastIndex) {
                groundTruth[i + 1].timeSeconds.toFloat()
            } else {
                recordingEndSec
            }

            val matches = detections.filter { td ->
                td.startTimeSec >= windowStart && td.startTimeSec <= windowEnd &&
                    matchesScientificName(gt.scientificName, td.detection.scientificName)
            }

            val best = matches.maxByOrNull { it.detection.confidence }
            MatchResult(gt, best, matches, windowStart, windowEnd)
        }
    }

    /**
     * Taxonomy synonyms: ground truth name → model label name.
     * BirdNET labels may use older taxonomy than current sources.
     */
    private val taxonomySynonyms = mapOf(
        "coloeus monedula" to "corvus monedula",           // Jackdaw: new genus → old
        "columba" to "columba livia",                       // Pigeon: genus only → full species
    )

    private fun matchesScientificName(groundTruth: String, modelLabel: String): Boolean {
        val gt = groundTruth.lowercase().trim()
        val ml = modelLabel.lowercase().trim()
        val gtNormalized = taxonomySynonyms[gt] ?: gt
        return gtNormalized == ml || ml.startsWith(gtNormalized) || gtNormalized.startsWith(ml)
    }

    // ── Output: Comparison table ──

    private fun printComparisonTable(
        benchmarkLabel: String,
        groundTruth: List<GroundTruth>,
        matchResults: List<MatchResult>,
        allDetections: List<TimedDetection>,
    ) {
        val sep = "═".repeat(175)
        val thinSep = "─".repeat(175)
        log(sep)
        log("РЕЗУЛЬТАТЫ БЕНЧМАРКА [$benchmarkLabel]: sample/1")
        log(sep)
        log("")

        log(String.format(
            "%-3s │ %-5s │ %-11s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
            "#", "Время", "Окно", "Ожидаемый вид", "Обнаруженный вид",
            "Макс.ув.", "Время обн.", "Результат",
        ))
        log(thinSep)

        var matched = 0
        var missed = 0

        for (mr in matchResults) {
            val gt = mr.groundTruth
            val expectedStr = formatSpeciesWithRu(gt.scientificName, gt.russianName)
            val windowStr = "${formatTime(mr.windowStartSec.toInt())}-${formatTime(mr.windowEndSec.toInt())}"

            if (mr.bestDetection != null) {
                matched++
                val det = mr.bestDetection.detection
                val detectedStr = formatSpeciesWithRu(det.scientificName)
                log(String.format(
                    "%-3d │ %-5s │ %-11s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index,
                    gt.timeFormatted,
                    windowStr,
                    expectedStr,
                    detectedStr,
                    "%.3f".format(det.confidence),
                    formatTime(mr.bestDetection.startTimeSec.toInt()),
                    "НАЙДЕН",
                ))
            } else {
                missed++
                log(String.format(
                    "%-3d │ %-5s │ %-11s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index,
                    gt.timeFormatted,
                    windowStr,
                    expectedStr,
                    "—",
                    "—",
                    "—",
                    "ПРОПУСК",
                ))
            }
        }

        log(thinSep)
        log("")
        log("ИТОГО:")
        log("  Всего видов в эталоне: ${groundTruth.size}")
        log("  Найдено (НАЙДЕН):     $matched (%.1f%%)".format(matched * 100.0 / groundTruth.size))
        log("  Не найдено (ПРОПУСК): $missed (%.1f%%)".format(missed * 100.0 / groundTruth.size))
        log("")

        // Unexpected species (false positives with high confidence)
        val expectedSciNames = groundTruth.map { it.scientificName.lowercase() }.toSet()
        val unexpected = allDetections
            .filter { it.detection.confidence >= 0.5f }
            .groupBy { it.detection.scientificName }
            .filter { (sp, _) -> expectedSciNames.none { exp -> matchesScientificName(exp, sp) } }

        if (unexpected.isNotEmpty()) {
            log("ЛОЖНЫЕ СРАБАТЫВАНИЯ (уверенность >= 0.5, вид отсутствует в эталоне):")
            log("  Всего ложных видов: ${unexpected.size}")
            log("")
            for ((sp, dets) in unexpected.entries.sortedByDescending { it.value.maxOf { d -> d.detection.confidence } }) {
                val best = dets.maxByOrNull { it.detection.confidence }!!
                val ruName = russianName(sp)
                val nameStr = if (ruName.isNotBlank()) "$sp ($ruName)" else "$sp (${best.detection.commonName})"
                log("  • $nameStr — макс: ${"%.3f".format(best.detection.confidence)} " +
                    "в ${formatTime(best.startTimeSec.toInt())}, встречается ${dets.size} раз(а)")
            }
            log("")
        }
    }

    // ── Output: Threshold analysis ──

    private fun printThresholdAnalysis(matchResults: List<MatchResult>) {
        log("ТОЧНОСТЬ ПРИ РАЗНЫХ ПОРОГАХ УВЕРЕННОСТИ:")
        log("  (Показывает, сколько эталонных видов будут найдены, если принимать")
        log("   только обнаружения с уверенностью не ниже указанного порога)")
        log("")
        log(String.format("  %-12s │ %-9s │ %-11s │ %-10s", "Порог", "Найдено", "Не найдено", "Точность"))
        log("  " + "─".repeat(50))

        val thresholds = listOf(0.1f, 0.2f, 0.3f, 0.5f, 0.6f, 0.8f)
        val total = matchResults.size

        for (threshold in thresholds) {
            val matched = matchResults.count { mr ->
                mr.bestDetection != null && mr.bestDetection.detection.confidence >= threshold
            }
            val missed = total - matched
            log(String.format(
                "  %-12s │ %-9d │ %-11d │ %-10s",
                "≥ %.1f".format(threshold), matched, missed,
                "%.1f%%".format(matched * 100.0 / total),
            ))
        }
        log("")
    }

    // ── Output: Timeline (top detections per time segment) ──

    private fun printTimeline(
        allDetections: List<TimedDetection>,
        groundTruth: List<GroundTruth>,
    ) {
        log("ХРОНОЛОГИЯ ОБНАРУЖЕНИЙ (топ видов в 5-секундных окнах, уверенность >= 0.3):")
        log("─".repeat(120))

        val totalSec = if (allDetections.isNotEmpty()) {
            allDetections.maxOf { it.endTimeSec }.toInt()
        } else return

        for (windowStart in 0..totalSec step 5) {
            val windowEnd = windowStart + 5
            val windowDets = allDetections.filter { td ->
                td.startTimeSec >= windowStart && td.startTimeSec < windowEnd &&
                    td.detection.confidence >= 0.3f
            }

            val gtInWindow = groundTruth.filter { it.timeSeconds in windowStart until windowEnd }
            val gtMarker = if (gtInWindow.isNotEmpty()) {
                " ◄◄ ЭТАЛОН: " + gtInWindow.joinToString(", ") {
                    "${it.scientificName} (${it.russianName})"
                }
            } else ""

            if (windowDets.isNotEmpty()) {
                val topSpecies = windowDets
                    .groupBy { it.detection.scientificName }
                    .map { (sp, dets) -> sp to dets.maxOf { it.detection.confidence } }
                    .sortedByDescending { it.second }
                    .take(3)

                val speciesStr = topSpecies.joinToString(", ") { (sp, conf) ->
                    val ruName = russianName(sp)
                    if (ruName.isNotBlank()) "$sp/$ruName(${"%.2f".format(conf)})"
                    else "$sp(${"%.2f".format(conf)})"
                }
                log("  ${formatTime(windowStart)}-${formatTime(windowEnd)}: $speciesStr$gtMarker")
            } else if (gtInWindow.isNotEmpty()) {
                log("  ${formatTime(windowStart)}-${formatTime(windowEnd)}: [нет обнаружений >= 0.3]$gtMarker")
            }
        }
        log("")
    }

    // ── Logging helpers ──

    /**
     * Логирует начало этапа с параметрами и возвращает метку времени старта.
     */
    private fun logBegin(stage: String, params: String = ""): Long {
        if (params.isBlank()) log("[START] $stage")
        else log("[START] $stage | $params")
        return System.currentTimeMillis()
    }

    /**
     * Логирует конец этапа с результатом и временем выполнения.
     */
    private fun logEnd(stage: String, startMs: Long, result: String = "") {
        val elapsed = formatElapsed(System.currentTimeMillis() - startMs)
        if (result.isBlank()) log("[ END ] $stage | $elapsed")
        else log("[ END ] $stage | $result | $elapsed")
    }

    // ── Utilities ──

    private fun formatTime(seconds: Int): String =
        "%d:%02d".format(seconds / 60, seconds % 60)

    /** До 1 с — миллисекунды; 1–59 с — секунды; 60+ с — минуты и секунды. */
    private fun formatElapsed(ms: Long): String = when {
        ms < 1_000 -> "$ms мс"
        ms < 60_000 -> "${ms / 1000} сек"
        else -> {
            val totalSec = ms / 1000
            "${totalSec / 60} мин ${"%02d".format(totalSec % 60)} сек"
        }
    }

    private fun loadModel(
        context: android.content.Context,
        assetPath: String,
    ): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    companion object {
        private const val TAG = "BENCH"

        private fun log(message: String) {
            Log.i(TAG, message)
        }
    }
}
