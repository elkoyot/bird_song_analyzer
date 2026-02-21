package com.birdsong.analyzer.ml

import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
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
 *   - 1.m4a — ~16 минут аудио с голосами 44 видов птиц
 *   - 1.txt — аннотации: время + название вида (RU / EN (Scientific))
 *
 * Результат: таблица сравнения + статистика точности.
 *
 * Запуск: Android Studio → правый клик → Run (на реальном устройстве).
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class BirdNetBenchmarkTest {

    private lateinit var classifier: BirdNetV24Classifier
    private lateinit var labels: List<Pair<String, String>>
    private lateinit var ruLabelMap: Map<String, String> // scientificName → русское название
    private val audioChunkProcessor = AudioChunkProcessor()
    private var modelLoadTimeMs: Long = 0

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
    )

    // ── Setup / Teardown ──

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val start = System.currentTimeMillis()

        val audioModel = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        val metaModel = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)
        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/en_us.txt"
        labels = context.assets.open(labelsPath).use { LabelParser.load(it) }

        val ruLabelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        val ruLabels = context.assets.open(ruLabelsPath).use { LabelParser.load(it) }
        ruLabelMap = ruLabels.associate { (sci, ru) -> sci to ru }

        classifier = BirdNetV24Classifier(audioModel, metaModel, labels, confidenceThreshold = 0.1f)
        modelLoadTimeMs = System.currentTimeMillis() - start
        log("Модель загружена: ${labels.size} видов, $modelLoadTimeMs мс")
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

        val groundTruth = testContext.assets.open("benchmark/1/1.txt").use { stream ->
            parseGroundTruth(stream.bufferedReader().readText())
        }
        log("Эталон: ${groundTruth.size} видов")
        log("")

        val tempFile = File(appContext.cacheDir, "benchmark_1.mp3")
        try {
            testContext.assets.open("benchmark/1/1.mp3").use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            log("Аудиофайл: ${tempFile.absolutePath} (${tempFile.length() / 1024} КБ)")

            val allDetections = mutableListOf<TimedDetection>()
            var chunkCount = 0
            val inferenceStart = System.currentTimeMillis()

            AudioFileDecoder.decodeChunked(
                appContext,
                Uri.fromFile(tempFile),
            ) { chunkIndex, startTimeSec, chunk ->
                val endTimeSec = startTimeSec + BirdClassifier.CHUNK_DURATION_SECONDS

                if (chunkIndex == 0) {
                    var peak = 0f
                    var sumSq = 0.0
                    for (s in chunk) { sumSq += s * s; val a = kotlin.math.abs(s); if (a > peak) peak = a }
                    val rms = kotlin.math.sqrt(sumSq / chunk.size)
                    log("Первый чанк: size=${chunk.size}, RMS=%.6f, peak=%.6f".format(rms, peak))
                }

                val detections = runBlocking { classifier.classify(chunk) }

                for (d in detections) {
                    allDetections.add(TimedDetection(chunkIndex, startTimeSec, endTimeSec, d))
                }
                chunkCount++

                if (chunkCount % 50 == 0) {
                    log("  ... обработано $chunkCount чанков (${formatTime(startTimeSec.toInt())}), " +
                        "обнаружений: ${allDetections.size}")
                }
            }

            val totalInferenceMs = System.currentTimeMillis() - inferenceStart
            val avgMs = if (chunkCount > 0) totalInferenceMs / chunkCount else 0
            log("")
            log("Декодирование + инференс: $chunkCount чанков за $totalInferenceMs мс" +
                if (chunkCount > 0) " (в среднем $avgMs мс/чанк)" else " — ЧАНКИ НЕ ПОЛУЧЕНЫ!")
            log("Всего сырых обнаружений (уверенность >= 0.1): ${allDetections.size}")
            log("")

            if (chunkCount == 0) {
                log("ОШИБКА: decodeChunked не вернул ни одного чанка.")
                return
            }

            val matchResults = matchDetections(groundTruth, allDetections)

            printComparisonTable("БЕЗ AudioChunkProcessor", groundTruth, matchResults, allDetections)
            printThresholdAnalysis(matchResults)
            printTimeline(allDetections, groundTruth)

            val totalTestMs = System.currentTimeMillis() - testStartMs
            log("═══ Общее время теста (без процессора): ${formatDuration(totalTestMs)} ═══")
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

        val groundTruth = testContext.assets.open("benchmark/1/1.txt").use { stream ->
            parseGroundTruth(stream.bufferedReader().readText())
        }
        log("═══ БЕНЧМАРК С AudioChunkProcessor ═══")
        log("Эталон: ${groundTruth.size} видов")
        log("")

        val tempFile = File(appContext.cacheDir, "benchmark_1_proc.mp3")
        try {
            testContext.assets.open("benchmark/1/1.mp3").use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }

            val allDetections = mutableListOf<TimedDetection>()
            var chunkCount = 0
            var skippedChunks = 0
            val inferenceStart = System.currentTimeMillis()

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
                    log("  ... обработано $chunkCount чанков ($skippedChunks пропущено), " +
                        "позиция ${formatTime(startTimeSec.toInt())}, обнаружений: ${allDetections.size}")
                }
            }

            val totalInferenceMs = System.currentTimeMillis() - inferenceStart
            val classifiedChunks = chunkCount - skippedChunks
            val avgMs = if (classifiedChunks > 0) totalInferenceMs / classifiedChunks else 0
            log("")
            log("Декодирование + обработка + инференс: $chunkCount чанков всего, $skippedChunks пропущено " +
                "(${skippedChunks * 100 / chunkCount.coerceAtLeast(1)}%), " +
                "$classifiedChunks классифицировано за $totalInferenceMs мс" +
                if (classifiedChunks > 0) " (в среднем $avgMs мс/классифицированный чанк)" else "")
            log("Всего сырых обнаружений (уверенность >= 0.1): ${allDetections.size}")
            log("")

            if (classifiedChunks == 0) {
                log("ОШИБКА: AudioChunkProcessor пропустил все чанки!")
                return
            }

            val matchResults = matchDetections(groundTruth, allDetections)

            printComparisonTable("С AudioChunkProcessor", groundTruth, matchResults, allDetections)
            printThresholdAnalysis(matchResults)
            printTimeline(allDetections, groundTruth)

            val totalTestMs = System.currentTimeMillis() - testStartMs
            log("═══ Общее время теста (с процессором): ${formatDuration(totalTestMs)} ═══")
            log("")
        } finally {
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
     * within a time window [gt.time - 3s, gt.time + 20s].
     * Window rationale:
     *   -3s: chunk overlap means a detection may start slightly before annotation
     *   +20s: birds may sing for 10-20 seconds after the annotated start
     */
    private fun matchDetections(
        groundTruth: List<GroundTruth>,
        detections: List<TimedDetection>,
    ): List<MatchResult> {
        return groundTruth.map { gt ->
            val windowStart = (gt.timeSeconds - 3).toFloat()
            val windowEnd = (gt.timeSeconds + 20).toFloat()

            val matches = detections.filter { td ->
                td.startTimeSec >= windowStart && td.startTimeSec <= windowEnd &&
                    matchesScientificName(gt.scientificName, td.detection.scientificName)
            }

            val best = matches.maxByOrNull { it.detection.confidence }
            MatchResult(gt, best, matches)
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
        val sep = "═".repeat(160)
        val thinSep = "─".repeat(160)
        log(sep)
        log("РЕЗУЛЬТАТЫ БЕНЧМАРКА [$benchmarkLabel]: sample/1")
        log(sep)
        log("")

        log(String.format(
            "%-3s │ %-5s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
            "#", "Время", "Ожидаемый вид", "Обнаруженный вид",
            "Уверенн.", "Время обн.", "Результат",
        ))
        log(thinSep)

        var matched = 0
        var missed = 0

        for (mr in matchResults) {
            val gt = mr.groundTruth
            val expectedStr = formatSpeciesWithRu(gt.scientificName, gt.russianName)

            if (mr.bestDetection != null) {
                matched++
                val det = mr.bestDetection.detection
                val detectedStr = formatSpeciesWithRu(det.scientificName)
                log(String.format(
                    "%-3d │ %-5s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index,
                    gt.timeFormatted,
                    expectedStr,
                    detectedStr,
                    "%.3f".format(det.confidence),
                    formatTime(mr.bestDetection.startTimeSec.toInt()),
                    "НАЙДЕН",
                ))
            } else {
                missed++
                log(String.format(
                    "%-3d │ %-5s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index,
                    gt.timeFormatted,
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

    // ── Utilities ──

    private fun formatTime(seconds: Int): String {
        return "%d:%02d".format(seconds / 60, seconds % 60)
    }

    private fun formatDuration(ms: Long): String {
        val totalSec = ms / 1000
        val min = totalSec / 60
        val sec = totalSec % 60
        return "%d мин %02d сек (%d мс)".format(min, sec, ms)
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
        private const val TAG = "BirdNetBenchmark"

        private fun log(message: String) {
            Log.i(TAG, message)
        }
    }
}
