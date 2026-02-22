package com.birdsong.analyzer.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

// ── Configuration ──

object BenchmarkConfig {
    const val CONFIDENCE_THRESHOLD = 0.1f
    const val FALSE_POSITIVE_THRESHOLD = 0.5f
    const val TIMELINE_THRESHOLD = 0.3f
    const val PARALLEL_WORKERS = 4
    const val TFLITE_THREADS_PER_WORKER = 1
    const val CHANNEL_CAPACITY = 8
    const val PROGRESS_INTERVAL = 50
    const val AUDIO_ASSET_PATH = "benchmark/1/1.mp3"
    const val GT_ASSET_PATH = "benchmark/1/1.txt"

    // Recording location for meta-model geo-filtering.
    // weekRange = full year: pure geographic filter, no temporal assumption.
    // Continental outliers (Africa, Americas) score ~0 across all weeks.
    val LOCATION: LocationMeta = LocationMeta(
        latitude = 53.9,
        longitude = 27.6,
        weekOfYear = 1,  // unused: weekRange takes precedence
        weekRange = 1..52,
    )

    // Blended meta-model alpha: score = alpha + (1 - alpha) * meta_score
    // Prevents complete suppression of low-eBird edge-case species (irruptive visitors, range edges).
    val META_ALPHA: Float = BirdNetV24Classifier.DEFAULT_META_ALPHA
}

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

data class InferenceResult(
    val detections: List<TimedDetection>,
    val totalChunks: Int,
    val skippedChunks: Int,
)

data class IndexedChunk(
    val chunkIndex: Int,
    val startTimeSec: Float,
    val samples: FloatArray,
)

// ── Logging ──

object BenchmarkLogger {
    private const val TAG = "BENCH"

    fun log(message: String) {
        Log.i(TAG, message)
    }

    fun logBegin(stage: String, params: String = ""): Long {
        if (params.isBlank()) log("[START] $stage")
        else log("[START] $stage | $params")
        return System.currentTimeMillis()
    }

    fun logEnd(stage: String, startMs: Long, result: String = "") {
        val elapsed = formatElapsed(System.currentTimeMillis() - startMs)
        if (result.isBlank()) log("[ END ] $stage | $elapsed")
        else log("[ END ] $stage | $result | $elapsed")
    }

    fun formatTime(seconds: Int): String =
        "%d:%02d".format(seconds / 60, seconds % 60)

    fun formatElapsed(ms: Long): String = when {
        ms < 1_000 -> "$ms мс"
        ms < 60_000 -> "${ms / 1000} сек"
        else -> {
            val totalSec = ms / 1000
            "${totalSec / 60} мин ${"%02d".format(totalSec % 60)} сек"
        }
    }
}

// ── Ground truth parsing ──

object GroundTruthParser {
    private val regex = Regex("""(\d+:\d+)\s+(.+?)\s*/\s*(.+?)\s*\((.+?)\)\s*$""")

    fun parse(text: String): List<GroundTruth> {
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
}

// ── Detection matching ──

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
object DetectionMatcher {

    val taxonomySynonyms = mapOf(
        "coloeus monedula" to "corvus monedula",
        "columba" to "columba livia",
    )

    fun matchDetections(
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

    fun matchesScientificName(groundTruth: String, modelLabel: String): Boolean {
        val gt = groundTruth.lowercase().trim()
        val ml = modelLabel.lowercase().trim()
        val gtNormalized = taxonomySynonyms[gt] ?: gt
        return gtNormalized == ml || ml.startsWith(gtNormalized) || gtNormalized.startsWith(ml)
    }
}

// ── Reporting ──

class BenchmarkReporter(private val ruLabelMap: Map<String, String>) {

    private fun log(message: String) = BenchmarkLogger.log(message)

    private fun russianName(scientificName: String): String =
        ruLabelMap[scientificName] ?: ""

    private fun formatSpeciesWithRu(
        scientificName: String,
        ruName: String = russianName(scientificName),
    ): String {
        return if (ruName.isNotBlank()) "$scientificName ($ruName)" else scientificName
    }

    fun printComparisonTable(
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
            val windowStr = "${BenchmarkLogger.formatTime(mr.windowStartSec.toInt())}-${BenchmarkLogger.formatTime(mr.windowEndSec.toInt())}"

            if (mr.bestDetection != null) {
                matched++
                val det = mr.bestDetection.detection
                val detectedStr = formatSpeciesWithRu(det.scientificName)
                log(String.format(
                    "%-3d │ %-5s │ %-11s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index, gt.timeFormatted, windowStr,
                    expectedStr, detectedStr,
                    "%.3f".format(det.confidence),
                    BenchmarkLogger.formatTime(mr.bestDetection.startTimeSec.toInt()),
                    "НАЙДЕН",
                ))
            } else {
                missed++
                log(String.format(
                    "%-3d │ %-5s │ %-11s │ %-45s │ %-45s │ %-8s │ %-10s │ %-9s",
                    gt.index, gt.timeFormatted, windowStr,
                    expectedStr, "—", "—", "—", "ПРОПУСК",
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

        // False positives (high confidence species absent from ground truth)
        val expectedSciNames = groundTruth.map { it.scientificName.lowercase() }.toSet()
        val unexpected = allDetections
            .filter { it.detection.confidence >= BenchmarkConfig.FALSE_POSITIVE_THRESHOLD }
            .groupBy { it.detection.scientificName }
            .filter { (sp, _) -> expectedSciNames.none { exp -> DetectionMatcher.matchesScientificName(exp, sp) } }

        if (unexpected.isNotEmpty()) {
            log("ЛОЖНЫЕ СРАБАТЫВАНИЯ (уверенность >= ${BenchmarkConfig.FALSE_POSITIVE_THRESHOLD}, вид отсутствует в эталоне):")
            log("  Всего ложных видов: ${unexpected.size}")
            log("")
            for ((sp, dets) in unexpected.entries.sortedByDescending { it.value.maxOf { d -> d.detection.confidence } }) {
                val best = dets.maxByOrNull { it.detection.confidence }!!
                val ruName = russianName(sp)
                val nameStr = if (ruName.isNotBlank()) "$sp ($ruName)" else "$sp (${best.detection.commonName})"
                log("  • $nameStr — макс: ${"%.3f".format(best.detection.confidence)} " +
                    "в ${BenchmarkLogger.formatTime(best.startTimeSec.toInt())}, встречается ${dets.size} раз(а)")
            }
            log("")
        }
    }

    fun printThresholdAnalysis(matchResults: List<MatchResult>) {
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

    fun printTimeline(
        allDetections: List<TimedDetection>,
        groundTruth: List<GroundTruth>,
    ) {
        log("ХРОНОЛОГИЯ ОБНАРУЖЕНИЙ (топ видов в 5-секундных окнах, уверенность >= ${BenchmarkConfig.TIMELINE_THRESHOLD}):")
        log("─".repeat(120))

        val totalSec = if (allDetections.isNotEmpty()) {
            allDetections.maxOf { it.endTimeSec }.toInt()
        } else return

        for (windowStart in 0..totalSec step 5) {
            val windowEnd = windowStart + 5
            val windowDets = allDetections.filter { td ->
                td.startTimeSec >= windowStart && td.startTimeSec < windowEnd &&
                    td.detection.confidence >= BenchmarkConfig.TIMELINE_THRESHOLD
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
                log("  ${BenchmarkLogger.formatTime(windowStart)}-${BenchmarkLogger.formatTime(windowEnd)}: $speciesStr$gtMarker")
            } else if (gtInWindow.isNotEmpty()) {
                log("  ${BenchmarkLogger.formatTime(windowStart)}-${BenchmarkLogger.formatTime(windowEnd)}: [нет обнаружений >= ${BenchmarkConfig.TIMELINE_THRESHOLD}]$gtMarker")
            }
        }
        log("")
    }
}

// ── Runner (template method) ──

/**
 * Orchestrates the benchmark flow: parse GT -> copy audio -> inference -> match -> report.
 * The inference step is provided as a lambda, allowing different test configurations
 * (sequential, with processor, parallel) to share the same orchestration logic.
 */
class BenchmarkRunner(ruLabelMap: Map<String, String>) {

    private val reporter = BenchmarkReporter(ruLabelMap)

    fun run(
        label: String,
        tempFileName: String,
        inference: (Context, Uri) -> InferenceResult,
    ) {
        val testStartMs = System.currentTimeMillis()
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext

        BenchmarkLogger.log("═══ БЕНЧМАРК $label ═══")
        BenchmarkLogger.log("")

        // [1] Parse ground truth
        var t = BenchmarkLogger.logBegin(
            "Парсинг эталонных аннотаций", "файл: ${BenchmarkConfig.GT_ASSET_PATH}",
        )
        val groundTruth = testContext.assets.open(BenchmarkConfig.GT_ASSET_PATH).use { stream ->
            GroundTruthParser.parse(stream.bufferedReader().readText())
        }
        BenchmarkLogger.logEnd("Парсинг эталонных аннотаций", t, "${groundTruth.size} видов")
        BenchmarkLogger.log("")

        // [2] Copy audio file to temp
        val tempFile = File(appContext.cacheDir, tempFileName)
        try {
            t = BenchmarkLogger.logBegin(
                "Копирование аудиофайла",
                "источник: ${BenchmarkConfig.AUDIO_ASSET_PATH} → ${tempFile.absolutePath}",
            )
            testContext.assets.open(BenchmarkConfig.AUDIO_ASSET_PATH).use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            BenchmarkLogger.logEnd("Копирование аудиофайла", t, "${tempFile.length() / 1024} КБ")
            BenchmarkLogger.log("")

            // [3] Inference (delegated to caller)
            val result = inference(appContext, Uri.fromFile(tempFile))

            // [4] Match detections to ground truth
            val recordingEndSec = result.detections.maxOfOrNull { it.endTimeSec } ?: 0f
            t = BenchmarkLogger.logBegin(
                "Сопоставление обнаружений с эталоном",
                "${result.detections.size} обнаружений × ${groundTruth.size} эталонных записей, окно: [GT-3с, след.GT]",
            )
            val matchResults = DetectionMatcher.matchDetections(groundTruth, result.detections, recordingEndSec)
            val hits = matchResults.count { it.bestDetection != null }
            BenchmarkLogger.logEnd("Сопоставление обнаружений с эталоном", t, "$hits совпадений из ${matchResults.size}")
            BenchmarkLogger.log("")

            // [5] Reports
            t = BenchmarkLogger.logBegin("Формирование отчёта", "режим: $label")
            reporter.printComparisonTable(label, groundTruth, matchResults, result.detections)
            reporter.printThresholdAnalysis(matchResults)
            reporter.printTimeline(result.detections, groundTruth)
            BenchmarkLogger.logEnd("Формирование отчёта", t)
            BenchmarkLogger.log("")

            BenchmarkLogger.log("═══ Общее время теста ($label): ${BenchmarkLogger.formatElapsed(System.currentTimeMillis() - testStartMs)} ═══")
            BenchmarkLogger.log("")
        } finally {
            tempFile.delete()
        }
    }
}
