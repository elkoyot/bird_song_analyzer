package com.birdsong.analyzer.ml

import android.content.Context
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicInteger

/**
 * Benchmark-тест: прогоняет реальный аудиофайл с аннотированными видами птиц
 * через параллельный ML-пайплайн и сравнивает результаты с ground truth.
 *
 * Входные данные: androidTest/assets/benchmark/1/
 *   - 1.mp3 — ~16 минут аудио с голосами 44 видов птиц
 *   - 1.txt — аннотации: время + название вида (RU / EN (Scientific))
 *
 * Pipeline: AudioChunkProcessor (bandpass + нормализация) → N воркеров с TFLite.
 *
 * Shared infrastructure (logging, parsing, matching, reporting) вынесена
 * в [BenchmarkTestInfra.kt].
 *
 * Запуск: Android Studio → правый клик → Run.
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class BirdNetBenchmarkTest {

    private lateinit var audioModelBuffer: MappedByteBuffer
    private lateinit var metaModelBuffer: MappedByteBuffer
    private lateinit var labels: List<Pair<String, String>>
    private lateinit var ruLabelMap: Map<String, String>
    private lateinit var runner: BenchmarkRunner

    // ── Setup ──

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        var t = BenchmarkLogger.logBegin("Загрузка аудиомодели", "путь: ${BirdNetV24Classifier.AUDIO_MODEL_PATH}")
        audioModelBuffer = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        BenchmarkLogger.logEnd("Загрузка аудиомодели", t)

        t = BenchmarkLogger.logBegin("Загрузка мета-модели", "путь: ${BirdNetV24Classifier.META_MODEL_PATH}")
        metaModelBuffer = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)
        BenchmarkLogger.logEnd("Загрузка мета-модели", t)

        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/en_us.txt"
        t = BenchmarkLogger.logBegin("Загрузка меток EN", "путь: $labelsPath")
        labels = context.assets.open(labelsPath).use { LabelParser.load(it) }
        BenchmarkLogger.logEnd("Загрузка меток EN", t, "${labels.size} видов")

        val ruLabelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/ru.txt"
        t = BenchmarkLogger.logBegin("Загрузка меток RU", "путь: $ruLabelsPath")
        val ruLabels = context.assets.open(ruLabelsPath).use { LabelParser.load(it) }
        ruLabelMap = ruLabels.associate { (sci, ru) -> sci to ru }
        BenchmarkLogger.logEnd("Загрузка меток RU", t, "${ruLabels.size} видов")

        runner = BenchmarkRunner(ruLabelMap)
    }

    // ── Test ──

    @Test
    fun benchmark_sample1_withProcessor_parallel() {
        val classifiers = createClassifiers(BenchmarkConfig.PARALLEL_WORKERS)
        try {
            runner.run("С AudioChunkProcessor (ПАРАЛЛЕЛЬНЫЙ)", "benchmark_1_par.mp3") { context, uri ->
                runParallelInference(context, uri, classifiers, AudioChunkProcessor(), BenchmarkConfig.LOCATION)
            }
        } finally {
            classifiers.forEach { it.close() }
        }
    }

    // ── Parallel inference ──

    /**
     * Parallel pipeline: Producer (IO) → Channel → N Workers (Default) → Channel → Collector.
     *
     * - trySendBlocking instead of runBlocking inside coroutine
     * - AtomicInteger for cross-coroutine counters
     * - Worker stats logged after runBlocking (all coroutines finished)
     */
    private fun runParallelInference(
        context: Context,
        uri: Uri,
        classifiers: List<BirdNetV24Classifier>,
        processor: AudioChunkProcessor,
        location: LocationMeta? = null,
    ): InferenceResult {
        val numWorkers = classifiers.size
        val allDetections = mutableListOf<TimedDetection>()
        val totalChunks = AtomicInteger(0)
        val skippedChunks = AtomicInteger(0)

        // Per-worker stats arrays: each index written by one worker only, no contention
        val workerChunkCounts = IntArray(numWorkers)
        val workerTotalMs = LongArray(numWorkers)

        val locationLabel = if (location != null)
            "lat=%.4f lon=%.4f неделя=${location.weekOfYear}".format(location.latitude, location.longitude)
        else "нет"
        val t = BenchmarkLogger.logBegin(
            "Параллельный pipeline ($numWorkers воркера)",
            "chunk: ${BirdClassifier.CHUNK_DURATION_SECONDS} с, порог: ${BenchmarkConfig.CONFIDENCE_THRESHOLD}, локация: $locationLabel",
        )

        runBlocking {
            val chunksChannel = Channel<IndexedChunk>(capacity = BenchmarkConfig.CHANNEL_CAPACITY)
            val resultsChannel = Channel<List<TimedDetection>>(capacity = BenchmarkConfig.CHANNEL_CAPACITY)

            // Producer: decode → send raw chunks to channel (no processing)
            launch(Dispatchers.IO) {
                try {
                    AudioFileDecoder.decodeChunked(
                        context, uri,
                        hopSize = BirdClassifier.SAMPLES_PER_CHUNK, // 0% overlap for benchmark
                    ) { chunkIndex, startTimeSec, chunk ->
                        totalChunks.incrementAndGet()
                        chunksChannel.trySendBlocking(
                            IndexedChunk(chunkIndex, startTimeSec, chunk),
                        ).getOrThrow()
                    }
                } finally {
                    chunksChannel.close()
                }
            }

            // Workers: process + classify in parallel
            val workers = classifiers.mapIndexed { wIdx, clf ->
                val workerId = wIdx + 1
                launch(Dispatchers.Default) {
                    for (ic in chunksChannel) {
                        val processed = processor.process(ic.samples)
                        if (processed == null) {
                            skippedChunks.incrementAndGet()
                            continue
                        }

                        val endTimeSec = ic.startTimeSec + BirdClassifier.CHUNK_DURATION_SECONDS
                        val classifyStart = System.currentTimeMillis()
                        val detections = clf.classify(processed.samples, location)
                        val classifyMs = System.currentTimeMillis() - classifyStart
                        workerChunkCounts[wIdx]++
                        workerTotalMs[wIdx] += classifyMs

                        val batch = detections.map { d ->
                            TimedDetection(ic.chunkIndex, ic.startTimeSec, endTimeSec, d)
                        }
                        resultsChannel.send(batch)
                    }

                    val avg = if (workerChunkCounts[wIdx] > 0) workerTotalMs[wIdx] / workerChunkCounts[wIdx] else 0
                    BenchmarkLogger.log(
                        "  [W$workerId] Воркер завершён: ${workerChunkCounts[wIdx]} чанков, " +
                            "всего ${workerTotalMs[wIdx]} мс, среднее $avg мс/чанк",
                    )
                }
            }

            // Closer: wait for all workers, then close resultsChannel
            launch {
                workers.forEach { it.join() }
                resultsChannel.close()
            }

            // Collector: receive results in main coroutine
            for (batch in resultsChannel) {
                allDetections.addAll(batch)
            }
        }

        // After runBlocking: all coroutines finished, stats arrays are stable
        BenchmarkLogger.log("")
        BenchmarkLogger.log("  ── Статистика воркеров ──")
        for (i in classifiers.indices) {
            val avg = if (workerChunkCounts[i] > 0) workerTotalMs[i] / workerChunkCounts[i] else 0
            BenchmarkLogger.log(
                "  [W${i + 1}] ${workerChunkCounts[i]} чанков, " +
                    "суммарно ${BenchmarkLogger.formatElapsed(workerTotalMs[i])}, среднее $avg мс/чанк",
            )
        }
        BenchmarkLogger.log("  Всего чанков по воркерам: ${workerChunkCounts.sum()}")
        BenchmarkLogger.log("")

        // Sort by chunkIndex for deterministic order
        allDetections.sortBy { it.chunkIndex }

        val total = totalChunks.get()
        val skipped = skippedChunks.get()
        val classifiedChunks = total - skipped

        if (classifiedChunks == 0) {
            BenchmarkLogger.log("ОШИБКА: AudioChunkProcessor пропустил все чанки!")
            return InferenceResult(emptyList(), total, skipped)
        }

        val skipPct = skipped * 100 / total.coerceAtLeast(1)
        val avgMs = (System.currentTimeMillis() - t) / classifiedChunks
        BenchmarkLogger.logEnd(
            "Параллельный pipeline ($numWorkers воркера)", t,
            "$total чанков ($skipped пропущено = $skipPct%), $classifiedChunks классифицировано, " +
                "${allDetections.size} обнаружений, среднее $avgMs мс/чанк",
        )
        BenchmarkLogger.log("")

        return InferenceResult(allDetections, total, skipped)
    }

    // ── Factory methods ──

    private fun createClassifiers(count: Int): List<BirdNetV24Classifier> {
        val t = BenchmarkLogger.logBegin(
            "Создание параллельных классификаторов",
            "$count × BirdNetV24Classifier, tfliteThreads=${BenchmarkConfig.TFLITE_THREADS_PER_WORKER}",
        )
        val classifiers = (1..count).map { i ->
            BirdNetV24Classifier(
                audioModelBuffer, metaModelBuffer, labels,
                confidenceThreshold = BenchmarkConfig.CONFIDENCE_THRESHOLD,
                tfliteThreads = BenchmarkConfig.TFLITE_THREADS_PER_WORKER,
                metaAlpha = BenchmarkConfig.META_ALPHA,
            ).also { BenchmarkLogger.log("  Классификатор #$i создан") }
        }
        BenchmarkLogger.logEnd("Создание параллельных классификаторов", t)

        // Warmup: sequential dummy inference to trigger TFLite JIT before pipeline starts
        val warmupT = BenchmarkLogger.logBegin("TFLite warmup", "$count интерпретаторов")
        val warmupChunk = FloatArray(BirdClassifier.SAMPLES_PER_CHUNK)
        for ((i, clf) in classifiers.withIndex()) {
            val ms = System.currentTimeMillis()
            runBlocking { clf.classify(warmupChunk) }
            BenchmarkLogger.log("  Классификатор #${i + 1} прогрет: ${System.currentTimeMillis() - ms} мс")
        }
        BenchmarkLogger.logEnd("TFLite warmup", warmupT)
        BenchmarkLogger.log("")

        return classifiers
    }

    // ── Utilities ──

    private fun loadModel(context: Context, assetPath: String): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}
