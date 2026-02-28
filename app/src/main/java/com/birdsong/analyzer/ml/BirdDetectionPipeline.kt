package com.birdsong.analyzer.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicInteger
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single source of truth for bird audio analysis.
 *
 * Encapsulates [AudioChunkProcessor] (pre-filtering) + [BirdClassifier] (inference).
 * Stateless — all mutable state (aggregation) is managed by callers.
 *
 * Reference pipeline matches benchmark [BirdNetBenchmarkTest.benchmark_sample1_withProcessor].
 */
@Singleton
class BirdDetectionPipeline @Inject constructor(
    private var audioChunkProcessor: AudioChunkProcessor,
    private var _classifier: BirdClassifier,
) {
    val classifier: BirdClassifier get() = _classifier

    fun configure(processor: AudioChunkProcessor, classifier: BirdClassifier) {
        _classifier.close()
        audioChunkProcessor = processor
        _classifier = classifier
    }

    data class ChunkResult(
        val detections: List<BirdDetection>,
        val processed: Boolean,
    )

    /**
     * Process a single audio chunk: pre-filter → inference.
     *
     * @param chunk raw float32 PCM samples (expected [classifier.samplesPerChunk])
     * @param location optional GPS + week-of-year for meta-model filtering
     * @return [ChunkResult] with detections; [ChunkResult.processed] = false if chunk was skipped
     */
    suspend fun processChunk(
        chunk: FloatArray,
        location: LocationMeta? = null,
    ): ChunkResult {
        val processed = audioChunkProcessor.process(chunk)
            ?: return ChunkResult(detections = emptyList(), processed = false)

        val detections = _classifier.classify(processed.samples, location)
        return ChunkResult(detections = detections, processed = true)
    }

    /**
     * Streaming file analysis: decode → processChunk per chunk → aggregate.
     *
     * Uses [AudioFileDecoder.decodeChunked] — memory-efficient, keeps only one chunk buffer.
     *
     * @param confirmationCount minimum chunks a species must appear in to be confirmed.
     *   Default = 1 (suitable for files); live detection uses higher values via its own aggregator.
     * @param onProgress optional callback reporting (processed, skipped, total) chunk counts
     * @return confirmed detections after aggregation
     */
    suspend fun analyzeFile(
        context: Context,
        uri: Uri,
        location: LocationMeta? = null,
        confirmationCount: Int = 1,
        onProgress: ((processed: Int, skipped: Int, total: Int) -> Unit)? = null,
    ): List<DetectionAggregator.AggregatedDetection> = withContext(Dispatchers.IO) {
        val aggregator = DetectionAggregator.forFileAnalysis(confirmationCount = confirmationCount)
        var totalChunks = 0
        var skippedChunks = 0

        AudioFileDecoder.decodeChunked(
            context, uri,
            targetRate = _classifier.sampleRate,
            chunkSize = _classifier.samplesPerChunk,
        ) { chunkIndex, startTimeSec, chunk ->
            totalChunks++

            val processed = audioChunkProcessor.process(chunk)
            if (processed == null) {
                skippedChunks++
                aggregator.addChunkResults(null)
                onProgress?.invoke(totalChunks - skippedChunks, skippedChunks, totalChunks)
                return@decodeChunked
            }

            val detections = runBlocking {
                _classifier.classify(processed.samples, location)
            }
            aggregator.addChunkResults(detections)

            Log.d(TAG, "Chunk $chunkIndex @ %.1fs: ${detections.size} detections".format(startTimeSec))
            onProgress?.invoke(totalChunks - skippedChunks, skippedChunks, totalChunks)
        }

        val confirmed = aggregator.getConfirmedDetections()
        Log.d(TAG, "analyzeFile done: $totalChunks chunks ($skippedChunks skipped), " +
            "${confirmed.size} confirmed species")

        confirmed
    }

    data class FileAnalysisProgress(
        val processedChunks: Int,
        val totalChunks: Int,
        val currentBirds: List<DetectionAggregator.AggregatedDetection>,
        val avgChunkMs: Long,
    )

    // --- Detailed file analysis (per-chunk records, no aggregation) ---

    data class ChunkDetectionRecord(
        val chunkIndex: Int,
        val startTimeSec: Float,
        val detections: List<BirdDetection>,
    )

    data class FileAnalysisResult(
        val chunkRecords: List<ChunkDetectionRecord>,
        val chunkDurationSec: Float,
    )

    data class DetailedProgress(
        val processedChunks: Int,
        val totalChunks: Int,
        val avgChunkMs: Long,
    )

    /**
     * Parallel file analysis: Producer → Channel → N Workers → Channel → Collector.
     *
     * Creates [numWorkers]-1 additional classifiers via [classifierFactory] (the pipeline's
     * own [_classifier] is used as the first worker). All additional classifiers are closed
     * in `finally`.
     *
     * [onProgress] is called from the collector coroutine with intermediate aggregation results,
     * allowing the UI to show live detections and ETA.
     */
    suspend fun analyzeFileParallel(
        context: Context,
        uri: Uri,
        classifierFactory: ClassifierFactory,
        numWorkers: Int = 2,
        location: LocationMeta? = null,
        confirmationCount: Int = 1,
        onProgress: ((FileAnalysisProgress) -> Unit)? = null,
    ): List<DetectionAggregator.AggregatedDetection> {
        val workerClassifiers = mutableListOf<BirdClassifier>()
        try {
            // First worker reuses the pipeline's classifier; create N-1 additional copies
            for (i in 1 until numWorkers) {
                workerClassifiers.add(classifierFactory.createWorkerClassifier(_classifier))
            }
            val allClassifiers = listOf(_classifier) + workerClassifiers

            return runParallelAnalysis(
                context, uri, allClassifiers, location, confirmationCount, onProgress,
            )
        } finally {
            for (clf in workerClassifiers) {
                try { clf.close() } catch (e: Exception) {
                    Log.w(TAG, "Failed to close worker classifier", e)
                }
            }
        }
    }

    private data class IndexedChunk(
        val chunkIndex: Int,
        val startTimeSec: Float,
        val samples: FloatArray,
    )

    private data class ChunkDetections(
        val chunkIndex: Int,
        val startTimeSec: Float,
        val detections: List<BirdDetection>?,
        val classifyMs: Long,
    )

    private suspend fun runParallelAnalysis(
        context: Context,
        uri: Uri,
        classifiers: List<BirdClassifier>,
        location: LocationMeta?,
        confirmationCount: Int,
        onProgress: ((FileAnalysisProgress) -> Unit)?,
    ): List<DetectionAggregator.AggregatedDetection> = coroutineScope {
        val aggregator = DetectionAggregator.forFileAnalysis(confirmationCount = confirmationCount)
        val totalChunksCounter = AtomicInteger(0)
        val processedCounter = AtomicInteger(0)
        var totalClassifyMs = 0L

        val chunksChannel = Channel<IndexedChunk>(capacity = 8)
        val resultsChannel = Channel<ChunkDetections>(capacity = 8)

        // Producer (IO): decode audio → send raw chunks
        launch(Dispatchers.IO) {
            try {
                AudioFileDecoder.decodeChunked(
                    context, uri,
                    targetRate = _classifier.sampleRate,
                    chunkSize = _classifier.samplesPerChunk,
                ) { chunkIndex, startTimeSec, chunk ->
                    totalChunksCounter.incrementAndGet()
                    chunksChannel.trySendBlocking(
                        IndexedChunk(chunkIndex, startTimeSec, chunk)
                    ).getOrThrow()
                }
            } finally {
                chunksChannel.close()
            }
        }

        // Workers (Default): process + classify
        val workers = classifiers.map { clf ->
            launch(Dispatchers.Default) {
                for (ic in chunksChannel) {
                    val processed = audioChunkProcessor.process(ic.samples)
                    if (processed == null) {
                        resultsChannel.send(ChunkDetections(ic.chunkIndex, ic.startTimeSec, null, 0))
                        continue
                    }
                    val t0 = System.currentTimeMillis()
                    val detections = clf.classify(processed.samples, location)
                    val ms = System.currentTimeMillis() - t0
                    resultsChannel.send(ChunkDetections(ic.chunkIndex, ic.startTimeSec, detections, ms))
                }
            }
        }

        // Closer: wait for all workers → close results channel
        launch {
            workers.forEach { it.join() }
            resultsChannel.close()
        }

        // Collector (this coroutine): aggregate + report progress
        for (result in resultsChannel) {
            aggregator.addChunkResults(result.detections)
            val processed = processedCounter.incrementAndGet()
            totalClassifyMs += result.classifyMs

            val avgMs = if (processed > 0) totalClassifyMs / processed else 0L

            onProgress?.invoke(
                FileAnalysisProgress(
                    processedChunks = processed,
                    totalChunks = totalChunksCounter.get(),
                    currentBirds = aggregator.getConfirmedDetections(),
                    avgChunkMs = avgMs,
                )
            )
        }

        val confirmed = aggregator.getConfirmedDetections()
        val total = totalChunksCounter.get()
        Log.d(TAG, "analyzeFileParallel done: $total chunks, ${classifiers.size} workers, " +
            "${confirmed.size} confirmed species")

        confirmed
    }

    /**
     * Detailed file analysis: returns per-chunk detection records (no aggregation).
     *
     * Does NOT use instance fields — accepts classifier/processor as parameters.
     * Caller is responsible for closing the classifier after use.
     */
    suspend fun analyzeFileDetailed(
        context: Context,
        uri: Uri,
        classifier: BirdClassifier,
        processor: AudioChunkProcessor,
        classifierFactory: ClassifierFactory,
        numWorkers: Int = 2,
        location: LocationMeta? = null,
        onProgress: ((DetailedProgress) -> Unit)? = null,
        onChunkResult: (suspend (ChunkDetectionRecord) -> Unit)? = null,
    ): FileAnalysisResult {
        val workerClassifiers = mutableListOf<BirdClassifier>()
        try {
            for (i in 1 until numWorkers) {
                workerClassifiers.add(classifierFactory.createWorkerClassifier(classifier))
            }
            val allClassifiers = listOf(classifier) + workerClassifiers

            return runDetailedAnalysis(
                context, uri, allClassifiers, processor, classifier, location,
                onProgress, onChunkResult,
            )
        } finally {
            for (clf in workerClassifiers) {
                try { clf.close() } catch (e: Exception) {
                    Log.w(TAG, "Failed to close worker classifier", e)
                }
            }
        }
    }

    private suspend fun runDetailedAnalysis(
        context: Context,
        uri: Uri,
        classifiers: List<BirdClassifier>,
        processor: AudioChunkProcessor,
        primaryClassifier: BirdClassifier,
        location: LocationMeta?,
        onProgress: ((DetailedProgress) -> Unit)?,
        onChunkResult: (suspend (ChunkDetectionRecord) -> Unit)?,
    ): FileAnalysisResult = coroutineScope {
        val totalChunksCounter = AtomicInteger(0)
        val processedCounter = AtomicInteger(0)
        var totalClassifyMs = 0L
        val chunkRecords = mutableListOf<ChunkDetectionRecord>()

        val chunksChannel = Channel<IndexedChunk>(capacity = 8)
        val resultsChannel = Channel<ChunkDetections>(capacity = 8)

        // Producer (IO): decode audio → send raw chunks
        launch(Dispatchers.IO) {
            try {
                AudioFileDecoder.decodeChunked(
                    context, uri,
                    targetRate = primaryClassifier.sampleRate,
                    chunkSize = primaryClassifier.samplesPerChunk,
                ) { chunkIndex, startTimeSec, chunk ->
                    totalChunksCounter.incrementAndGet()
                    chunksChannel.trySendBlocking(
                        IndexedChunk(chunkIndex, startTimeSec, chunk)
                    ).getOrThrow()
                }
            } finally {
                chunksChannel.close()
            }
        }

        // Workers (Default): process + classify
        val workers = classifiers.map { clf ->
            launch(Dispatchers.Default) {
                for (ic in chunksChannel) {
                    val processed = processor.process(ic.samples)
                    if (processed == null) {
                        resultsChannel.send(ChunkDetections(ic.chunkIndex, ic.startTimeSec, null, 0))
                        continue
                    }
                    val t0 = System.currentTimeMillis()
                    val detections = clf.classify(processed.samples, location)
                    val ms = System.currentTimeMillis() - t0
                    resultsChannel.send(ChunkDetections(ic.chunkIndex, ic.startTimeSec, detections, ms))
                }
            }
        }

        // Closer: wait for all workers → close results channel
        launch {
            workers.forEach { it.join() }
            resultsChannel.close()
        }

        // Collector: collect per-chunk records + report progress
        for (result in resultsChannel) {
            val processed = processedCounter.incrementAndGet()
            totalClassifyMs += result.classifyMs

            if (result.detections != null && result.detections.isNotEmpty()) {
                val record = ChunkDetectionRecord(
                    chunkIndex = result.chunkIndex,
                    startTimeSec = result.startTimeSec,
                    detections = result.detections,
                )
                chunkRecords.add(record)
                onChunkResult?.invoke(record)
            }

            val avgMs = if (processed > 0) totalClassifyMs / processed else 0L
            onProgress?.invoke(
                DetailedProgress(
                    processedChunks = processed,
                    totalChunks = totalChunksCounter.get(),
                    avgChunkMs = avgMs,
                )
            )
        }

        val chunkDuration = primaryClassifier.chunkDurationSeconds.toFloat()
        Log.d(TAG, "analyzeFileDetailed done: ${totalChunksCounter.get()} chunks, " +
            "${classifiers.size} workers, ${chunkRecords.size} records with detections")

        FileAnalysisResult(
            chunkRecords = chunkRecords.sortedBy { it.chunkIndex },
            chunkDurationSec = chunkDuration,
        )
    }

    companion object {
        private const val TAG = "BirdDetectionPipeline"
    }
}
