package com.birdsong.analyzer.ml

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
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
    private val audioChunkProcessor: AudioChunkProcessor,
    private val classifier: BirdClassifier,
) {
    data class ChunkResult(
        val detections: List<BirdDetection>,
        val processed: Boolean,
    )

    /**
     * Process a single audio chunk: pre-filter → inference.
     *
     * @param chunk raw float32 PCM samples (expected [BirdClassifier.SAMPLES_PER_CHUNK])
     * @param location optional GPS + week-of-year for meta-model filtering
     * @return [ChunkResult] with detections; [ChunkResult.processed] = false if chunk was skipped
     */
    suspend fun processChunk(
        chunk: FloatArray,
        location: LocationMeta? = null,
    ): ChunkResult {
        val processed = audioChunkProcessor.process(chunk)
            ?: return ChunkResult(detections = emptyList(), processed = false)

        val detections = classifier.classify(processed.samples, location)
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

        AudioFileDecoder.decodeChunked(context, uri) { chunkIndex, startTimeSec, chunk ->
            totalChunks++

            val processed = audioChunkProcessor.process(chunk)
            if (processed == null) {
                skippedChunks++
                aggregator.addChunkResults(null)
                onProgress?.invoke(totalChunks - skippedChunks, skippedChunks, totalChunks)
                return@decodeChunked
            }

            val detections = runBlocking {
                classifier.classify(processed.samples, location)
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

    companion object {
        private const val TAG = "BirdDetectionPipeline"
    }
}
