package com.birdsong.analyzer.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import android.util.Log
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.exp

class BirdNetV24Classifier(
    audioModel: ByteBuffer,
    metaModel: ByteBuffer,
    private val labels: List<Pair<String, String>>,
    private val confidenceThreshold: Float = DEFAULT_THRESHOLD,
    private val topK: Int = DEFAULT_TOP_K,
    tfliteThreads: Int = DEFAULT_NUM_THREADS,
    private val metaAlpha: Float = DEFAULT_META_ALPHA,
) : BirdClassifier {

    override val modelId: String = MODEL_ID

    private val options = Interpreter.Options().apply { numThreads = tfliteThreads }
    private val audioInterpreter = Interpreter(audioModel, options)
    private val metaInterpreter = Interpreter(metaModel, options)

    override suspend fun classify(
        audioChunk: FloatArray,
        location: LocationMeta?,
    ): List<BirdDetection> = withContext(Dispatchers.Default) {
        require(audioChunk.size == BirdClassifier.SAMPLES_PER_CHUNK) {
            "Expected ${BirdClassifier.SAMPLES_PER_CHUNK} samples, got ${audioChunk.size}"
        }

        // Log audio level for diagnostics
        var sumSq = 0.0
        var peak = 0f
        for (s in audioChunk) {
            sumSq += s * s
            val a = abs(s)
            if (a > peak) peak = a
        }
        val rms = kotlin.math.sqrt(sumSq / audioChunk.size)
        //Log.d(TAG, "Chunk RMS=%.6f peak=%.6f (dBFS=%.1f)".format(rms, peak, 20 * kotlin.math.log10(rms.coerceAtLeast(1e-10))))

        val logits = runAudioModel(audioChunk)

        // Apply sigmoid — model outputs raw logits, not probabilities
        val scores = FloatArray(logits.size) { i -> sigmoid(logits[i]) }

        // Diagnostic: log top-5 scores after sigmoid
        val top5 = scores.indices.sortedByDescending { scores[it] }.take(3)
        val top5Str = top5.joinToString { i ->
            val (sci, common) = labels[i]
            "$common(${String.format("%.3f", scores[i])})"
        }
            //Log.d(TAG, "Top-5 sigmoid: $top5Str")

        if (location != null) {
            applyMetaModel(location, scores)
        }

        buildDetections(scores)
    }

    private fun runAudioModel(audioChunk: FloatArray): FloatArray {
        val input = arrayOf(audioChunk)
        val output = Array(1) { FloatArray(labels.size) }
        audioInterpreter.run(input, output)
        return output[0]
    }

    private fun applyMetaModel(location: LocationMeta, scores: FloatArray) {
        val metaInput = arrayOf(FloatArray(3))
        val metaOutput = Array(1) { FloatArray(labels.size) }
        metaInput[0][0] = location.latitude.toFloat()
        metaInput[0][1] = location.longitude.toFloat()

        val weeks = location.weekRange ?: (location.weekOfYear..location.weekOfYear)

        // Collect raw meta scores: single week or max over range
        val rawMeta = FloatArray(labels.size)
        if (weeks.first == weeks.last) {
            metaInput[0][2] = weeks.first.toFloat()
            metaInterpreter.run(metaInput, metaOutput)
            metaOutput[0].copyInto(rawMeta)
        } else {
            // Run meta-model for each week, keep max score per species.
            // Semantics: "has this species ever been expected here?" rather than "is it here now?"
            for (week in weeks) {
                metaInput[0][2] = week.toFloat()
                metaInterpreter.run(metaInput, metaOutput)
                val weekScores = metaOutput[0]
                for (i in rawMeta.indices) {
                    if (weekScores[i] > rawMeta[i]) rawMeta[i] = weekScores[i]
                }
            }
        }

        // Blended meta: alpha + (1 - alpha) * rawMeta
        // alpha > 0 prevents complete suppression of low-eBird edge-case species
        // while still significantly downweighting continental outliers
        for (i in scores.indices) {
            scores[i] *= metaAlpha + (1f - metaAlpha) * rawMeta[i]
        }
    }

    private fun sigmoid(x: Float): Float = (1.0f / (1.0f + exp(-x)))

    private fun buildDetections(scores: FloatArray): List<BirdDetection> =
        buildDetections(scores, labels, confidenceThreshold, topK)

    override fun close() {
        audioInterpreter.close()
        metaInterpreter.close()
    }

    companion object {
        private const val TAG = "BirdNetClassifier"
        const val MODEL_ID = "BirdNET-V2.4-FP16"
        const val ASSET_BASE = "birdnet/v24"
        const val AUDIO_MODEL_PATH = "$ASSET_BASE/audio-model-fp16.tflite"
        const val META_MODEL_PATH = "$ASSET_BASE/meta-model.tflite"
        const val DEFAULT_NUM_THREADS = 2
        const val DEFAULT_THRESHOLD = 0.1f
        const val DEFAULT_TOP_K = 10
        const val DEFAULT_META_ALPHA = 0.15f

        internal fun buildDetections(
            scores: FloatArray,
            labels: List<Pair<String, String>>,
            confidenceThreshold: Float,
            topK: Int,
        ): List<BirdDetection> =
            scores.indices
                .filter { scores[it] >= confidenceThreshold }
                .sortedByDescending { scores[it] }
                .take(topK)
                .map { i ->
                    val (scientificName, commonName) = labels[i]
                    BirdDetection(
                        scientificName = scientificName,
                        commonName = commonName,
                        confidence = scores[i],
                        labelIndex = i,
                    )
                }
    }
}
