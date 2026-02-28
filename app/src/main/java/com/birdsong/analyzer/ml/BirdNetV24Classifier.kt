package com.birdsong.analyzer.ml

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
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
    override val sampleRate: Int = 48_000
    override val chunkDurationSeconds: Int = 3

    @Volatile
    override var metaProfile: MetaProfile? = null

    private val options = Interpreter.Options().apply { numThreads = tfliteThreads }
    private val audioInterpreter = Interpreter(audioModel, options)
    private val metaInterpreter = Interpreter(metaModel, options)

    override suspend fun classify(
        audioChunk: FloatArray,
        location: LocationMeta?,
    ): List<BirdDetection> = withContext(Dispatchers.Default) {
        require(audioChunk.size == samplesPerChunk) {
            "Expected $samplesPerChunk samples, got ${audioChunk.size}"
        }

        val logits = runAudioModel(audioChunk)

        // Apply sigmoid — model outputs raw logits, not probabilities
        val scores = FloatArray(logits.size) { i -> sigmoid(logits[i]) }

        val profile = metaProfile
        when {
            profile != null  -> profile.apply(scores, metaAlpha)
            location != null -> applyMetaModel(location, scores)
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
        const val MODEL_ID = "BirdNET-V2.4-FP16"
        const val ASSET_BASE = "birdnet/v24"
        const val AUDIO_MODEL_PATH = "$ASSET_BASE/audio-model-fp16.tflite"
        const val META_MODEL_PATH = "$ASSET_BASE/meta-model.tflite"
        const val DEFAULT_NUM_THREADS = 2
        const val DEFAULT_THRESHOLD = 0.1f
        const val DEFAULT_TOP_K = 10
        const val DEFAULT_META_ALPHA = 0.10f

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
