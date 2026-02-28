package com.birdsong.analyzer.ml

import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * BirdClassifier implementation for BirdNET V3.0 (ONNX, FP32).
 *
 * Input:  [1, 160000] float32 (5s @ 32 kHz)
 * Output: [1, 1225]   float32 logits → sigmoid
 *
 * Non-bird filtering: classNames[i] != "Aves" → skip.
 * Geo-filter: species with V2.4 MetaProfile maxScore < [GEO_MIN_SCORE] are penalized.
 * No built-in meta-model — geo-filtering reuses V2.4's MetaProfile via [birdnetLabelIndex].
 */
class BirdNetV30Classifier(
    modelPath: String,
    private val labels: List<Pair<String, String>>,
    private val classNames: List<String>,
    private val birdnetLabelIndex: IntArray,  // V3.0 idx → V2.4 idx (-1 = no match)
    private val confidenceThreshold: Float = DEFAULT_THRESHOLD,
    private val topK: Int = DEFAULT_TOP_K,
) : BirdClassifier {

    override val modelId: String = MODEL_ID
    override val sampleRate: Int = 32_000
    override val chunkDurationSeconds: Int = 5
    // samplesPerChunk = 160_000 via default getter

    @Volatile
    override var metaProfile: MetaProfile? = null

    override fun metaProfileIndex(labelIndex: Int): Int = birdnetLabelIndex[labelIndex]

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath)

    init {
        Log.i(TAG, "ONNX session created: ${labels.size} labels, " +
            "inputs=${session.inputNames}, outputs=${session.outputNames}")
    }

    override suspend fun classify(
        audioChunk: FloatArray,
        location: LocationMeta?,
    ): List<BirdDetection> = withContext(Dispatchers.Default) {
        require(audioChunk.size == samplesPerChunk) {
            "Expected $samplesPerChunk samples, got ${audioChunk.size}"
        }

        // V3.0 "predictions" output is already sigmoid probabilities in [0,1]
        val scores = runModel(audioChunk)

        // Apply geo-filter via V2.4 MetaProfile
        applyGeoFilter(scores)

        buildDetections(scores)
    }

    private fun runModel(audioChunk: FloatArray): FloatArray {
        val inputShape = longArrayOf(1, audioChunk.size.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(audioChunk), inputShape)

        val inputName = session.inputNames.first()
        val results = inputTensor.use { tensor ->
            session.run(mapOf(inputName to tensor))
        }

        val outputTensor = results.get("predictions").get() as OnnxTensor
        val output = (outputTensor.value as Array<FloatArray>)[0]
        results.close()
        return output
    }

    /**
     * Geo-filter: penalize species with low MetaProfile scores.
     * Species not mapped to V2.4 label space pass through unpenalized.
     */
    private fun applyGeoFilter(scores: FloatArray) {
        val profile = metaProfile ?: return
        var penalized = 0
        for (i in scores.indices) {
            val v24Idx = birdnetLabelIndex[i]
            if (v24Idx < 0) continue  // not in V2.4 → no data to filter
            if (profile.maxScores[v24Idx] < GEO_MIN_SCORE) {
                scores[i] *= GEO_PENALTY
                penalized++
            }
        }
        if (penalized > 0) {
            Log.d(TAG, "Geo-filter: penalized $penalized species")
        }
    }

    private fun buildDetections(scores: FloatArray): List<BirdDetection> =
        scores.indices
            .filter { i ->
                scores[i] >= confidenceThreshold && classNames[i] == "Aves"
            }
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

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "BirdNetV30"
        const val MODEL_ID = "BirdNET-V3.0-EUNA"
        const val MODEL_FILENAME = "birdnet_v30_euna.onnx"
        const val DEFAULT_THRESHOLD = 0.1f
        const val DEFAULT_TOP_K = 10
        private const val GEO_MIN_SCORE = 0.03f
        private const val GEO_PENALTY = 0.1f
    }
}
