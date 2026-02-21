package com.birdsong.analyzer.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BirdNetV24ClassifierUnitTest {

    private val testLabels = listOf(
        "Parus major" to "Great Tit",
        "Turdus merula" to "Common Blackbird",
        "Corvus corax" to "Common Raven",
        "Erithacus rubecula" to "European Robin",
        "Passer domesticus" to "House Sparrow",
    )

    @Test
    fun `filters detections below threshold`() {
        val scores = floatArrayOf(0.9f, 0.3f, 0.8f, 0.1f, 0.6f)

        val detections = buildDetections(scores, threshold = 0.5f)

        assertEquals(3, detections.size)
        assertTrue(detections.all { it.confidence >= 0.5f })
        assertTrue(detections.none { it.commonName == "Common Blackbird" })
        assertTrue(detections.none { it.commonName == "European Robin" })
    }

    @Test
    fun `sorts by confidence descending`() {
        val scores = floatArrayOf(0.3f, 0.9f, 0.1f, 0.7f, 0.5f)

        val detections = buildDetections(scores, threshold = 0.0f)

        assertEquals(listOf(0.9f, 0.7f, 0.5f, 0.3f, 0.1f), detections.map { it.confidence })
    }

    @Test
    fun `maps labels to detection correctly`() {
        val scores = floatArrayOf(0.0f, 0.0f, 0.0f, 0.95f, 0.0f)

        val detections = buildDetections(scores, threshold = 0.0f)

        val top = detections.first()
        assertEquals("Erithacus rubecula", top.scientificName)
        assertEquals("European Robin", top.commonName)
        assertEquals(0.95f, top.confidence)
        assertEquals(3, top.labelIndex)
    }

    @Test
    fun `respects topK limit`() {
        val scores = floatArrayOf(0.5f, 0.9f, 0.3f, 0.8f, 0.6f)

        val detections = buildDetections(scores, threshold = 0.0f, topK = 2)

        assertEquals(2, detections.size)
        assertEquals("Common Blackbird", detections[0].commonName)
        assertEquals("European Robin", detections[1].commonName)
    }

    @Test
    fun `returns empty when all scores below threshold`() {
        val scores = floatArrayOf(0.5f, 0.9f, 0.3f, 0.8f, 0.6f)

        val detections = buildDetections(scores, threshold = 0.99f)

        assertTrue(detections.isEmpty())
    }

    @Test
    fun `meta-model score multiplication affects filtering`() {
        val audioScores = floatArrayOf(0.9f, 0.8f, 0.7f, 0.6f, 0.5f)
        val metaScores = floatArrayOf(1.0f, 0.0f, 0.5f, 0.8f, 1.0f)

        // Simulate applyMetaModel: element-wise multiplication
        for (i in audioScores.indices) {
            audioScores[i] *= metaScores[i]
        }

        val detections = buildDetections(audioScores, threshold = 0.1f)

        // 0.8 * 0.0 = 0.0 → Blackbird filtered out
        assertTrue(detections.none { it.commonName == "Common Blackbird" })
        // 0.9 * 1.0 = 0.9 → Great Tit stays on top
        assertEquals("Great Tit", detections[0].commonName)
        assertEquals(0.9f, detections[0].confidence)
        // 0.7 * 0.5 = 0.35
        val raven = detections.find { it.commonName == "Common Raven" }!!
        assertEquals(0.35f, raven.confidence, 0.001f)
    }

    private fun buildDetections(
        scores: FloatArray,
        threshold: Float = BirdNetV24Classifier.DEFAULT_THRESHOLD,
        topK: Int = BirdNetV24Classifier.DEFAULT_TOP_K,
    ): List<BirdDetection> =
        BirdNetV24Classifier.buildDetections(scores, testLabels, threshold, topK)
}
