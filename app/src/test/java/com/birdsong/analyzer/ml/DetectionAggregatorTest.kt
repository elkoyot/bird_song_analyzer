package com.birdsong.analyzer.ml

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class DetectionAggregatorTest {

    private fun detection(sci: String, common: String, conf: Float) =
        BirdDetection(sci, common, conf, labelIndex = 0)

    // --- Confirmation logic ---

    @Test
    fun `single chunk does not confirm species`() {
        val agg = DetectionAggregator.forLiveDetection()
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.9f)))

        val confirmed = agg.getConfirmedDetections()
        assertTrue(confirmed.isEmpty(), "1 chunk should not confirm: ${confirmed.size}")
    }

    @Test
    fun `two chunks above threshold confirms species`() {
        val agg = DetectionAggregator.forLiveDetection()
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.8f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        val confirmed = agg.getConfirmedDetections()
        assertEquals(1, confirmed.size)
        assertEquals("Parus major", confirmed[0].scientificName)
        assertEquals(2, confirmed[0].confirmedChunks)
    }

    @Test
    fun `two chunks below threshold does not confirm`() {
        val agg = DetectionAggregator.forLiveDetection(threshold = 0.5f)
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.3f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.4f)))

        val confirmed = agg.getConfirmedDetections()
        assertTrue(confirmed.isEmpty())
    }

    @Test
    fun `species disappears after sliding window expires`() {
        val agg = DetectionAggregator.forLiveDetection(windowSize = 3, confirmationCount = 2)
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.8f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        assertEquals(1, agg.getConfirmedDetections().size, "Should be confirmed after 2 chunks")

        // 3 more empty chunks to slide the window past the detections
        agg.addChunkResults(emptyList())
        agg.addChunkResults(emptyList())
        agg.addChunkResults(emptyList())

        assertTrue(agg.getConfirmedDetections().isEmpty(), "Should expire after window slides past")
    }

    // --- Avg-top-3 confidence ---

    @Test
    fun `live mode uses average of top-3 scores`() {
        val agg = DetectionAggregator.forLiveDetection(windowSize = 8, confirmationCount = 2)
        val scores = listOf(0.9f, 0.8f, 0.7f, 0.3f, 0.2f)
        for (s in scores) {
            agg.addChunkResults(listOf(detection("Parus major", "Great Tit", s)))
        }

        val confirmed = agg.getConfirmedDetections()
        assertEquals(1, confirmed.size)

        // Top-3: 0.9, 0.8, 0.7 → avg = 0.8
        val expectedConf = (0.9f + 0.8f + 0.7f) / 3f
        assertEquals(expectedConf, confirmed[0].confidence, 0.001f)
    }

    // --- File mode (max confidence) ---

    @Test
    fun `file mode uses max confidence`() {
        val agg = DetectionAggregator.forFileAnalysis(confirmationCount = 2)
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.6f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.9f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        val confirmed = agg.getConfirmedDetections()
        assertEquals(1, confirmed.size)
        assertEquals(0.9f, confirmed[0].confidence, 0.001f)
    }

    @Test
    fun `file mode does not use sliding window`() {
        val agg = DetectionAggregator.forFileAnalysis(confirmationCount = 2)
        // Add detections spread over many chunks
        for (i in 0 until 20) {
            agg.addChunkResults(emptyList())
        }
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.8f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        // All 22 chunks still in window (Int.MAX_VALUE)
        val confirmed = agg.getConfirmedDetections()
        assertEquals(1, confirmed.size)
    }

    // --- Non-bird filtering ---

    @Test
    fun `filters out non-bird labels`() {
        val agg = DetectionAggregator.forLiveDetection()
        agg.addChunkResults(listOf(
            detection("Engine", "Engine", 0.9f),
            detection("Noise", "Noise", 0.8f),
            detection("Human vocal", "Human vocal", 0.7f),
            detection("Parus major", "Great Tit", 0.6f),
        ))
        agg.addChunkResults(listOf(
            detection("Engine", "Engine", 0.9f),
            detection("Parus major", "Great Tit", 0.7f),
        ))

        val confirmed = agg.getConfirmedDetections()
        assertTrue(confirmed.none { it.scientificName == "Engine" })
        assertTrue(confirmed.none { it.scientificName == "Noise" })
        assertTrue(confirmed.none { it.scientificName == "Human vocal" })
        assertEquals(1, confirmed.size)
        assertEquals("Great Tit", confirmed[0].commonName)
    }

    @Test
    fun `filters Apis mellifera`() {
        val agg = DetectionAggregator.forLiveDetection()
        agg.addChunkResults(listOf(detection("Apis mellifera", "Western Honey Bee", 0.9f)))
        agg.addChunkResults(listOf(detection("Apis mellifera", "Western Honey Bee", 0.8f)))

        assertTrue(agg.getConfirmedDetections().isEmpty())
    }

    // --- Multiple species ---

    @Test
    fun `tracks multiple species independently`() {
        val agg = DetectionAggregator.forLiveDetection(windowSize = 4, confirmationCount = 2)
        agg.addChunkResults(listOf(
            detection("Parus major", "Great Tit", 0.8f),
            detection("Turdus merula", "Common Blackbird", 0.7f),
        ))
        agg.addChunkResults(listOf(
            detection("Parus major", "Great Tit", 0.9f),
        ))

        val confirmed = agg.getConfirmedDetections()
        // Great Tit: 2 chunks above threshold → confirmed
        assertEquals(1, confirmed.size)
        assertEquals("Parus major", confirmed[0].scientificName)

        // Add another Blackbird detection
        agg.addChunkResults(listOf(
            detection("Turdus merula", "Common Blackbird", 0.6f),
        ))

        val confirmed2 = agg.getConfirmedDetections()
        assertEquals(2, confirmed2.size)
    }

    @Test
    fun `results sorted by confidence descending`() {
        val agg = DetectionAggregator.forLiveDetection(confirmationCount = 2)
        // Add two chunks for each species
        agg.addChunkResults(listOf(
            detection("Parus major", "Great Tit", 0.6f),
            detection("Turdus merula", "Common Blackbird", 0.9f),
        ))
        agg.addChunkResults(listOf(
            detection("Parus major", "Great Tit", 0.7f),
            detection("Turdus merula", "Common Blackbird", 0.8f),
        ))

        val confirmed = agg.getConfirmedDetections()
        assertEquals(2, confirmed.size)
        assertTrue(confirmed[0].confidence >= confirmed[1].confidence,
            "Should be sorted descending: ${confirmed[0].confidence} >= ${confirmed[1].confidence}")
    }

    // --- Reset ---

    @Test
    fun `reset clears all state`() {
        val agg = DetectionAggregator.forLiveDetection()
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.8f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.9f)))
        assertEquals(1, agg.getConfirmedDetections().size)

        agg.reset()
        assertTrue(agg.getConfirmedDetections().isEmpty(), "Reset should clear all detections")
    }

    // --- Null chunk results ---

    @Test
    fun `null detections treated as empty chunk`() {
        val agg = DetectionAggregator.forLiveDetection(windowSize = 4, confirmationCount = 2)
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.8f)))
        agg.addChunkResults(null) // skipped chunk
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        // Still 2 chunks above threshold in window of 4
        val confirmed = agg.getConfirmedDetections()
        assertEquals(1, confirmed.size)
    }

    // --- Adaptive thresholds ---

    @Test
    fun `per-species threshold override`() {
        val agg = DetectionAggregator.forLiveDetection(threshold = 0.5f)
        agg.setThresholdOverride("Parus major", 0.8f)

        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.6f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.7f)))

        // Both at 0.6 and 0.7 are below the 0.8 override threshold
        assertTrue(agg.getConfirmedDetections().isEmpty())

        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.85f)))
        agg.addChunkResults(listOf(detection("Parus major", "Great Tit", 0.9f)))

        assertEquals(1, agg.getConfirmedDetections().size)
    }
}
