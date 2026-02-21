package com.birdsong.analyzer.ml

/**
 * Aggregates bird detections over a sliding window of chunks.
 *
 * Supports two modes:
 * - **Live** (default): sliding window of [windowSize] chunks, confidence = average of top-3 scores
 * - **File**: unlimited window (all chunks), confidence = max score
 *
 * Confirmation: a species is confirmed when ≥ [confirmationCount] scores in the window
 * meet the species threshold ([defaultThreshold] or per-species override).
 *
 * Non-bird labels (Engine, Noise, Human vocal, etc.) are filtered out automatically.
 */
class DetectionAggregator(
    private val windowSize: Int = DEFAULT_WINDOW_SIZE,
    private val confirmationCount: Int = DEFAULT_CONFIRMATION_COUNT,
    private val defaultThreshold: Float = DEFAULT_THRESHOLD,
    private val useAvgTop3: Boolean = true,
) {
    data class AggregatedDetection(
        val scientificName: String,
        val commonName: String,
        val confidence: Float,
        val confirmedChunks: Int,
    )

    /** species scientificName → rolling window of confidence scores */
    private val speciesWindows = LinkedHashMap<String, ArrayDeque<Float>>()

    /** species scientificName → common name (latest seen) */
    private val speciesNames = HashMap<String, String>()

    /** Per-species threshold overrides for adaptive thresholds */
    private val thresholdOverrides = HashMap<String, Float>()

    private var chunkCount = 0

    fun addChunkResults(detections: List<BirdDetection>?) {
        chunkCount++
        val validDetections = detections
            ?.filter { !isNonBirdLabel(it.scientificName) && !isNonBirdLabel(it.commonName) }
            ?: emptyList()

        val detectedSpecies = validDetections.map { it.scientificName }.toSet()

        // Add confidence for detected species
        for (det in validDetections) {
            val window = speciesWindows.getOrPut(det.scientificName) { ArrayDeque() }
            window.addLast(det.confidence)
            speciesNames[det.scientificName] = det.commonName
            trimWindow(window)
        }

        // Add 0.0 for tracked-but-not-detected species
        for ((species, window) in speciesWindows) {
            if (species !in detectedSpecies) {
                window.addLast(0f)
                trimWindow(window)
            }
        }

        // Clean up species with all-zero windows (only in windowed mode)
        if (windowSize != Int.MAX_VALUE) {
            val toRemove = speciesWindows.entries
                .filter { (_, window) -> window.all { it == 0f } }
                .map { it.key }
            for (key in toRemove) {
                speciesWindows.remove(key)
                speciesNames.remove(key)
            }
        }
    }

    fun getConfirmedDetections(): List<AggregatedDetection> {
        val result = mutableListOf<AggregatedDetection>()

        for ((species, window) in speciesWindows) {
            val threshold = thresholdOverrides[species] ?: defaultThreshold
            val aboveThreshold = window.count { it >= threshold }

            if (aboveThreshold >= confirmationCount) {
                val confidence = if (useAvgTop3) {
                    averageTop3(window)
                } else {
                    window.max()
                }
                val commonName = speciesNames[species] ?: species
                result.add(AggregatedDetection(species, commonName, confidence, aboveThreshold))
            }
        }

        return result.sortedByDescending { it.confidence }
    }

    fun setThresholdOverride(scientificName: String, threshold: Float) {
        thresholdOverrides[scientificName] = threshold
    }

    fun reset() {
        speciesWindows.clear()
        speciesNames.clear()
        chunkCount = 0
    }

    private fun trimWindow(window: ArrayDeque<Float>) {
        while (window.size > windowSize) {
            window.removeFirst()
        }
    }

    private fun averageTop3(window: ArrayDeque<Float>): Float {
        val sorted = window.sortedDescending()
        val top = sorted.take(3)
        return if (top.isEmpty()) 0f else top.sum() / top.size
    }

    companion object {
        const val DEFAULT_WINDOW_SIZE = 8
        const val DEFAULT_CONFIRMATION_COUNT = 2
        const val DEFAULT_THRESHOLD = 0.5f

        private fun isNonBirdLabel(label: String): Boolean =
            label in BirdClassifier.NON_BIRD_LABELS

        /** Create aggregator for live detection (sliding window, avg-top-3). */
        fun forLiveDetection(
            windowSize: Int = DEFAULT_WINDOW_SIZE,
            confirmationCount: Int = DEFAULT_CONFIRMATION_COUNT,
            threshold: Float = DEFAULT_THRESHOLD,
        ) = DetectionAggregator(
            windowSize = windowSize,
            confirmationCount = confirmationCount,
            defaultThreshold = threshold,
            useAvgTop3 = true,
        )

        /** Create aggregator for file analysis (unlimited window, max confidence). */
        fun forFileAnalysis(
            confirmationCount: Int = DEFAULT_CONFIRMATION_COUNT,
            threshold: Float = DEFAULT_THRESHOLD,
        ) = DetectionAggregator(
            windowSize = Int.MAX_VALUE,
            confirmationCount = confirmationCount,
            defaultThreshold = threshold,
            useAvgTop3 = false,
        )
    }
}
