package com.birdsong.analyzer.ml

interface BirdClassifier {

    val modelId: String

    /**
     * Classifies a 3-second audio chunk.
     *
     * @param audioChunk float32 PCM, 144000 samples (48 kHz × 3 s), normalized to [-1, 1]
     * @param location optional GPS + week-of-year used by the meta-model filter
     * @return detections sorted by confidence descending, filtered by threshold
     */
    suspend fun classify(
        audioChunk: FloatArray,
        location: LocationMeta? = null,
    ): List<BirdDetection>

    fun close()

    companion object {
        const val SAMPLE_RATE = 48_000
        const val CHUNK_DURATION_SECONDS = 3
        const val SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION_SECONDS  // 144 000

        val NON_BIRD_LABELS = setOf(
            "Engine", "Environmental", "Fireworks", "Gun",
            "Human vocal", "Noise", "Power tools", "Siren",
            "Apis mellifera",
        )
    }
}
