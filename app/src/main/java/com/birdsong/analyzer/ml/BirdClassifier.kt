package com.birdsong.analyzer.ml

interface BirdClassifier {

    var metaProfile: MetaProfile?
        get() = null
        set(_) {}

    val modelId: String
    val sampleRate: Int
    val chunkDurationSeconds: Int
    val samplesPerChunk: Int get() = sampleRate * chunkDurationSeconds

    /**
     * Returns the MetaProfile label index for a detection's label index.
     * For V2.4, this is identity (same label space).
     * For V3.0, this maps through birdnetLabelIndex.
     * Returns -1 if no mapping exists.
     */
    fun metaProfileIndex(labelIndex: Int): Int = labelIndex

    /**
     * Classifies an audio chunk.
     *
     * @param audioChunk float32 PCM, [samplesPerChunk] samples, normalized to [-1, 1]
     * @param location optional GPS + week-of-year used by the meta-model filter
     * @return detections sorted by confidence descending, filtered by threshold
     */
    suspend fun classify(
        audioChunk: FloatArray,
        location: LocationMeta? = null,
    ): List<BirdDetection>

    fun close()

    companion object {
        val NON_BIRD_LABELS = setOf(
            "Engine", "Environmental", "Fireworks", "Gun",
            "Human vocal", "Noise", "Power tools", "Siren",
            "Apis mellifera",
        )
    }
}
