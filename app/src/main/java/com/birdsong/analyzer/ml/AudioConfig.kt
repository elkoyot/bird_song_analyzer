package com.birdsong.analyzer.ml

data class AudioConfig(
    val sampleRate: Int,
    val samplesPerChunk: Int,
    val hopSize: Int = samplesPerChunk / 2,
) {
    val readSize: Int get() = sampleRate / 10  // ~100ms reads
}
