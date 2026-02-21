package com.birdsong.analyzer.ml

import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

class AudioChunkProcessorTest {

    private val sampleRate = 48_000
    private val samplesPerChunk = 144_000 // 3 seconds at 48 kHz
    private val processor = AudioChunkProcessor(sampleRate)

    private fun sineChunk(freq: Float, amplitude: Float = 0.3f): FloatArray {
        return FloatArray(samplesPerChunk) { i ->
            amplitude * sin(2.0 * PI * freq / sampleRate * i).toFloat()
        }
    }

    @Test
    fun `skips silence below RMS threshold`() {
        val silence = FloatArray(samplesPerChunk) { 0.001f * (it % 2 * 2 - 1).toFloat() }
        // RMS will be ~0.001, below threshold of 0.005

        val result = processor.process(silence)
        assertNull(result, "Silence chunk should be skipped")
    }

    @Test
    fun `skips all-zero silence`() {
        val silence = FloatArray(samplesPerChunk)
        val result = processor.process(silence)
        assertNull(result, "Zero chunk should be skipped")
    }

    @Test
    fun `skips clipped signal`() {
        // Create clipped signal: peak > 0.99 and RMS > 0.3
        val clipped = FloatArray(samplesPerChunk) { i ->
            if (i % 2 == 0) 1.0f else -1.0f
        }
        // This square wave has peak=1.0 and RMS=1.0

        val result = processor.process(clipped)
        assertNull(result, "Clipped signal should be skipped")
    }

    @Test
    fun `does not skip high-peak but low-RMS signal`() {
        // Short transient with high peak but low RMS — not really clipping
        val signal = FloatArray(samplesPerChunk)
        // Put a 1 kHz bird-like sine with moderate amplitude
        for (i in signal.indices) {
            signal[i] = 0.1f * sin(2.0 * PI * 3000.0 / sampleRate * i).toFloat()
        }
        // Add a single spike
        signal[1000] = 0.995f

        // RMS will be around 0.07, well below 0.3 → should NOT trigger clipping
        val result = processor.process(signal)
        assertNotNull(result, "High peak + low RMS should not be clipped")
    }

    @Test
    fun `skips spectral reject - low frequency dominated`() {
        // 100 Hz sine — energy concentrated at non-bird frequency
        // Spectral check: Goertzel at 100 Hz detects all energy → lowRatio ≈ 1.0 → rejected
        val lowFreq = sineChunk(100f, amplitude = 0.3f)

        val result = processor.process(lowFreq)
        assertNull(result, "Low frequency dominated signal should be skipped")
    }

    @Test
    fun `passes low-frequency bird signal like pigeon`() {
        // Pigeon cooing: ~120 Hz fundamental + harmonics at 240, 480 Hz
        // None of these match the 100 Hz "non-bird" Goertzel exactly,
        // and 480 Hz is near the 500 Hz "bird-low" Goertzel → passes spectral check
        val pigeon = FloatArray(samplesPerChunk) { i ->
            val t = i.toDouble() / sampleRate
            (0.15f * sin(2.0 * PI * 120 * t) +
             0.10f * sin(2.0 * PI * 240 * t) +
             0.08f * sin(2.0 * PI * 480 * t)).toFloat()
        }

        val result = processor.process(pigeon)
        assertNotNull(result, "Pigeon-frequency signal (120+240+480 Hz) should pass")
    }

    @Test
    fun `passes owl-frequency signal`() {
        // Long-eared owl (Asio otus): calls around 300-500 Hz
        // 300 Hz is no longer in the "low/non-bird" band (moved to 100 Hz)
        val owl = sineChunk(350f, amplitude = 0.3f)

        val result = processor.process(owl)
        assertNotNull(result, "Owl-frequency signal (350 Hz) should pass")
    }

    @Test
    fun `passes bird-frequency signal`() {
        // 3 kHz sine — squarely in bird vocalization range
        val birdFreq = sineChunk(3000f, amplitude = 0.3f)

        val result = processor.process(birdFreq)
        assertNotNull(result, "Bird-frequency signal should pass")
    }

    @Test
    fun `normalizes quiet signal to target peak`() {
        // Quiet 3 kHz signal with peak ~0.05
        val quiet = sineChunk(3000f, amplitude = 0.05f)

        val result = processor.process(quiet)
        assertNotNull(result, "Quiet bird signal should pass")

        // After bandpass + normalization, peak should be close to 0.5
        var peak = 0f
        for (s in result!!.samples) {
            val a = abs(s)
            if (a > peak) peak = a
        }
        assertTrue(peak > 0.4f && peak <= 0.55f,
            "Normalized peak should be ~0.5, got $peak")
    }

    @Test
    fun `does not over-normalize loud signal`() {
        // Loud 3 kHz signal with peak ~0.8
        val loud = sineChunk(3000f, amplitude = 0.8f)

        val result = processor.process(loud)
        assertNotNull(result, "Loud bird signal should pass")

        // Peak > 0.5 should not be normalized further
        var peak = 0f
        for (s in result!!.samples) {
            val a = abs(s)
            if (a > peak) peak = a
        }
        assertTrue(peak > 0.5f, "Loud signal peak should remain >0.5, got $peak")
    }

    @Test
    fun `output size matches input size`() {
        val input = sineChunk(3000f)
        val result = processor.process(input)
        assertNotNull(result)
        assertTrue(result!!.samples.size == samplesPerChunk,
            "Output size=${result.samples.size}, expected $samplesPerChunk")
    }

    @Test
    fun `result contains valid rms and peak`() {
        val input = sineChunk(3000f, amplitude = 0.3f)
        val result = processor.process(input)
        assertNotNull(result)

        assertTrue(result!!.rms > 0f, "RMS should be positive")
        assertTrue(result.peak > 0f, "Peak should be positive")
        assertTrue(result.peak >= result.rms, "Peak should be >= RMS")
    }
}
