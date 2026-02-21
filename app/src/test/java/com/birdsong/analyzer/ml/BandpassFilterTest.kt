package com.birdsong.analyzer.ml

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

class BandpassFilterTest {

    private val sampleRate = 48_000
    private val filter = BandpassFilter(sampleRate, lowCutoff = 80f, highCutoff = 15_000f)

    /** Generate a sine wave at the given frequency. */
    private fun sineWave(freq: Float, durationSamples: Int = sampleRate, amplitude: Float = 0.5f): FloatArray {
        return FloatArray(durationSamples) { i ->
            amplitude * sin(2.0 * PI * freq / sampleRate * i).toFloat()
        }
    }

    /** Compute RMS of a signal. */
    private fun rms(signal: FloatArray): Float {
        var sumSq = 0.0
        for (s in signal) sumSq += s * s
        return sqrt(sumSq / signal.size).toFloat()
    }

    @Test
    fun `passes 1 kHz signal with minimal attenuation`() {
        val input = sineWave(1000f)
        val output = filter.apply(input)

        val inputRms = rms(input)
        val outputRms = rms(output)
        val ratio = outputRms / inputRms

        // 1 kHz is well within passband — expect >90% energy pass-through
        assertTrue(ratio > 0.9f, "1 kHz passband ratio=$ratio, expected >0.9")
    }

    @Test
    fun `passes 5 kHz signal with minimal attenuation`() {
        val input = sineWave(5000f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        assertTrue(ratio > 0.9f, "5 kHz passband ratio=$ratio, expected >0.9")
    }

    @Test
    fun `attenuates 50 Hz hum significantly`() {
        val input = sineWave(50f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        // 50 Hz is below 80 Hz cutoff — 2nd order Butterworth gives ~-9 dB at 50/80=0.625
        assertTrue(ratio < 0.45f, "50 Hz should be attenuated: ratio=$ratio, expected <0.45")
    }

    @Test
    fun `attenuates 20 kHz signal`() {
        val input = sineWave(20_000f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        // 20 kHz is above 15 kHz cutoff
        assertTrue(ratio < 0.3f, "20 kHz should be attenuated: ratio=$ratio, expected <0.3")
    }

    @Test
    fun `80 Hz edge is within -3 dB`() {
        val input = sineWave(80f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        // At cutoff frequency, Butterworth should be at -3 dB ≈ 0.707
        assertTrue(ratio > 0.5f, "80 Hz cutoff ratio=$ratio, expected >0.5 (-6 dB)")
        assertTrue(ratio < 1.0f, "80 Hz cutoff ratio=$ratio, expected <1.0")
    }

    @Test
    fun `passes 150 Hz with minimal attenuation`() {
        val input = sineWave(150f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        // 150 Hz is well above 80 Hz cutoff — expect >85% pass-through
        assertTrue(ratio > 0.85f, "150 Hz passband ratio=$ratio, expected >0.85")
    }

    @Test
    fun `15 kHz edge is within -3 dB`() {
        val input = sineWave(15_000f)
        val output = filter.apply(input)

        val ratio = rms(output) / rms(input)
        assertTrue(ratio > 0.5f, "15 kHz cutoff ratio=$ratio, expected >0.5")
        assertTrue(ratio < 1.0f, "15 kHz cutoff ratio=$ratio, expected <1.0")
    }

    @Test
    fun `output length matches input length`() {
        val input = sineWave(1000f, durationSamples = 144_000)
        val output = filter.apply(input)

        assertTrue(output.size == input.size, "Output size=${output.size}, expected ${input.size}")
    }

    @Test
    fun `silence in produces silence out`() {
        val input = FloatArray(48_000) // all zeros
        val output = filter.apply(input)

        val outputRms = rms(output)
        assertTrue(outputRms < 1e-6f, "Silence should produce silence: rms=$outputRms")
    }
}
