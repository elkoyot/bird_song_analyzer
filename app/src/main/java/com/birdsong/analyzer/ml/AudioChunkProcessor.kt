package com.birdsong.analyzer.ml

import android.util.Log
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Stateless audio pre-processor applied before ML inference.
 *
 * Pipeline:
 * 1. RMS silence check — skip if chunk is too quiet
 * 2. Clipping check — skip if signal is saturated
 * 3. Spectral check via Goertzel — skip wind/electronics (energy concentrated outside bird range)
 * 4. Bandpass filter (Butterworth biquad 80 Hz – 15 kHz)
 * 5. Post-filter silence check
 * 6. Peak normalization to [NORM_TARGET]
 */
class AudioChunkProcessor(private val sampleRate: Int = BirdClassifier.SAMPLE_RATE) {

    enum class SkipReason { SILENCE, CLIPPING, SPECTRAL_REJECT, POST_FILTER_SILENCE }

    data class Result(val samples: FloatArray, val rms: Float, val peak: Float)

    private val bandpass = BandpassFilter(sampleRate, LOW_CUTOFF, HIGH_CUTOFF)

    /**
     * Process a raw audio chunk. Returns [Result] with filtered+normalized samples,
     * or null if the chunk should be skipped (silence, clipping, non-bird noise).
     */
    fun process(chunk: FloatArray): Result? {
        // 1. Compute RMS and peak
        var sumSq = 0.0
        var peak = 0f
        for (s in chunk) {
            sumSq += s * s
            val a = abs(s)
            if (a > peak) peak = a
        }
        val rms = sqrt(sumSq / chunk.size).toFloat()

        // 1a. Silence check
        if (rms < SILENCE_RMS_THRESHOLD) {
            Log.d(TAG, "SKIP: silence (RMS=%.6f < %.4f)".format(rms, SILENCE_RMS_THRESHOLD))
            return null
        }

        // 2. Clipping check
        if (peak > CLIPPING_PEAK_THRESHOLD && rms > CLIPPING_RMS_THRESHOLD) {
            Log.d(TAG, "SKIP: clipping (peak=%.4f rms=%.4f)".format(peak, rms))
            return null
        }

        // 3. Spectral check via Goertzel at 4 bands
        if (!passesSpectralCheck(chunk)) {
            Log.d(TAG, "SKIP: spectral reject (>80%% energy outside bird range)")
            return null
        }

        // 4. Bandpass filter
        val filtered = bandpass.apply(chunk)

        // 5. Post-filter silence check
        var postPeak = 0f
        for (s in filtered) {
            val a = abs(s)
            if (a > postPeak) postPeak = a
        }
        if (postPeak < POST_FILTER_SILENCE_THRESHOLD) {
            Log.d(TAG, "SKIP: post-filter silence (peak=%.6f)".format(postPeak))
            return null
        }

        // 6. Peak normalization
        val normalized = if (postPeak in POST_FILTER_SILENCE_THRESHOLD..NORM_TARGET) {
            val gain = NORM_TARGET / postPeak
            FloatArray(filtered.size) { i -> (filtered[i] * gain).coerceIn(-1f, 1f) }
        } else {
            filtered
        }

        // Compute output stats
        var outSumSq = 0.0
        var outPeak = 0f
        for (s in normalized) {
            outSumSq += s * s
            val a = abs(s)
            if (a > outPeak) outPeak = a
        }
        val outRms = sqrt(outSumSq / normalized.size).toFloat()

        Log.d(TAG, "PASS: rms=%.4f peak=%.4f → postBP_peak=%.4f → outRms=%.4f outPeak=%.4f".format(
            rms, peak, postPeak, outRms, outPeak,
        ))

        return Result(normalized, outRms, outPeak)
    }

    /**
     * Spectral check using Goertzel algorithm at 4 frequency bands.
     * Rejects chunks where >80% of energy is at non-bird frequencies
     * (below ~200 Hz or above ~10 kHz).
     *
     * Bands:
     *   100 Hz  — non-bird noise (motors, HVAC, wind, 50/60 Hz hum harmonics)
     *   500 Hz  — low-frequency birds (pigeons ~120 Hz harmonics, owls ~300-500 Hz)
     *   3000 Hz — typical bird vocalizations
     *   12000 Hz — above most bird song (electronics, insects)
     */
    private fun passesSpectralCheck(chunk: FloatArray): Boolean {
        val lowEnergy = goertzelEnergy(chunk, 100f)       // non-bird low-frequency noise
        val birdLowEnergy = goertzelEnergy(chunk, 500f)   // low-freq birds: pigeons, owls
        val birdMidEnergy = goertzelEnergy(chunk, 3000f)  // typical bird vocalizations
        val highEnergy = goertzelEnergy(chunk, 12000f)    // above most bird song

        val totalEnergy = lowEnergy + birdLowEnergy + birdMidEnergy + highEnergy
        if (totalEnergy < 1e-12) return true // negligible energy at all bands — let silence check handle it

        val lowRatio = lowEnergy / totalEnergy
        val highRatio = highEnergy / totalEnergy

        return lowRatio < SPECTRAL_REJECT_RATIO && highRatio < SPECTRAL_REJECT_RATIO
    }

    /**
     * Goertzel algorithm — computes energy at a single target frequency.
     * O(N) per frequency, much cheaper than full FFT for a few bands.
     */
    private fun goertzelEnergy(samples: FloatArray, targetFreq: Float): Double {
        val k = (0.5 + samples.size.toDouble() * targetFreq / sampleRate).toInt()
        val w = 2.0 * Math.PI * k / samples.size
        val coeff = 2.0 * cos(w)

        var s0 = 0.0
        var s1 = 0.0
        var s2 = 0.0

        for (sample in samples) {
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
        }

        return s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    companion object {
        private const val TAG = "AudioChunkProcessor"

        const val SILENCE_RMS_THRESHOLD = 0.005f
        const val CLIPPING_PEAK_THRESHOLD = 0.99f
        const val CLIPPING_RMS_THRESHOLD = 0.3f
        const val SPECTRAL_REJECT_RATIO = 0.80
        const val LOW_CUTOFF = 80f
        const val HIGH_CUTOFF = 15_000f
        const val NORM_TARGET = 0.5f
        const val POST_FILTER_SILENCE_THRESHOLD = 0.001f
    }
}
