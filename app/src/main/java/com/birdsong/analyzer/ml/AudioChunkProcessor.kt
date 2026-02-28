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
enum class PreprocessingMode { FULL, LIGHT }

class AudioChunkProcessor(
    private val sampleRate: Int = 48_000,
    private val mode: PreprocessingMode = PreprocessingMode.FULL,
) {

    enum class SkipReason { SILENCE, CLIPPING, SPECTRAL_REJECT, POST_FILTER_SILENCE }

    data class Result(val samples: FloatArray, val rms: Float, val peak: Float)

    /* ── Diagnostic counters ── */
    var totalChunks = 0; private set
    var silenceRejects = 0; private set
    var clippingRejects = 0; private set
    var spectralRejects = 0; private set
    var postFilterRejects = 0; private set
    var passedChunks = 0; private set

    fun resetStats() {
        totalChunks = 0; silenceRejects = 0; clippingRejects = 0
        spectralRejects = 0; postFilterRejects = 0; passedChunks = 0
    }

    fun statsLine(): String =
        "total=$totalChunks passed=$passedChunks " +
            "silence=$silenceRejects clip=$clippingRejects " +
            "spectral=$spectralRejects postFilter=$postFilterRejects"

    private val bandpass by lazy { BandpassFilter(sampleRate, LOW_CUTOFF, HIGH_CUTOFF) }

    /**
     * Process a raw audio chunk. Returns [Result] with filtered+normalized samples,
     * or null if the chunk should be skipped (silence, clipping, non-bird noise).
     *
     * [PreprocessingMode.FULL] — silence + clipping + spectral + bandpass + normalization (BirdNET)
     * [PreprocessingMode.LIGHT] — silence + clipping + normalization (models with built-in STFT)
     */
    fun process(chunk: FloatArray): Result? {
        totalChunks++

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
            silenceRejects++
            Log.d(TAG, "SKIP silence: rms=%.5f peak=%.4f [%s]".format(rms, peak, statsLine()))
            return null
        }

        // 2. Clipping check
        if (peak > CLIPPING_PEAK_THRESHOLD && rms > CLIPPING_RMS_THRESHOLD) {
            clippingRejects++
            Log.d(TAG, "SKIP clipping: rms=%.4f peak=%.4f [%s]".format(rms, peak, statsLine()))
            return null
        }

        return if (mode == PreprocessingMode.FULL) processFull(chunk, rms, peak)
               else processLight(chunk, rms, peak)
    }

    private fun processFull(chunk: FloatArray, rms: Float, peak: Float): Result? {
        // 3. Spectral check via Goertzel at 4 bands
        if (!passesSpectralCheck(chunk)) {
            spectralRejects++
            Log.d(TAG, "SKIP spectral: rms=%.4f peak=%.4f [%s]".format(rms, peak, statsLine()))
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
            postFilterRejects++
            Log.d(TAG, "SKIP postFilter: postPeak=%.5f rms=%.4f [%s]"
                .format(postPeak, rms, statsLine()))
            return null
        }

        // 6. Peak normalization
        val normalized = if (postPeak in POST_FILTER_SILENCE_THRESHOLD..NORM_TARGET) {
            val gain = NORM_TARGET / postPeak
            FloatArray(filtered.size) { i -> (filtered[i] * gain).coerceIn(-1f, 1f) }
        } else {
            filtered
        }

        return finalize(normalized, rms, peak)
    }

    private fun processLight(chunk: FloatArray, rms: Float, peak: Float): Result? {
        // LIGHT mode: only peak normalization (no bandpass/spectral)
        val normalized = if (peak in POST_FILTER_SILENCE_THRESHOLD..NORM_TARGET) {
            val gain = NORM_TARGET / peak
            FloatArray(chunk.size) { i -> (chunk[i] * gain).coerceIn(-1f, 1f) }
        } else {
            chunk.copyOf()
        }

        return finalize(normalized, rms, peak)
    }

    private fun finalize(normalized: FloatArray, inRms: Float, inPeak: Float): Result {
        var outSumSq = 0.0
        var outPeak = 0f
        for (s in normalized) {
            outSumSq += s * s
            val a = abs(s)
            if (a > outPeak) outPeak = a
        }
        val outRms = sqrt(outSumSq / normalized.size).toFloat()

        passedChunks++
        Log.d(TAG, "PASS: inRms=%.4f inPeak=%.4f → outRms=%.4f outPeak=%.4f [%s]"
            .format(inRms, inPeak, outRms, outPeak, statsLine()))

        return Result(normalized, outRms, outPeak)
    }

    /**
     * Spectral check using Goertzel algorithm at 5 frequency bands.
     * Rejects chunks where ≥95% of energy is at non-bird frequencies.
     *
     * Bands:
     *   100 Hz  — non-bird noise (motors, HVAC, wind, 50/60 Hz hum harmonics)
     *   250 Hz  — low-frequency birds: owls fundamental (200-400 Hz)
     *   500 Hz  — low-frequency birds: pigeons, owl harmonics
     *   3000 Hz — typical bird vocalizations
     *   12000 Hz — above most bird song (electronics, insects)
     */
    private fun passesSpectralCheck(chunk: FloatArray): Boolean {
        val lowEnergy = goertzelEnergy(chunk, 100f)        // non-bird low-frequency noise
        val birdOwlEnergy = goertzelEnergy(chunk, 250f)    // owls: eagle-owl ~300, tawny ~440
        val birdLowEnergy = goertzelEnergy(chunk, 500f)    // pigeons, owl harmonics
        val birdMidEnergy = goertzelEnergy(chunk, 3000f)   // typical bird vocalizations
        val highEnergy = goertzelEnergy(chunk, 12000f)     // above most bird song

        val totalEnergy = lowEnergy + birdOwlEnergy + birdLowEnergy + birdMidEnergy + highEnergy
        if (totalEnergy < 1e-12) return true // negligible energy at all bands — let silence check handle it

        val lowRatio = lowEnergy / totalEnergy
        val highRatio = highEnergy / totalEnergy

        val passes = lowRatio < SPECTRAL_REJECT_RATIO && highRatio < SPECTRAL_REJECT_RATIO
        if (!passes) {
            Log.d(TAG, "Spectral detail: low100=%.2f%% owl250=%.2f%% bird500=%.2f%% bird3k=%.2f%% high12k=%.2f%%"
                .format(lowRatio * 100, birdOwlEnergy / totalEnergy * 100,
                    birdLowEnergy / totalEnergy * 100,
                    birdMidEnergy / totalEnergy * 100, highRatio * 100))
        }
        return passes
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
        const val SILENCE_RMS_THRESHOLD = 0.001f
        const val CLIPPING_PEAK_THRESHOLD = 0.99f
        const val CLIPPING_RMS_THRESHOLD = 0.3f
        const val SPECTRAL_REJECT_RATIO = 0.98
        const val LOW_CUTOFF = 80f
        const val HIGH_CUTOFF = 15_000f
        const val NORM_TARGET = 0.9f
        const val POST_FILTER_SILENCE_THRESHOLD = 0.001f
    }
}
