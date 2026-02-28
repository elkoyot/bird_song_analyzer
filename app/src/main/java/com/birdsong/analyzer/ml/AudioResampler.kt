package com.birdsong.analyzer.ml

import android.util.Log

/**
 * Linear-interpolation resampler for converting audio between sample rates.
 *
 * No anti-aliasing filter — sufficient for model comparison, not for production audio.
 */
object AudioResampler {

    private const val TAG = "AudioResampler"

    fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return input
        val ratio = fromRate.toDouble() / toRate
        val outSize = (input.size / ratio).toInt()
        Log.d(TAG, "Resampling ${fromRate}Hz -> ${toRate}Hz ($outSize samples)")
        return FloatArray(outSize) { i ->
            val pos = i * ratio
            val idx = pos.toInt()
            val frac = (pos - idx).toFloat()
            if (idx + 1 < input.size) {
                input[idx] * (1 - frac) + input[idx + 1] * frac
            } else {
                input[idx.coerceIn(input.indices)]
            }
        }
    }
}
