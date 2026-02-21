package com.birdsong.analyzer.ml

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Bandpass filter implemented as two cascaded 2nd-order Butterworth biquad filters:
 * high-pass at [lowCutoff] Hz and low-pass at [highCutoff] Hz.
 *
 * Coefficients computed from the Audio EQ Cookbook (Robert Bristow-Johnson).
 * Filter state is reset per call — no state carried between chunks.
 */
class BandpassFilter(
    sampleRate: Int,
    lowCutoff: Float,
    highCutoff: Float,
) {
    private val highPass = butterworthHighPass(sampleRate.toFloat(), lowCutoff)
    private val lowPass = butterworthLowPass(sampleRate.toFloat(), highCutoff)

    fun apply(input: FloatArray): FloatArray {
        val afterHp = highPass.process(input)
        return lowPass.process(afterHp)
    }

    companion object {
        /**
         * 2nd-order Butterworth high-pass biquad.
         * Q = 1/sqrt(2) for maximally-flat Butterworth response.
         */
        internal fun butterworthHighPass(sampleRate: Float, cutoff: Float): BiquadFilter {
            val w0 = 2.0 * PI * cutoff / sampleRate
            val cosW0 = cos(w0)
            val sinW0 = sin(w0)
            val alpha = sinW0 / (2.0 * sqrt(2.0)) // Q = sqrt(2)/2

            val b0 = ((1.0 + cosW0) / 2.0).toFloat()
            val b1 = (-(1.0 + cosW0)).toFloat()
            val b2 = ((1.0 + cosW0) / 2.0).toFloat()
            val a0 = (1.0 + alpha).toFloat()
            val a1 = (-2.0 * cosW0).toFloat()
            val a2 = (1.0 - alpha).toFloat()

            return BiquadFilter(
                b0 = b0 / a0, b1 = b1 / a0, b2 = b2 / a0,
                a1 = a1 / a0, a2 = a2 / a0,
            )
        }

        /**
         * 2nd-order Butterworth low-pass biquad.
         */
        internal fun butterworthLowPass(sampleRate: Float, cutoff: Float): BiquadFilter {
            val w0 = 2.0 * PI * cutoff / sampleRate
            val cosW0 = cos(w0)
            val sinW0 = sin(w0)
            val alpha = sinW0 / (2.0 * sqrt(2.0))

            val b0 = ((1.0 - cosW0) / 2.0).toFloat()
            val b1 = (1.0 - cosW0).toFloat()
            val b2 = ((1.0 - cosW0) / 2.0).toFloat()
            val a0 = (1.0 + alpha).toFloat()
            val a1 = (-2.0 * cosW0).toFloat()
            val a2 = (1.0 - alpha).toFloat()

            return BiquadFilter(
                b0 = b0 / a0, b1 = b1 / a0, b2 = b2 / a0,
                a1 = a1 / a0, a2 = a2 / a0,
            )
        }
    }
}

/**
 * Direct Form II Transposed biquad filter.
 * State is reset at the start of each [process] call.
 */
internal class BiquadFilter(
    private val b0: Float,
    private val b1: Float,
    private val b2: Float,
    private val a1: Float,
    private val a2: Float,
) {
    fun process(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var z1 = 0f
        var z2 = 0f
        for (i in input.indices) {
            val x = input[i]
            val y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            output[i] = y
        }
        return output
    }
}
