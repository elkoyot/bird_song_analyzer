package com.birdsong.analyzer.ml

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Computes STFT-based spectrogram columns from raw PCM audio.
 * Designed for incremental use during file analysis — call [addChunk] per audio chunk,
 * then [snapshot] or [build] for the current/final spectrogram.
 *
 * Each column has [numBins] frequency bins (low→high), values normalized to [0..1].
 */
class SpectrogramComputer(
    private val fftSize: Int = 512,
    private val hopSize: Int = 256,
    private val numBins: Int = 32,
    private val targetColumns: Int = 200,
) {
    private val columns = mutableListOf<FloatArray>()
    private val window = hanningWindow(fftSize)
    private var globalMax = 1e-10f

    fun addChunk(samples: FloatArray) {
        if (samples.isEmpty()) return
        var offset = 0
        while (offset + fftSize <= samples.size) {
            val col = computeColumn(samples, offset)
            columns.add(col)
            offset += hopSize
        }
    }

    fun snapshot(): List<FloatArray> = downsample()

    fun build(): List<FloatArray> = downsample()

    fun columnCount(): Int = columns.size

    private fun computeColumn(samples: FloatArray, offset: Int): FloatArray {
        val re = FloatArray(fftSize)
        val im = FloatArray(fftSize)
        for (i in 0 until fftSize) {
            re[i] = samples[offset + i] * window[i]
        }
        fft(re, im)

        // Power spectrum (only first half — Nyquist)
        val halfN = fftSize / 2
        val power = FloatArray(halfN)
        for (i in 0 until halfN) {
            power[i] = re[i] * re[i] + im[i] * im[i]
        }

        // Bin into numBins mel-like bands (log-spaced)
        val binned = FloatArray(numBins)
        val binSize = halfN.toFloat() / numBins
        for (b in 0 until numBins) {
            val lo = (b * binSize).toInt()
            val hi = ((b + 1) * binSize).toInt().coerceAtMost(halfN)
            var sum = 0f
            for (i in lo until hi) sum += power[i]
            val avg = if (hi > lo) sum / (hi - lo) else 0f
            // Log scale
            binned[b] = (10f * log10safe(avg + 1e-10f) + 50f).coerceAtLeast(0f)
        }

        val peak = binned.maxOrNull() ?: 0f
        if (peak > globalMax) globalMax = peak

        return binned
    }

    private fun downsample(): List<FloatArray> {
        if (columns.isEmpty()) return emptyList()
        val result = mutableListOf<FloatArray>()
        val step = columns.size.toFloat() / targetColumns.coerceAtMost(columns.size)
        val count = targetColumns.coerceAtMost(columns.size)
        val normMax = if (globalMax > 0f) globalMax else 1f

        for (i in 0 until count) {
            val srcIdx = (i * step).toInt().coerceIn(0, columns.lastIndex)
            val col = columns[srcIdx]
            val normalized = FloatArray(numBins) { (col[it] / normMax).coerceIn(0f, 1f) }
            result.add(normalized)
        }
        return result
    }

    companion object {
        private fun hanningWindow(size: Int): FloatArray {
            return FloatArray(size) { i ->
                (0.5f * (1f - cos(2.0 * PI * i / (size - 1)))).toFloat()
            }
        }

        private fun log10safe(x: Float): Float =
            (ln(x.toDouble()) / ln(10.0)).toFloat()

        /**
         * In-place Cooley-Tukey FFT. [re] and [im] must have power-of-2 length.
         */
        private fun fft(re: FloatArray, im: FloatArray) {
            val n = re.size
            // Bit-reversal permutation
            var j = 0
            for (i in 0 until n) {
                if (i < j) {
                    var tmp = re[i]; re[i] = re[j]; re[j] = tmp
                    tmp = im[i]; im[i] = im[j]; im[j] = tmp
                }
                var m = n shr 1
                while (m >= 1 && j >= m) {
                    j -= m
                    m = m shr 1
                }
                j += m
            }
            // Butterfly
            var step = 1
            while (step < n) {
                val halfStep = step
                step = step shl 1
                val angleStep = -PI / halfStep
                for (k in 0 until halfStep) {
                    val angle = k * angleStep
                    val wr = cos(angle).toFloat()
                    val wi = sin(angle).toFloat()
                    var i = k
                    while (i < n) {
                        val jj = i + halfStep
                        val tr = wr * re[jj] - wi * im[jj]
                        val ti = wr * im[jj] + wi * re[jj]
                        re[jj] = re[i] - tr
                        im[jj] = im[i] - ti
                        re[i] += tr
                        im[i] += ti
                        i += step
                    }
                }
            }
        }
    }
}
