package com.birdsong.analyzer.ml

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

data class WaveformData(
    val amplitudes: FloatArray,
    val durationSec: Float,
    val fileSizeBytes: Long,
) {
    fun toByteArray(): ByteArray {
        val bb = ByteBuffer.allocate(amplitudes.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (a in amplitudes) bb.putFloat(a)
        return bb.array()
    }

    companion object {
        fun fromByteArray(bytes: ByteArray, durationSec: Float, fileSizeBytes: Long): WaveformData {
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val amplitudes = FloatArray(bytes.size / 4) { bb.float }
            return WaveformData(amplitudes, durationSec, fileSizeBytes)
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is WaveformData) return false
        return durationSec == other.durationSec && fileSizeBytes == other.fileSizeBytes &&
            amplitudes.contentEquals(other.amplitudes)
    }

    override fun hashCode(): Int = amplitudes.contentHashCode()
}

object WaveformExtractor {

    fun extractDuration(context: Context, uri: Uri): Float {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, uri, null)
            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/") && format.containsKey(MediaFormat.KEY_DURATION)) {
                    return format.getLong(MediaFormat.KEY_DURATION) / 1_000_000f
                }
            }
        } finally {
            extractor.release()
        }
        return 0f
    }

    fun extractFileSize(context: Context, uri: Uri): Long {
        return context.contentResolver.openAssetFileDescriptor(uri, "r")?.use {
            it.length
        } ?: 0L
    }
}

/**
 * Incremental waveform builder: accumulates RMS values from raw V2.4 chunks
 * and produces a normalized waveform of [targetPoints] points.
 *
 * When [totalChunks] is provided, each chunk is placed at its correct position
 * in the output array so the waveform fills in progressively without jumping.
 */
class IncrementalWaveformBuilder(
    private val targetPoints: Int = 400,
    private val totalChunks: Int = 0,
) {
    private val rmsValues = mutableListOf<Float>()
    private var runningPeak = 0f

    /** Compute RMS of one raw PCM chunk and append to the internal list. */
    fun addChunk(samples: FloatArray) {
        if (samples.isEmpty()) return
        var sumSq = 0.0
        for (s in samples) sumSq += s.toDouble() * s
        val rms = sqrt(sumSq / samples.size).toFloat()
        rmsValues.add(rms)
        if (rms > runningPeak) runningPeak = rms
    }

    /** Current waveform snapshot. Safe to call during analysis. */
    fun snapshot(): FloatArray = downsample()

    /** Final waveform after all chunks have been added. */
    fun build(): FloatArray = downsample()

    private fun downsample(): FloatArray {
        if (rmsValues.isEmpty()) return FloatArray(targetPoints)

        val total = if (totalChunks > 0) totalChunks else rmsValues.size
        val result = FloatArray(targetPoints)
        val step = total.toFloat() / targetPoints
        val peak = if (runningPeak > 0f) runningPeak else 1f

        for (i in 0 until targetPoints) {
            val start = (i * step).toInt()
            val end = ((i + 1) * step).toInt().coerceAtMost(rmsValues.size)
            if (start >= rmsValues.size) break
            var max = 0f
            for (j in start until end) {
                if (rmsValues[j] > max) max = rmsValues[j]
            }
            result[i] = max / peak
        }

        return result
    }
}
