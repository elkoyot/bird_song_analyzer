package com.birdsong.analyzer.service

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder.AudioSource
import android.util.Log
import com.birdsong.analyzer.ml.AudioConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

/**
 * Captures microphone audio and emits overlapping float32 chunks ready for ML inference.
 *
 * Default config: 48 kHz, 144 000 samples (3 s), 50% overlap.
 * Call [configure] to switch to a different [AudioConfig] (e.g. 32 kHz / 5 s for BirdNET V3.0).
 *
 * Each emitted [FloatArray] is an independent copy — callers may hold it freely.
 *
 * Collection drives the lifecycle: recording starts when [chunksFlow] is collected
 * and stops automatically when the collector is cancelled or an error occurs.
 *
 * Audio source priority: VOICE_RECOGNITION (AGC) → UNPROCESSED (raw, no DSP).
 */
@Singleton
class AudioRecorder @Inject constructor() {

    @Volatile
    private var config = AudioConfig(
        sampleRate = DEFAULT_SAMPLE_RATE,
        samplesPerChunk = DEFAULT_SAMPLES_PER_CHUNK,
    )

    fun configure(config: AudioConfig) {
        this.config = config
        Log.i(TAG, "Configured: sampleRate=${config.sampleRate}, " +
            "chunk=${config.samplesPerChunk}, hop=${config.hopSize}, read=${config.readSize}")
    }

    private val _audioLevel = MutableStateFlow(0f)

    /** Current mic RMS level, 0..1. Updated ~10 times/sec while recording. */
    val audioLevel: StateFlow<Float> = _audioLevel.asStateFlow()

    /**
     * Try to create an [AudioRecord] with the given [source].
     * Returns initialized instance or null if the source is not supported.
     */
    private fun tryCreate(source: Int, sampleRate: Int, bufBytes: Int): AudioRecord? = try {
        val record = AudioRecord(
            source,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufBytes,
        )
        if (record.state == AudioRecord.STATE_INITIALIZED) {
            record
        } else {
            record.release()
            null
        }
    } catch (e: Exception) {
        Log.w(TAG, "AudioSource ${sourceName(source)} failed: ${e.message}")
        null
    }

    private fun sourceName(source: Int): String = when (source) {
        AudioSource.UNPROCESSED -> "UNPROCESSED"
        AudioSource.VOICE_RECOGNITION -> "VOICE_RECOGNITION"
        AudioSource.CAMCORDER -> "CAMCORDER"
        AudioSource.MIC -> "MIC"
        else -> "UNKNOWN($source)"
    }

    /**
     * Emits small raw PCM float32 portions (~100 ms each) without accumulation or overlap.
     * Callers handle their own accumulation (used by DualDetectionViewModel for dual-model inference).
     */
    fun rawSamplesFlow(sampleRate: Int = DEFAULT_SAMPLE_RATE): Flow<FloatArray> = flow {
        val readSize = sampleRate / 10  // ~100ms

        val minBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBytes <= 0) {
            error("Device does not support ${sampleRate} Hz mono PCM16 recording (code=$minBytes)")
        }

        val bufBytes = maxOf(minBytes, sampleRate * 2 * Short.SIZE_BYTES)

        val audioRecord = tryCreate(AudioSource.VOICE_RECOGNITION, sampleRate, bufBytes)
            ?: tryCreate(AudioSource.UNPROCESSED, sampleRate, bufBytes)
            ?: error("No AudioSource available — check RECORD_AUDIO permission")

        Log.i(TAG, "rawSamplesFlow: AudioSource=${sourceName(audioRecord.audioSource)}, ${sampleRate}Hz")

        val readBuf = ShortArray(readSize)

        audioRecord.startRecording()
        try {
            while (currentCoroutineContext().isActive) {
                val read = audioRecord.read(readBuf, 0, readSize)
                if (read <= 0) continue

                var sumSq = 0.0
                val floats = FloatArray(read)
                for (j in 0 until read) {
                    val s = readBuf[j] / 32_768f
                    floats[j] = s
                    sumSq += s * s
                }
                _audioLevel.value = sqrt(sumSq / read).toFloat()

                emit(floats)
            }
        } finally {
            _audioLevel.value = 0f
            audioRecord.stop()
            audioRecord.release()
        }
    }.flowOn(Dispatchers.IO)

    fun chunksFlow(): Flow<FloatArray> = flow {
        val cfg = config
        val sampleRate = cfg.sampleRate
        val samplesPerChunk = cfg.samplesPerChunk
        val hopSize = cfg.hopSize
        val readSize = cfg.readSize

        val minBytes = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        Log.d(TAG, "getMinBufferSize=$minBytes for ${sampleRate} Hz")

        if (minBytes <= 0) {
            error("Device does not support ${sampleRate} Hz mono PCM16 recording (code=$minBytes)")
        }

        val bufBytes = maxOf(minBytes, sampleRate * 2 * Short.SIZE_BYTES)

        // VOICE_RECOGNITION first: Android AGC amplifies quiet signals at distance.
        // UNPROCESSED as fallback (raw, no DSP — better if mic is close to source).
        val audioRecord = tryCreate(AudioSource.VOICE_RECOGNITION, sampleRate, bufBytes)
            ?: tryCreate(AudioSource.UNPROCESSED, sampleRate, bufBytes)
            ?: error("No AudioSource available — check RECORD_AUDIO permission")

        Log.i(TAG, "Using AudioSource: ${sourceName(audioRecord.audioSource)}")

        val accumulator = FloatArray(samplesPerChunk)
        val readBuf = ShortArray(readSize)
        var filled = 0

        audioRecord.startRecording()
        Log.d(TAG, "Recording started")
        try {
            while (currentCoroutineContext().isActive) {
                val read = audioRecord.read(readBuf, 0, readSize)
                if (read <= 0) {
                    Log.w(TAG, "AudioRecord.read returned $read")
                    continue
                }

                // Update level meter (~10 times/sec)
                var sumSq = 0.0
                for (j in 0 until read) {
                    val s = readBuf[j] / 32_768f
                    sumSq += s * s
                }
                _audioLevel.value = sqrt(sumSq / read).toFloat()

                var i = 0
                while (i < read && filled < samplesPerChunk) {
                    accumulator[filled++] = readBuf[i++] / 32_768f
                }

                if (filled == samplesPerChunk) {
                    emit(accumulator.copyOf())

                    accumulator.copyInto(
                        destination = accumulator,
                        destinationOffset = 0,
                        startIndex = hopSize,
                        endIndex = samplesPerChunk,
                    )
                    filled = samplesPerChunk - hopSize
                }
            }
        } finally {
            _audioLevel.value = 0f
            Log.d(TAG, "Stopping AudioRecord")
            audioRecord.stop()
            audioRecord.release()
        }
    }.flowOn(Dispatchers.IO)

    companion object {
        private const val TAG = "AudioRecorder"
        private const val DEFAULT_SAMPLE_RATE = 48_000
        private const val DEFAULT_SAMPLES_PER_CHUNK = DEFAULT_SAMPLE_RATE * 3  // 144 000
    }
}
