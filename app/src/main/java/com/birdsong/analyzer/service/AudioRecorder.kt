package com.birdsong.analyzer.service

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder.AudioSource
import android.util.Log
import com.birdsong.analyzer.ml.BirdClassifier
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
 * Captures microphone audio and emits overlapping float32 chunks ready for BirdNET inference.
 *
 * Chunk size : 144 000 samples = 3 seconds at 48 kHz
 * Hop size   :  72 000 samples = 1.5 seconds (50 % overlap)
 *
 * Each emitted [FloatArray] is an independent copy — callers may hold it freely.
 *
 * Collection drives the lifecycle: recording starts when [chunksFlow] is collected
 * and stops automatically when the collector is cancelled or an error occurs.
 */
@Singleton
class AudioRecorder @Inject constructor() {

    private val _audioLevel = MutableStateFlow(0f)

    /** Current mic RMS level, 0..1. Updated ~10 times/sec while recording. */
    val audioLevel: StateFlow<Float> = _audioLevel.asStateFlow()

    fun chunksFlow(): Flow<FloatArray> = flow {
        val minBytes = AudioRecord.getMinBufferSize(
            BirdClassifier.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        Log.d(TAG, "getMinBufferSize=${minBytes} for ${BirdClassifier.SAMPLE_RATE} Hz")

        if (minBytes <= 0) {
            error("Device does not support ${BirdClassifier.SAMPLE_RATE} Hz mono PCM16 recording (code=$minBytes)")
        }

        val bufBytes = maxOf(minBytes, BirdClassifier.SAMPLE_RATE * 2 * Short.SIZE_BYTES)
        val audioRecord = AudioRecord(
            AudioSource.VOICE_RECOGNITION,
            BirdClassifier.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufBytes,
        )
        Log.d(TAG, "AudioRecord state=${audioRecord.state} (need ${AudioRecord.STATE_INITIALIZED})")

        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            error("AudioRecord failed to initialize (state=${audioRecord.state}) — check RECORD_AUDIO permission")
        }

        val accumulator = FloatArray(SAMPLES_PER_CHUNK)
        val readBuf = ShortArray(READ_SIZE)
        var filled = 0

        audioRecord.startRecording()
        Log.d(TAG, "Recording started")
        try {
            while (currentCoroutineContext().isActive) {
                val read = audioRecord.read(readBuf, 0, READ_SIZE)
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
                while (i < read && filled < SAMPLES_PER_CHUNK) {
                    accumulator[filled++] = readBuf[i++] / 32_768f
                }

                if (filled == SAMPLES_PER_CHUNK) {
                    emit(accumulator.copyOf())

                    accumulator.copyInto(
                        destination = accumulator,
                        destinationOffset = 0,
                        startIndex = HOP_SIZE,
                        endIndex = SAMPLES_PER_CHUNK,
                    )
                    filled = SAMPLES_PER_CHUNK - HOP_SIZE
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
        const val SAMPLES_PER_CHUNK = BirdClassifier.SAMPLES_PER_CHUNK
        const val HOP_SIZE = SAMPLES_PER_CHUNK / 2
        const val READ_SIZE = BirdClassifier.SAMPLE_RATE / 10
    }
}
