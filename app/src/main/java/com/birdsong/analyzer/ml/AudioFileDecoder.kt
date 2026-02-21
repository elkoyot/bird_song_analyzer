package com.birdsong.analyzer.ml

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Decodes audio files to raw float32 mono PCM at [BirdClassifier.SAMPLE_RATE].
 */
object AudioFileDecoder {

    private const val TAG = "AudioFileDecoder"
    private const val TIMEOUT_US = 10_000L

    /**
     * Decodes any audio file via content URI (WAV, MP3, OGG, FLAC, etc.).
     * Uses MediaExtractor + MediaCodec.
     *
     * WARNING: loads the entire file into memory — not suitable for files longer than ~1 min.
     * For long files use [decodeChunked].
     */
    fun decode(context: Context, uri: Uri): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        return decodeWithMediaCodec(extractor)
    }

    /**
     * Decodes audio from URI in a streaming fashion, emitting fixed-size chunks
     * with configurable overlap. Memory-efficient: keeps only one chunk buffer (~576 KB).
     *
     * @param chunkSize samples per chunk at target rate (default: [BirdClassifier.SAMPLES_PER_CHUNK])
     * @param hopSize samples to advance between chunks (default: chunkSize / 2 = 50% overlap)
     * @param onChunk callback invoked for each complete chunk (index, startTimeSec, samples)
     */
    fun decodeChunked(
        context: Context,
        uri: Uri,
        chunkSize: Int = BirdClassifier.SAMPLES_PER_CHUNK,
        hopSize: Int = chunkSize / 2,
        onChunk: (chunkIndex: Int, startTimeSec: Float, chunk: FloatArray) -> Unit,
    ) {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)

        val audioTrackIndex = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                ?.startsWith("audio/") == true
        } ?: error("No audio track found")

        extractor.selectTrack(audioTrackIndex)
        val format = extractor.getTrackFormat(audioTrackIndex)
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val targetRate = BirdClassifier.SAMPLE_RATE
        val needsResample = sampleRate != targetRate
        val expectedDurationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
            format.getLong(MediaFormat.KEY_DURATION)
        } else -1L

        Log.i(TAG, "decodeChunked: $mime, ${sampleRate}Hz, ${channels}ch → " +
            "${targetRate}Hz mono, chunk=$chunkSize, hop=$hopSize, resample=$needsResample")
        Log.i(TAG, "Input format: $format")
        Log.i(TAG, "Expected duration: ${expectedDurationUs / 1_000_000.0}s")

        // Sliding window buffer at target sample rate
        val buffer = FloatArray(chunkSize)
        var filled = 0
        var chunkIndex = 0

        fun emitChunk() {
            // Log stats for first chunk
            if (chunkIndex == 0) {
                var peak = 0f
                var sumSq = 0.0
                for (s in buffer) { sumSq += s * s; val a = abs(s); if (a > peak) peak = a }
                val rms = sqrt(sumSq / buffer.size)
                Log.i(TAG, "First chunk stats: RMS=%.6f, peak=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]"
                    .format(rms, peak, buffer[0], buffer[1], buffer[2], buffer[3], buffer[4]))
            }
            val startTimeSec = (chunkIndex.toLong() * hopSize).toFloat() / targetRate
            onChunk(chunkIndex, startTimeSec, buffer.copyOf())
            chunkIndex++
            buffer.copyInto(buffer, 0, hopSize, chunkSize)
            filled = chunkSize - hopSize
        }

        // Resampling: collect native-rate samples in batches of ~1 second
        val nativeBatch = if (needsResample) FloatArray(sampleRate) else null
        var nativeFilled = 0

        fun flushNativeBatch() {
            if (nativeBatch == null || nativeFilled == 0) return
            val resampled = resample(nativeBatch.copyOf(nativeFilled), sampleRate, targetRate)
            nativeFilled = 0
            for (s in resampled) {
                buffer[filled++] = s
                if (filled == chunkSize) emitChunk()
            }
        }

        // Force int16 PCM output — prevents float32 PCM issues on Android 10+
        // where AAC/MP3 decoders may default to float output
        format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        // Diagnostics
        var outputBufferCount = 0
        var totalMonoSamples = 0L

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val idx = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (idx >= 0) {
                        val buf = codec.getInputBuffer(idx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(
                                idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(idx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)

                if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val outFormat = codec.outputFormat
                    val pcmEnc = if (outFormat.containsKey(MediaFormat.KEY_PCM_ENCODING))
                        outFormat.getInteger(MediaFormat.KEY_PCM_ENCODING) else -1
                    Log.i(TAG, "Output format: $outFormat (pcmEncoding=$pcmEnc)")
                } else if (outIdx >= 0) {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }

                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.order(ByteOrder.LITTLE_ENDIAN)
                    outputBufferCount++

                    // Int16 PCM (forced via KEY_PCM_ENCODING): 2 bytes per sample
                    val shortBuf = outBuf.asShortBuffer()
                    while (shortBuf.remaining() >= channels) {
                        val monoSample = if (channels > 1) {
                            var sum = 0f
                            for (ch in 0 until channels) sum += shortBuf.get()
                            sum / channels / 32_768f
                        } else {
                            shortBuf.get() / 32_768f
                        }
                        totalMonoSamples++

                        if (needsResample) {
                            nativeBatch!![nativeFilled++] = monoSample
                            if (nativeFilled == nativeBatch.size) flushNativeBatch()
                        } else {
                            buffer[filled++] = monoSample
                            if (filled == chunkSize) emitChunk()
                        }
                    }

                    // Log first output buffer diagnostics
                    if (outputBufferCount == 1) {
                        Log.i(TAG, "First output buffer: ${bufferInfo.size} bytes, " +
                            "offset=${bufferInfo.offset}, shorts=${bufferInfo.size / 2}, " +
                            "monoSamples=$totalMonoSamples")
                    }

                    codec.releaseOutputBuffer(outIdx, false)
                }
            }

            // Flush remaining resampling batch
            flushNativeBatch()
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }

        val decodedDurationSec = totalMonoSamples.toFloat() / sampleRate
        Log.i(TAG, "decodeChunked complete: $outputBufferCount output buffers, " +
            "$totalMonoSamples mono samples (${decodedDurationSec}s at ${sampleRate}Hz), " +
            "$chunkIndex chunks emitted")

        if (expectedDurationUs > 0) {
            val expectedSec = expectedDurationUs / 1_000_000.0f
            val ratio = decodedDurationSec / expectedSec
            Log.i(TAG, "Duration check: expected=${expectedSec}s, decoded=${decodedDurationSec}s, " +
                "ratio=%.2f".format(ratio))
            if (ratio > 1.5f || ratio < 0.5f) {
                Log.w(TAG, "WARNING: Decoded duration differs from expected by %.0fx! ".format(ratio) +
                    "Possible sample rate mismatch or float PCM issue.")
            }
        }
    }

    /**
     * Reads a PCM WAV file from assets. Simple and reliable — no MediaCodec needed.
     */
    fun decodeFromAssets(context: Context, assetPath: String): FloatArray {
        return context.assets.open(assetPath).use { stream -> readPcmWav(stream) }
    }

    private fun readPcmWav(input: InputStream): FloatArray {
        val header = ByteArray(44)
        var read = 0
        while (read < 44) {
            val n = input.read(header, read, 44 - read)
            if (n < 0) break
            read += n
        }
        val buf = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)

        buf.position(22)
        val channels = buf.short.toInt()
        val sampleRate = buf.int

        buf.position(34)
        val bitsPerSample = buf.short.toInt()

        buf.position(40)
        val dataSize = buf.int

        Log.d(TAG, "WAV: ${sampleRate}Hz, ${channels}ch, ${bitsPerSample}bit, ${dataSize} bytes")
        require(bitsPerSample == 16) { "Only 16-bit WAV supported, got $bitsPerSample" }

        val data = ByteArray(dataSize)
        var totalRead = 0
        while (totalRead < dataSize) {
            val n = input.read(data, totalRead, dataSize - totalRead)
            if (n < 0) break
            totalRead += n
        }

        val dataBuf = ByteBuffer.wrap(data, 0, totalRead).order(ByteOrder.LITTLE_ENDIAN)
        val shortBuf = dataBuf.asShortBuffer()
        val numSamples = totalRead / (channels * 2)

        val mono = if (channels > 1) {
            FloatArray(numSamples) { i ->
                var sum = 0f
                for (ch in 0 until channels) sum += shortBuf.get(i * channels + ch)
                sum / channels / 32_768f
            }
        } else {
            FloatArray(numSamples) { i -> shortBuf.get(i) / 32_768f }
        }

        Log.d(TAG, "WAV decoded: ${mono.size} samples at ${sampleRate}Hz (%.1fs)".format(mono.size.toFloat() / sampleRate))

        return if (sampleRate != BirdClassifier.SAMPLE_RATE) {
            resample(mono, sampleRate, BirdClassifier.SAMPLE_RATE)
        } else {
            mono
        }
    }

    private fun decodeWithMediaCodec(extractor: MediaExtractor): FloatArray {
        val audioTrackIndex = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                ?.startsWith("audio/") == true
        } ?: error("No audio track found")

        extractor.selectTrack(audioTrackIndex)
        val format = extractor.getTrackFormat(audioTrackIndex)
        val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val mime = format.getString(MediaFormat.KEY_MIME)!!

        Log.d(TAG, "MediaCodec decoding: $mime, ${sampleRate}Hz, ${channels}ch")

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmShorts = mutableListOf<Short>()
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val idx = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (idx >= 0) {
                        val buf = codec.getInputBuffer(idx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(idx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                if (outIdx >= 0) {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                    val outBuf = codec.getOutputBuffer(outIdx)!!
                    outBuf.order(ByteOrder.LITTLE_ENDIAN)
                    val shortBuf = outBuf.asShortBuffer()
                    while (shortBuf.hasRemaining()) {
                        pcmShorts.add(shortBuf.get())
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }

        val mono = if (channels > 1) {
            FloatArray(pcmShorts.size / channels) { i ->
                var sum = 0f
                for (ch in 0 until channels) sum += pcmShorts[i * channels + ch]
                sum / channels / 32_768f
            }
        } else {
            FloatArray(pcmShorts.size) { i -> pcmShorts[i] / 32_768f }
        }

        Log.d(TAG, "Decoded ${mono.size} samples at ${sampleRate}Hz (%.1fs)".format(mono.size.toFloat() / sampleRate))

        return if (sampleRate != BirdClassifier.SAMPLE_RATE) {
            resample(mono, sampleRate, BirdClassifier.SAMPLE_RATE)
        } else {
            mono
        }
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
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
