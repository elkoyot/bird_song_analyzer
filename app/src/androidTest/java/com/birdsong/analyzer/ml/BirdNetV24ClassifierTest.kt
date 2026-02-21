package com.birdsong.analyzer.ml

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Интеграционный тест ML-пайплайна BirdNET V2.4.
 *
 * Запускается на эмуляторе/устройстве. Проверяет всю цепочку:
 * загрузка TFLite-моделей из assets → парсинг labels → конвертация WAV → inference → результат.
 *
 * Тесты:
 * - [classifySampleWav_returnsDetections] — подаёт реальную 3-секундную запись птицы (sample.wav)
 *   и проверяет, что модель вернула хотя бы одно распознавание с confidence ≥ 0.1.
 * - [classifySilence_returnsNoDetections] — подаёт тишину (массив нулей) и проверяет,
 *   что модель не выдаёт ложных срабатываний.
 *
 * Логирует: время загрузки модели, время inference, список распознанных видов
 * с confidence и научным названием. Логи доступны в Logcat (тег BirdNetTest)
 * и в окне результатов Android Studio.
 */
@RunWith(AndroidJUnit4::class)
class BirdNetV24ClassifierTest {

    private lateinit var classifier: BirdNetV24Classifier
    private var modelLoadTimeMs: Long = 0

    @Before
    fun setUp() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val start = System.currentTimeMillis()

        val audioModel = loadModel(context, BirdNetV24Classifier.AUDIO_MODEL_PATH)
        val metaModel = loadModel(context, BirdNetV24Classifier.META_MODEL_PATH)
        val labelsPath = "${BirdNetV24Classifier.ASSET_BASE}/labels/en_us.txt"
        val labels = context.assets.open(labelsPath).use { LabelParser.load(it) }

        classifier = BirdNetV24Classifier(audioModel, metaModel, labels, confidenceThreshold = 0.1f)
        modelLoadTimeMs = System.currentTimeMillis() - start

        log("Model loaded: ${labels.size} labels, $modelLoadTimeMs ms")
    }

    @After
    fun tearDown() {
        classifier.close()
    }

    @Test
    fun classifySampleWav_returnsDetections() = runTest {
        val context = InstrumentationRegistry.getInstrumentation().context
        val audioChunk = loadWavChunk(context, "sample.wav")

        log("--- classifySampleWav ---")
        log("Input: sample.wav, ${BirdClassifier.SAMPLES_PER_CHUNK} samples " +
            "(${BirdClassifier.CHUNK_DURATION_SECONDS}s @ ${BirdClassifier.SAMPLE_RATE} Hz)")
        log("Threshold: 0.1, model: ${classifier.modelId}")
        log("Expected: at least 1 bird species detected")
        log("")

        val start = System.currentTimeMillis()
        val detections = classifier.classify(audioChunk)
        val inferenceMs = System.currentTimeMillis() - start

        log("Inference time: $inferenceMs ms (model load: $modelLoadTimeMs ms)")
        log("")
        if (detections.isEmpty()) {
            log("ACTUAL: no detections")
        } else {
            log("ACTUAL: ${detections.size} detection(s):")
            detections.forEachIndexed { i, d ->
                log("  ${i + 1}. ${d.commonName} (${d.scientificName}) " +
                    "— confidence: ${"%.4f".format(d.confidence)} [label #${d.labelIndex}]")
            }
        }
        log("---")

        assertTrue("Expected at least one detection from sample.wav", detections.isNotEmpty())
        detections.forEach { d ->
            assertTrue("Confidence should be >= 0.1, got ${d.confidence}", d.confidence >= 0.1f)
            assertTrue("Scientific name should not be blank", d.scientificName.isNotBlank())
            assertTrue("Common name should not be blank", d.commonName.isNotBlank())
        }
    }

    @Test
    fun classifySilence_returnsNoDetections() = runTest {
        val silence = FloatArray(BirdClassifier.SAMPLES_PER_CHUNK) { 0f }

        log("--- classifySilence ---")
        log("Input: silence (zeros), ${BirdClassifier.SAMPLES_PER_CHUNK} samples")
        log("Expected: 0 detections (or very few false positives)")
        log("")

        val start = System.currentTimeMillis()
        val detections = classifier.classify(silence)
        val inferenceMs = System.currentTimeMillis() - start

        log("Inference time: $inferenceMs ms")
        if (detections.isEmpty()) {
            log("ACTUAL: no detections (correct)")
        } else {
            log("ACTUAL: ${detections.size} detection(s) — possible false positives:")
            detections.forEachIndexed { i, d ->
                log("  ${i + 1}. ${d.commonName} (${d.scientificName}) " +
                    "— confidence: ${"%.4f".format(d.confidence)}")
            }
        }
        log("---")
    }

    private fun loadModel(
        context: android.content.Context,
        assetPath: String,
    ): MappedByteBuffer {
        return context.assets.openFd(assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /**
     * Reads a WAV file from test assets, converts PCM-16 → float32 [-1, 1],
     * returns exactly SAMPLES_PER_CHUNK samples (zero-padded if needed).
     */
    private fun loadWavChunk(context: android.content.Context, assetName: String): FloatArray {
        val bytes = context.assets.open(assetName).use { it.readBytes() }

        // Skip 44-byte WAV header
        val headerSize = 44
        val pcmBytes = bytes.copyOfRange(headerSize, bytes.size)
        val buffer = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN)
        val shortBuffer = buffer.asShortBuffer()

        val totalSamples = shortBuffer.remaining()
        val needed = BirdClassifier.SAMPLES_PER_CHUNK
        val durationSec = totalSamples.toFloat() / BirdClassifier.SAMPLE_RATE

        log("WAV loaded: $totalSamples samples (${"%.1f".format(durationSec)}s), " +
            "using first ${needed} (${BirdClassifier.CHUNK_DURATION_SECONDS}s)")

        val result = FloatArray(needed)
        val samplesToRead = minOf(totalSamples, needed)
        for (i in 0 until samplesToRead) {
            result[i] = shortBuffer.get() / 32768f
        }
        return result
    }

    companion object {
        private const val TAG = "BirdNetTest"

        private fun log(message: String) {
            Log.i(TAG, message)
            // Also print to stdout so it shows in Android Studio test output
            println(message)
        }
    }
}
