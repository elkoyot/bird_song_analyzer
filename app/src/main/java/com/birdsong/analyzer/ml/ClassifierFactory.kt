package com.birdsong.analyzer.ml

import android.content.Context
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.nio.MappedByteBuffer
import javax.inject.Inject
import javax.inject.Named
import javax.inject.Singleton

/**
 * Lazy factory for creating classifiers, processors, and audio configs by model type.
 *
 * BirdNET V2.4 assets are injected via Hilt (loaded at app startup).
 * BirdNET V3.0 model is loaded from [Context.getFilesDir]/models/birdnet_v30/ on first use.
 */
@Singleton
class ClassifierFactory @Inject constructor(
    @ApplicationContext private val context: Context,
    @Named("birdnetAudioModel") private val birdnetAudioModel: MappedByteBuffer,
    @Named("birdnetMetaModel") private val birdnetMetaModel: MappedByteBuffer,
    @Named("birdnetLabels") private val birdnetLabels: List<Pair<String, String>>,
) {
    // V3.0 labels loaded lazily on first createBirdNetV30() call
    private var v30Labels: BirdNetV30LabelLoader.V30Labels? = null

    fun createBirdNet(): BirdClassifier {
        return BirdNetV24Classifier(
            audioModel = birdnetAudioModel,
            metaModel = birdnetMetaModel,
            labels = birdnetLabels,
        )
    }

    fun createBirdNetV30(): BirdClassifier {
        val labels = loadV30Labels()
        val modelFile = v30ModelFile()
        require(modelFile.exists()) { "BirdNET V3.0 model not found at ${modelFile.absolutePath}" }

        val birdnetIndex = buildBirdnetLabelIndex(labels.labels)

        return BirdNetV30Classifier(
            modelPath = modelFile.absolutePath,
            labels = labels.labels,
            classNames = labels.classNames,
            birdnetLabelIndex = birdnetIndex,
        )
    }

    /**
     * Build mapping from V3.0 label index → V2.4 label index.
     * Uses scientific name (first element of label pair) for cross-referencing.
     * Returns -1 for V3.0 species not present in V2.4.
     */
    private fun buildBirdnetLabelIndex(v30Labels: List<Pair<String, String>>): IntArray {
        val birdnetNameToIdx = HashMap<String, Int>(birdnetLabels.size)
        for ((i, label) in birdnetLabels.withIndex()) {
            birdnetNameToIdx[label.first] = i
        }
        val result = IntArray(v30Labels.size) { -1 }
        var matched = 0
        for ((i, label) in v30Labels.withIndex()) {
            val idx = birdnetNameToIdx[label.first]
            if (idx != null) { result[i] = idx; matched++ }
        }
        Log.i(TAG, "V3.0→V2.4 mapping: $matched/${v30Labels.size} matched")
        return result
    }

    fun createProcessor(classifier: BirdClassifier): AudioChunkProcessor {
        return AudioChunkProcessor(sampleRate = classifier.sampleRate, mode = PreprocessingMode.FULL)
    }

    fun audioConfigFor(classifier: BirdClassifier): AudioConfig {
        return AudioConfig(
            sampleRate = classifier.sampleRate,
            samplesPerChunk = classifier.samplesPerChunk,
        )
    }

    fun isBirdNetV30Available(): Boolean {
        val file = v30ModelFile()
        val exists = file.exists()
        Log.i(TAG, "V3.0 model check: ${file.absolutePath} → exists=$exists" +
            if (exists) " (${file.length() / 1024}KB)" else "")
        return exists
    }

    private fun v30ModelFile(): File =
        File(context.filesDir, "models/birdnet_v30/${BirdNetV30Classifier.MODEL_FILENAME}")

    private fun loadV30Labels(): BirdNetV30LabelLoader.V30Labels {
        return v30Labels ?: run {
            val loaded = BirdNetV30LabelLoader.load(context, birdnetLabels)
            v30Labels = loaded
            Log.i(TAG, "V3.0 labels loaded: ${loaded.labels.size} species")
            loaded
        }
    }

    companion object {
        private const val TAG = "ClassifierFactory"
        const val MODEL_BIRDNET = "birdnet_v24"
        const val MODEL_BIRDNET_V30 = "birdnet_v30"
    }
}
