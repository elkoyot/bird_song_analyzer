package com.birdsong.analyzer.presentation.detection

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.ml.AudioChunkProcessor
import com.birdsong.analyzer.ml.AudioFileDecoder
import com.birdsong.analyzer.ml.BirdClassifier
import com.birdsong.analyzer.ml.DetectionAggregator
import com.birdsong.analyzer.service.AudioRecorder
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.UUID
import javax.inject.Inject
import kotlin.math.roundToInt

@HiltViewModel
class LiveDetectionViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val audioRecorder: AudioRecorder,
    private val classifier: BirdClassifier,
    private val audioChunkProcessor: AudioChunkProcessor,
) : ViewModel() {

    private val _uiState = MutableStateFlow(LiveDetectionUiState())
    val uiState: StateFlow<LiveDetectionUiState> = _uiState.asStateFlow()

    private var recordingJob: Job? = null
    private var timerJob: Job? = null
    private var levelJob: Job? = null
    private var sessionStartMs: Long = 0L

    private var liveAggregator = DetectionAggregator.forLiveDetection()

    fun onStart() {
        val state = _uiState.value.state
        if (state != DetectionState.IDLE && state != DetectionState.STOPPED) return
        liveAggregator.reset()
        _uiState.update { it.copy(state = DetectionState.ANALYZING, detectedBirds = emptyList()) }
        sessionStartMs = System.currentTimeMillis()
        startTimerLoop()
        startLevelCollection()
        startRecordingLoop()
    }

    fun onPause() {
        recordingJob?.cancel()
        recordingJob = null
        levelJob?.cancel()
        levelJob = null
        _uiState.update { it.copy(state = DetectionState.PAUSED, audioLevel = 0f) }
    }

    fun onResume() {
        if (_uiState.value.state != DetectionState.PAUSED) return
        _uiState.update { it.copy(state = DetectionState.ANALYZING) }
        startLevelCollection()
        startRecordingLoop()
    }

    fun onStop() {
        recordingJob?.cancel()
        recordingJob = null
        timerJob?.cancel()
        timerJob = null
        levelJob?.cancel()
        levelJob = null
        _uiState.update { it.copy(state = DetectionState.STOPPED, audioLevel = 0f) }
    }

    fun onReset() {
        liveAggregator.reset()
        _uiState.update { it.copy(detectedBirds = emptyList()) }
    }

    fun onTestSample() {
        viewModelScope.launch {
            _uiState.update { it.copy(state = DetectionState.ANALYZING, detectedBirds = emptyList()) }
            try {
                Log.d(TAG, "=== TEST SAMPLE: $SAMPLE_ASSET ===")
                val allSamples = AudioFileDecoder.decodeFromAssets(context, SAMPLE_ASSET)
                classifyAllChunks(allSamples)
                Log.d(TAG, "=== TEST SAMPLE DONE ===")
            } catch (e: Exception) {
                Log.e(TAG, "Test sample classification failed", e)
            } finally {
                _uiState.update { it.copy(state = DetectionState.STOPPED) }
            }
        }
    }

    fun onTestFile(uri: Uri) {
        viewModelScope.launch {
            _uiState.update { it.copy(state = DetectionState.ANALYZING, detectedBirds = emptyList()) }
            try {
                Log.d(TAG, "=== TEST FILE: $uri ===")
                val allSamples = AudioFileDecoder.decode(context, uri)
                classifyAllChunks(allSamples)
                Log.d(TAG, "=== TEST FILE DONE ===")
            } catch (e: Exception) {
                Log.e(TAG, "Test file classification failed", e)
            } finally {
                _uiState.update { it.copy(state = DetectionState.STOPPED) }
            }
        }
    }

    private suspend fun classifyAllChunks(allSamples: FloatArray) {
        val durationSec = allSamples.size.toFloat() / BirdClassifier.SAMPLE_RATE
        Log.d(TAG, "Decoded ${allSamples.size} samples (%.1fs at ${BirdClassifier.SAMPLE_RATE}Hz)".format(durationSec))

        val chunkSize = BirdClassifier.SAMPLES_PER_CHUNK
        val hopSize = chunkSize / 2  // 50% overlap

        val offsets = mutableListOf<Int>()
        var offset = 0
        while (offset + chunkSize <= allSamples.size) {
            offsets.add(offset)
            offset += hopSize
        }
        Log.d(TAG, "Will classify ${offsets.size} overlapping chunks (hop=${hopSize} = %.1fs)".format(hopSize.toFloat() / BirdClassifier.SAMPLE_RATE))

        val fileAggregator = DetectionAggregator.forFileAnalysis()

        for ((idx, off) in offsets.withIndex()) {
            val chunk = allSamples.copyOfRange(off, off + chunkSize)
            Log.d(TAG, "--- Chunk ${idx + 1}/${offsets.size} (offset %.1fs) ---".format(off.toFloat() / BirdClassifier.SAMPLE_RATE))

            val processed = audioChunkProcessor.process(chunk)
            if (processed == null) {
                fileAggregator.addChunkResults(null)
                continue
            }

            val detections = classifier.classify(processed.samples)
            fileAggregator.addChunkResults(detections)
        }

        val confirmed = fileAggregator.getConfirmedDetections()
        Log.d(TAG, "Aggregated ${confirmed.size} confirmed species:")
        confirmed.forEach { det ->
            Log.d(TAG, "  ${det.commonName} (${det.scientificName}): ${(det.confidence * 100).roundToInt()}% (${det.confirmedChunks} chunks)")
        }

        val birds = confirmed.map { det ->
            DetectedBirdUi(
                id = UUID.randomUUID().toString(),
                commonName = det.commonName,
                scientificName = det.scientificName,
                confidence = (det.confidence * 100).roundToInt(),
                detectedAt = "${offsets.size} chunks",
                durationSec = "%.1f".format(durationSec),
            )
        }

        _uiState.update { s ->
            s.copy(detectedBirds = birds.take(MAX_DETECTIONS))
        }
    }

    private fun startRecordingLoop() {
        recordingJob = viewModelScope.launch {
            audioRecorder.chunksFlow()
                .catch { e ->
                    Log.e(TAG, "Audio recording failed", e)
                    _uiState.update { it.copy(state = DetectionState.IDLE) }
                }
                .buffer(capacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST)
                .collect { chunk ->
                    try {
                        Log.d(TAG, "Chunk received: ${chunk.size} samples")

                        val processed = audioChunkProcessor.process(chunk)
                        if (processed == null) {
                            liveAggregator.addChunkResults(null)
                            updateUiFromAggregator()
                            return@collect
                        }

                        val detections = classifier.classify(processed.samples)
                        Log.d(TAG, "Detections above classifier threshold: ${detections.size}")

                        liveAggregator.addChunkResults(detections)
                        updateUiFromAggregator()
                    } catch (e: Exception) {
                        Log.e(TAG, "Classification failed", e)
                    }
                }
        }
    }

    private fun updateUiFromAggregator() {
        val confirmed = liveAggregator.getConfirmedDetections()
        Log.d(TAG, "Confirmed species: ${confirmed.size}")

        val newBirds = confirmed.map { det ->
            DetectedBirdUi(
                id = UUID.randomUUID().toString(),
                commonName = det.commonName,
                scientificName = det.scientificName,
                confidence = (det.confidence * 100).roundToInt(),
                detectedAt = LocalTime.now().format(TIME_FORMATTER),
                durationSec = CHUNK_DURATION_LABEL,
            )
        }

        _uiState.update { s ->
            // Merge: update existing species or add new
            val existing = s.detectedBirds.associateBy { it.scientificName }.toMutableMap()
            for (bird in newBirds) {
                val prev = existing[bird.scientificName]
                if (prev == null || bird.confidence > prev.confidence) {
                    existing[bird.scientificName] = bird
                }
            }
            // Remove species no longer confirmed
            val confirmedNames = newBirds.map { it.scientificName }.toSet()
            existing.keys.retainAll(confirmedNames)

            s.copy(
                detectedBirds = existing.values
                    .sortedByDescending { it.confidence }
                    .take(MAX_DETECTIONS)
                    .toList(),
            )
        }
    }

    private fun startLevelCollection() {
        levelJob = viewModelScope.launch {
            audioRecorder.audioLevel.collect { level ->
                _uiState.update { it.copy(audioLevel = level) }
            }
        }
    }

    private fun startTimerLoop() {
        timerJob = viewModelScope.launch {
            while (true) {
                delay(1_000L)
                val elapsedMs = System.currentTimeMillis() - sessionStartMs
                _uiState.update { it.copy(sessionTimer = formatDuration(elapsedMs)) }
            }
        }
    }

    private fun formatDuration(elapsedMs: Long): String {
        val s = elapsedMs / 1_000
        return "%02d:%02d:%02d".format(s / 3_600, s % 3_600 / 60, s % 60)
    }

    override fun onCleared() {
        onStop()
    }

    companion object {
        private const val TAG = "LiveDetectionVM"
        private const val MAX_DETECTIONS = 200
        private const val CHUNK_DURATION_LABEL = "3.0"
        private const val SAMPLE_ASSET = "birdnet/v24/sample.wav"
        private val TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm")
    }
}
