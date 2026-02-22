package com.birdsong.analyzer.presentation.detection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.ml.AudioFileDecoder
import com.birdsong.analyzer.ml.BirdClassifier
import com.birdsong.analyzer.ml.BirdDetectionPipeline
import com.birdsong.analyzer.ml.LocationMeta
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
import java.time.LocalDate
import java.time.temporal.WeekFields
import java.util.UUID
import javax.inject.Inject
import kotlin.math.roundToInt

@HiltViewModel
class LiveDetectionViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val audioRecorder: AudioRecorder,
    private val pipeline: BirdDetectionPipeline,
) : ViewModel() {

    private val _uiState = MutableStateFlow(LiveDetectionUiState())
    val uiState: StateFlow<LiveDetectionUiState> = _uiState.asStateFlow()

    private var recordingJob: Job? = null
    private var timerJob: Job? = null
    private var levelJob: Job? = null
    private var sessionStartMs: Long = 0L
    private var sessionLocation: LocationMeta? = null

    fun onStart() {
        val state = _uiState.value.state
        if (state != DetectionState.IDLE && state != DetectionState.STOPPED) return
        sessionLocation = resolveLocation()
        _uiState.update {
            it.copy(
                state = DetectionState.ANALYZING,
                detectedBirds = emptyList(),
                hasGps = sessionLocation != null,
            )
        }
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
        _uiState.update { it.copy(detectedBirds = emptyList()) }
    }

    fun onTestSample() {
        viewModelScope.launch {
            _uiState.update { it.copy(state = DetectionState.ANALYZING, detectedBirds = emptyList()) }
            try {
                Log.d(TAG, "=== TEST SAMPLE: $SAMPLE_ASSET ===")
                val samples = AudioFileDecoder.decodeFromAssets(context, SAMPLE_ASSET)
                val result = pipeline.processChunk(samples)
                Log.d(TAG, "Test sample: ${result.detections.size} detections, processed=${result.processed}")

                val birds = result.detections
                    .filter { it.confidence >= SAMPLE_MIN_CONFIDENCE }
                    .map { det ->
                        DetectedBirdUi(
                            id = UUID.randomUUID().toString(),
                            commonName = det.commonName,
                            scientificName = det.scientificName,
                            confidence = (det.confidence * 100).roundToInt(),
                            detectedAt = "sample",
                            durationSec = "%.1f".format(samples.size.toFloat() / BirdClassifier.SAMPLE_RATE),
                        )
                    }

                _uiState.update { s -> s.copy(detectedBirds = birds.take(MAX_DETECTIONS)) }
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

                val confirmed = pipeline.analyzeFile(
                    context = context,
                    uri = uri,
                    onProgress = { processed, skipped, total ->
                        Log.d(TAG, "File progress: $processed processed, $skipped skipped, $total total")
                    },
                )

                Log.d(TAG, "Aggregated ${confirmed.size} confirmed species:")
                confirmed.forEach { det ->
                    Log.d(TAG, "  ${det.commonName} (${det.scientificName}): " +
                        "${(det.confidence * 100).roundToInt()}% (${det.confirmedChunks} chunks)")
                }

                val birds = confirmed.map { det ->
                    DetectedBirdUi(
                        id = UUID.randomUUID().toString(),
                        commonName = det.commonName,
                        scientificName = det.scientificName,
                        confidence = (det.confidence * 100).roundToInt(),
                        detectedAt = "${det.confirmedChunks} chunks",
                        durationSec = "",
                    )
                }

                _uiState.update { s -> s.copy(detectedBirds = birds.take(MAX_DETECTIONS)) }
                Log.d(TAG, "=== TEST FILE DONE ===")
            } catch (e: Exception) {
                Log.e(TAG, "Test file classification failed", e)
            } finally {
                _uiState.update { it.copy(state = DetectionState.STOPPED) }
            }
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

                        val result = pipeline.processChunk(chunk, sessionLocation)
                        if (!result.processed) return@collect

                        val elapsedMs = System.currentTimeMillis() - sessionStartMs
                        val elapsedSec = elapsedMs / 1_000
                        val detectedAt = formatMmSs(elapsedSec)
                        val windowStart = formatMmSs((elapsedSec - CHUNK_DURATION_SEC).coerceAtLeast(0))
                        val durationSec = "$windowStart – $detectedAt"

                        val newBirds = result.detections
                            .map { det ->
                                DetectedBirdUi(
                                    id = UUID.randomUUID().toString(),
                                    commonName = det.commonName,
                                    scientificName = det.scientificName,
                                    confidence = (det.confidence * 100).roundToInt(),
                                    detectedAt = detectedAt,
                                    durationSec = durationSec,
                                )
                            }

                        if (newBirds.isNotEmpty()) {
                            Log.d(TAG, "Appending ${newBirds.size} detections at $detectedAt")
                            _uiState.update { s ->
                                var birds = s.detectedBirds
                                for (bird in newBirds) {
                                    val top = birds.firstOrNull()
                                    if (top != null && top.scientificName == bird.scientificName) {
                                        // Same species as previous — extend time window
                                        birds = listOf(
                                            top.copy(
                                                detectedAt = bird.detectedAt,
                                                durationSec = "${top.durationSec.substringBefore(" – ")} – ${bird.detectedAt}",
                                                confidence = maxOf(top.confidence, bird.confidence),
                                            ),
                                        ) + birds.drop(1)
                                    } else {
                                        birds = listOf(bird) + birds
                                    }
                                }
                                s.copy(detectedBirds = birds.take(MAX_DETECTIONS))
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Classification failed", e)
                    }
                }
        }
    }

    private fun resolveLocation(): LocationMeta? {
        val hasPermission = context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        if (!hasPermission) {
            Log.d(TAG, "Location permission not granted, meta-model will run without geo-filter")
            return null
        }
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val loc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            ?: return null
        val week = LocalDate.now().get(WeekFields.ISO.weekOfWeekBasedYear())
        // ±LIVE_WEEK_WINDOW so recently returned migrants aren't suppressed.
        // At year boundary (wrap-around) fall back to full year — geographic filter only.
        val lo = week - LIVE_WEEK_WINDOW
        val hi = week + LIVE_WEEK_WINDOW
        val weekRange = if (lo < 1 || hi > 52) 1..52 else lo..hi
        Log.d(TAG, "Session location: lat=${loc.latitude}, lon=${loc.longitude}, weekRange=$weekRange")
        return LocationMeta(latitude = loc.latitude, longitude = loc.longitude, weekOfYear = week, weekRange = weekRange)
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

    private fun formatMmSs(totalSec: Long): String =
        "%02d:%02d".format(totalSec / 60, totalSec % 60)

    override fun onCleared() {
        onStop()
    }

    companion object {
        private const val TAG = "LiveDetectionVM"
        private const val MAX_DETECTIONS = 200
        private const val CHUNK_DURATION_SEC = 3L
        private const val LIVE_WEEK_WINDOW = 4  // ±4 weeks around current date
        private const val SAMPLE_ASSET = "birdnet/v24/sample.wav"
        private const val SAMPLE_MIN_CONFIDENCE = 0.5f
    }
}
