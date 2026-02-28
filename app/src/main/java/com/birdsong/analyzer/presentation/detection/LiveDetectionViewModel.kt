package com.birdsong.analyzer.presentation.detection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.PreferencesRepository
import com.birdsong.analyzer.ml.AudioFileDecoder
import com.birdsong.analyzer.ml.BirdDetection
import com.birdsong.analyzer.ml.BirdDetectionPipeline
import com.birdsong.analyzer.ml.ClassifierFactory
import com.birdsong.analyzer.ml.DetectionAggregator
import com.birdsong.analyzer.ml.CountryConfig
import com.birdsong.analyzer.ml.CountryConfigLoader
import com.birdsong.analyzer.ml.FamilyTaxonomy
import com.birdsong.analyzer.ml.LocationMeta
import com.birdsong.analyzer.ml.MetaProfile
import com.birdsong.analyzer.ml.MetaProfileBuilder
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
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
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
    private val classifierFactory: ClassifierFactory,
    private val metaProfileBuilder: MetaProfileBuilder,
    private val prefsRepo: PreferencesRepository,
    private val countries: List<CountryConfig>,
    private val familyTaxonomy: FamilyTaxonomy,
) : ViewModel() {

    private val _uiState = MutableStateFlow(LiveDetectionUiState())
    val uiState: StateFlow<LiveDetectionUiState> = _uiState.asStateFlow()

    private var preparingJob: Job? = null
    private var recordingJob: Job? = null
    private var timerJob: Job? = null
    private var levelJob: Job? = null
    private var metaProfileJob: Job? = null
    private var sessionStartMs: Long = 0L
    private var sessionLocation: LocationMeta? = null
    private val aggregator = DetectionAggregator.forLiveDetection(
        threshold = AGGREGATOR_THRESHOLD,
        confirmationCount = AGGREGATOR_CONFIRMATION,
    )

    private var currentModelId: String = ClassifierFactory.MODEL_BIRDNET

    @Volatile
    private var cachedMetaProfile: MetaProfile? = null

    init {
        buildMetaProfileAsync()
        observeModelChanges()
    }

    private fun buildMetaProfileAsync() {
        metaProfileJob = viewModelScope.launch {
            try {
                val code = prefsRepo.countryCode.first()
                val region = prefsRepo.regionCode.first()
                val config = CountryConfigLoader.findByCode(countries, code, region) ?: return@launch
                Log.d(TAG, "Building MetaProfile for ${config.nameEn} (${config.code})")
                val profile = metaProfileBuilder.build(config.bbox, config.bufferDeg)
                cachedMetaProfile = profile
                pipeline.classifier.metaProfile = profile
                Log.d(TAG, "MetaProfile ready for ${config.nameEn}")
            } catch (e: Exception) {
                Log.e(TAG, "MetaProfile build failed", e)
            }
        }
    }

    private fun observeModelChanges() {
        viewModelScope.launch {
            prefsRepo.activeModel
                .distinctUntilChanged()
                .collect { modelId ->
                    if (modelId != currentModelId) {
                        switchModel(modelId)
                    }
                }
        }
    }

    private suspend fun switchModel(modelId: String) {
        Log.i(TAG, "Switching model: $currentModelId → $modelId")
        val wasRunning = _uiState.value.state == DetectionState.ANALYZING ||
            _uiState.value.state == DetectionState.PAUSED
        if (wasRunning) onStop()

        try {
            val classifier = when (modelId) {
                ClassifierFactory.MODEL_BIRDNET_V30 -> classifierFactory.createBirdNetV30()
                else -> classifierFactory.createBirdNet()
            }
            val processor = classifierFactory.createProcessor(classifier)
            val config = classifierFactory.audioConfigFor(classifier)

            pipeline.configure(processor, classifier)
            audioRecorder.configure(config)
            currentModelId = modelId

            // Rebuild MetaProfile for BirdNET V2.4; apply cached for V3.0
            if (modelId == ClassifierFactory.MODEL_BIRDNET) {
                buildMetaProfileAsync()
            } else {
                cachedMetaProfile?.let { pipeline.classifier.metaProfile = it }
            }

            Log.i(TAG, "Model switched to ${classifier.modelId}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to switch to $modelId, keeping current model", e)
        }
    }

    fun onStart() {
        val state = _uiState.value.state
        if (state != DetectionState.IDLE && state != DetectionState.STOPPED) return
        _uiState.update { it.copy(state = DetectionState.PREPARING) }
        preparingJob = viewModelScope.launch {
            metaProfileJob?.join()
            if (_uiState.value.state != DetectionState.PREPARING) return@launch
            startDetection()
        }
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
        preparingJob?.cancel()
        preparingJob = null
        recordingJob?.cancel()
        recordingJob = null
        timerJob?.cancel()
        timerJob = null
        levelJob?.cancel()
        levelJob = null
        _uiState.update { it.copy(state = DetectionState.STOPPED, audioLevel = 0f) }
    }

    fun onReset() {
        aggregator.reset()
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
                            durationSec = "%.1f".format(samples.size.toFloat() / pipeline.classifier.sampleRate),
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

    private fun startDetection() {
        sessionLocation = resolveLocation()
        aggregator.reset()
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
                        if (!result.processed) {
                            aggregator.addChunkResults(null)
                            return@collect
                        }

                        // Feed all detections to aggregator (before any UI filtering)
                        aggregator.addChunkResults(result.detections)

                        val elapsedMs = System.currentTimeMillis() - sessionStartMs
                        val elapsedSec = elapsedMs / 1_000
                        val detectedAt = formatMmSs(elapsedSec)
                        val chunkDur = pipeline.classifier.chunkDurationSeconds.toLong()
                        val windowStart = formatMmSs((elapsedSec - chunkDur).coerceAtLeast(0))
                        val durationSec = "$windowStart – $detectedAt"

                        // Log all detections returned by classifier (before UI filtering)
                        if (result.detections.isNotEmpty()) {
                            Log.d(TAG, "Classifier @ $detectedAt → ${result.detections.size} detections:")
                            result.detections.forEach { det ->
                                Log.d(TAG, "  ${det.commonName} (${det.scientificName}) " +
                                    "conf=%.3f (%.1f%%)".format(det.confidence, det.confidence * 100))
                            }
                        } else {
                            Log.d(TAG, "Classifier @ $detectedAt → no detections")
                        }

                        // Species confirmed by aggregator (≥2 chunks in sliding window)
                        val confirmedSpecies = aggregator.getConfirmedDetections()
                            .associate { it.scientificName to it }

                        // 2-path filter: anchor + family-aware aggregator
                        val filtered = filterDetections(result.detections, confirmedSpecies)

                        val newBirds = filtered
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
                            Log.d(TAG, "→ UI: appending ${newBirds.size} birds at $detectedAt")
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

    /**
     * 2-path filter for live detections + family dedup:
     *
     * 1. **Anchor** — confidence ≥ [LIVE_MIN_CONFIDENCE] → immediate.
     *    Species from the same taxonomic family with lower confidence are suppressed
     *    (model confusion between similar species, e.g. Sylvia/Curruca are both Sylviidae).
     *
     * 2. **Aggregator-confirmed** — species confirmed by ≥ [AGGREGATOR_CONFIRMATION] chunks
     *    in sliding window → pass with aggregated confidence.
     *    Only species whose family has NO anchor pass — if a family already has
     *    a confident representative, weaker alternatives are model noise.
     *
     * 3. **Family dedup** — when multiple species from the same family pass
     *    (e.g. all via aggregator when none reaches anchor threshold),
     *    only the highest-confidence per family survives.
     */
    private fun filterDetections(
        detections: List<BirdDetection>,
        confirmedSpecies: Map<String, DetectionAggregator.AggregatedDetection>,
    ): List<BirdDetection> {
        // Families that already have a high-confidence anchor in this chunk
        val anchorFamilies = detections
            .filter { it.confidence >= LIVE_MIN_CONFIDENCE }
            .mapNotNull { familyTaxonomy.getFamily(it.scientificName) }
            .toSet()

        val candidates = mutableListOf<BirdDetection>()

        for (det in detections) {
            val family = familyTaxonomy.getFamily(det.scientificName)
            when {
                // Path 1: high confidence anchor
                det.confidence >= LIVE_MIN_CONFIDENCE -> candidates.add(det)

                // Path 2: aggregator-confirmed, but only if family has NO anchor
                family !in anchorFamilies && det.scientificName in confirmedSpecies -> {
                    val agg = confirmedSpecies.getValue(det.scientificName)
                    Log.d(TAG, "  aggregator-confirmed: ${det.commonName} " +
                        "chunkConf=%.1f%% aggConf=%.1f%% (%d chunks) family=$family"
                            .format(det.confidence * 100, agg.confidence * 100, agg.confirmedChunks))
                    candidates.add(det.copy(confidence = agg.confidence))
                }

                // Suppressed: same family as anchor, or not confirmed by aggregator
                family != null && family in anchorFamilies -> {
                    Log.d(TAG, ("  family-suppressed: ${det.commonName} conf=%.1f%% " +
                        "(family=$family has anchor)").format(det.confidence * 100))
                }
            }
        }

        // Dedup: keep only the best species per family
        val bestPerFamily = HashMap<String, BirdDetection>()
        val result = mutableListOf<BirdDetection>()

        for (det in candidates) {
            val family = familyTaxonomy.getFamily(det.scientificName)
            if (family == null) {
                result.add(det)
                continue
            }
            val existing = bestPerFamily[family]
            if (existing == null || det.confidence > existing.confidence) {
                bestPerFamily[family] = det
            }
        }

        for (det in candidates) {
            val family = familyTaxonomy.getFamily(det.scientificName) ?: continue
            if (bestPerFamily[family]?.scientificName == det.scientificName) {
                result.add(det)
            } else {
                Log.d(TAG, "  family-dedup: ${det.commonName} conf=%.1f%% " +
                    "(family=$family, better=${bestPerFamily[family]?.commonName})"
                        .format(det.confidence * 100))
            }
        }

        return result
    }

    override fun onCleared() {
        onStop()
    }

    companion object {
        private const val TAG = "LiveDetectionVM"
        private const val MAX_DETECTIONS = 200
        private const val LIVE_WEEK_WINDOW = 4  // ±4 weeks around current date
        private const val SAMPLE_ASSET = "birdnet/v24/sample.wav"
        private const val LIVE_MIN_CONFIDENCE = 0.75f
        private const val AGGREGATOR_THRESHOLD = 0.10f   // min per-chunk confidence to count
        private const val AGGREGATOR_CONFIRMATION = 2     // chunks needed to confirm
        private const val SAMPLE_MIN_CONFIDENCE = 0.5f
    }
}
