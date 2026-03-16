package com.birdsong.analyzer.presentation.detection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.repository.GeoRepository
import com.birdsong.analyzer.ml.AudioChunkProcessor
import com.birdsong.analyzer.ml.AudioResampler
import com.birdsong.analyzer.ml.BirdClassifier
import com.birdsong.analyzer.ml.BirdDetection
import com.birdsong.analyzer.ml.BoundingBox
import com.birdsong.analyzer.ml.ClassifierFactory
import com.birdsong.analyzer.ml.DetectionAggregator
import com.birdsong.analyzer.ml.FamilyTaxonomy
import com.birdsong.analyzer.ml.LocationMeta
import com.birdsong.analyzer.ml.MetaProfile
import com.birdsong.analyzer.ml.MetaProfileBuilder
import com.birdsong.analyzer.service.AudioRecorder
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.temporal.WeekFields
import java.util.UUID
import javax.inject.Inject
import kotlin.math.roundToInt

data class DualDetectedBirdUi(
    val id: String,
    val commonName: String,
    val scientificName: String,
    val v24Confidence: Int? = null,
    val v30Confidence: Int? = null,
    val detectedAt: String,
)

data class DualDetectionUiState(
    val state: DetectionState = DetectionState.IDLE,
    val sessionTimer: String = "00:00",
    val hasGps: Boolean = false,
    val audioLevel: Float = 0f,
    val birds: List<DualDetectedBirdUi> = emptyList(),
    val v30Available: Boolean = false,
    val flashBirdId: String? = null,
    val newBirdIds: Set<String> = emptySet(),
    val luringBirdId: String? = null,
    val regionLabel: String? = null,
    val blipSeq: Int = 0,
)

@HiltViewModel
class DualDetectionViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val audioRecorder: AudioRecorder,
    private val classifierFactory: ClassifierFactory,
    private val metaProfileBuilder: MetaProfileBuilder,
    private val geoRepository: GeoRepository,
    private val familyTaxonomy: FamilyTaxonomy,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DualDetectionUiState(
        v30Available = classifierFactory.isBirdNetV30Available(),
    ))
    val uiState: StateFlow<DualDetectionUiState> = _uiState.asStateFlow()

    private var recordingJob: Job? = null
    private var timerJob: Job? = null
    private var levelJob: Job? = null
    private var metaProfileJob: Job? = null
    private var sessionStartMs: Long = 0L
    private var sessionLocation: LocationMeta? = null

    // Pre-built meta profile (built in init, applied to BirdNET on start)
    @Volatile
    private var cachedMetaProfile: MetaProfile? = null

    // Chunk counters for structured logging
    private var v24ChunkNum = 0
    private var v30ChunkNum = 0

    // Per-model shadow maps — each pair accessed only under its own mutex
    // V2.4 shadow (protected by birdnetMutex)
    private val v24FamilyShadow = HashMap<String, Long>()
    private val v24AnchorSpecies = HashMap<String, Long>()

    // V3.0 shadow (protected by v30Mutex)
    private val v30FamilyShadow = HashMap<String, Long>()
    private val v30AnchorSpecies = HashMap<String, Long>()

    // BirdNET V2.4 pipeline state
    private var birdnetClassifier: BirdClassifier? = null
    private var birdnetProcessor: AudioChunkProcessor? = null
    private val birdnetAggregator = DetectionAggregator.forLiveDetection(
        threshold = AGGREGATOR_THRESHOLD,
        confirmationCount = AGGREGATOR_CONFIRMATION,
    )

    // BirdNET V3.0 pipeline state
    private var v30Classifier: BirdClassifier? = null
    private var v30Processor: AudioChunkProcessor? = null
    private val v30Aggregator = DetectionAggregator.forLiveDetection(
        threshold = AGGREGATOR_THRESHOLD,
        confirmationCount = AGGREGATOR_CONFIRMATION,
    )

    @Volatile private var clearActiveJob: Job? = null
    private var lureJob: Job? = null

    init {
        // Collect region label for header display
        viewModelScope.launch {
            geoRepository.currentSelectionDisplay.collect { display ->
                val label = if (display == "\u2014") null else display
                _uiState.update { it.copy(regionLabel = label) }
            }
        }
        metaProfileJob = viewModelScope.launch {
            try {
                val geo = geoRepository.resolveCurrentGeo() ?: return@launch
                val bbox = BoundingBox(geo.minLat!!, geo.maxLat!!, geo.minLon!!, geo.maxLon!!)
                cachedMetaProfile = metaProfileBuilder.build(bbox, geo.bufferDeg)
                Log.d(TAG, "MetaProfile ready for ${geo.nameEn}")
            } catch (e: Exception) {
                Log.e(TAG, "MetaProfile build failed", e)
            }
        }
    }

    fun onStart() {
        val state = _uiState.value.state
        if (state != DetectionState.IDLE && state != DetectionState.STOPPED) return
        _uiState.update { it.copy(state = DetectionState.PREPARING) }

        viewModelScope.launch {
            metaProfileJob?.join()
            if (_uiState.value.state != DetectionState.PREPARING) return@launch

            try {
                // Create BirdNET V2.4 classifier
                val birdnet = classifierFactory.createBirdNet()
                birdnetClassifier = birdnet
                birdnetProcessor = classifierFactory.createProcessor(birdnet)

                // Apply cached meta-profile to BirdNET V2.4
                cachedMetaProfile?.let { birdnet.metaProfile = it }

                // Create BirdNET V3.0 classifier (if available)
                val v30Available = classifierFactory.isBirdNetV30Available()
                Log.i(TAG, "V3.0 available=$v30Available")
                if (v30Available) {
                    try {
                        val v30 = classifierFactory.createBirdNetV30()
                        cachedMetaProfile?.let { v30.metaProfile = it }
                        v30Classifier = v30
                        v30Processor = classifierFactory.createProcessor(v30)
                        Log.i(TAG, "BirdNET V3.0 classifier ready (geo-filter: ${cachedMetaProfile != null})")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to create V3.0 classifier", e)
                        _uiState.update { it.copy(v30Available = false) }
                    }
                } else {
                    Log.w(TAG, "V3.0 model not found — push with: adb push <model>.onnx " +
                        "/data/data/com.birdsong.analyzer/files/models/birdnet_v30/birdnet_v30_euna.onnx")
                }

                startDetection()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start dual detection", e)
                _uiState.update { it.copy(state = DetectionState.IDLE) }
            }
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
        recordingJob?.cancel()
        recordingJob = null
        timerJob?.cancel()
        timerJob = null
        levelJob?.cancel()
        levelJob = null
        clearActiveJob?.cancel()
        clearActiveJob = null
        lureJob?.cancel()
        lureJob = null
        _uiState.update { it.copy(state = DetectionState.STOPPED, audioLevel = 0f, flashBirdId = null, luringBirdId = null) }

        birdnetClassifier?.close()
        birdnetClassifier = null
        birdnetProcessor = null
        v30Classifier?.close()
        v30Classifier = null
        v30Processor = null
    }

    fun onSaveSession() {
        // TODO: persist to ObservationEntity when schema is ready
        resetAfterSession()
    }

    fun onDiscardSession() {
        resetAfterSession()
    }

    private fun resetAfterSession() {
        birdnetAggregator.reset()
        v30Aggregator.reset()
        v24ChunkNum = 0
        v30ChunkNum = 0
        v24FamilyShadow.clear(); v24AnchorSpecies.clear()
        v30FamilyShadow.clear(); v30AnchorSpecies.clear()
        _uiState.update {
            DualDetectionUiState(
                v30Available = it.v30Available,
                regionLabel = it.regionLabel,
            )
        }
    }

    fun onReset() {
        onClearList()
    }

    fun onClearList() {
        clearActiveJob?.cancel()
        clearActiveJob = null
        birdnetAggregator.reset()
        v30Aggregator.reset()
        v24ChunkNum = 0
        v30ChunkNum = 0
        v24FamilyShadow.clear(); v24AnchorSpecies.clear()
        v30FamilyShadow.clear(); v30AnchorSpecies.clear()
        _uiState.update { it.copy(birds = emptyList(), flashBirdId = null, newBirdIds = emptySet()) }
    }

    fun onRemoveBird(birdId: String) {
        _uiState.update { s ->
            s.copy(
                birds = s.birds.filter { it.id != birdId },
                newBirdIds = s.newBirdIds - birdId,
            )
        }
    }

    fun onLure(birdId: String) {
        val current = _uiState.value.luringBirdId
        if (current == birdId) {
            // Stop luring
            lureJob?.cancel()
            lureJob = null
            _uiState.update { it.copy(luringBirdId = null) }
            if (_uiState.value.state == DetectionState.PAUSED) onResume()
            return
        }
        // Start luring — pause detection for 8s
        if (_uiState.value.state == DetectionState.ANALYZING) onPause()
        _uiState.update { it.copy(luringBirdId = birdId) }
        lureJob?.cancel()
        lureJob = viewModelScope.launch {
            delay(LURE_DURATION_MS)
            _uiState.update { it.copy(luringBirdId = null) }
            if (_uiState.value.state == DetectionState.PAUSED) onResume()
        }
    }

    private fun startDetection() {
        sessionLocation = resolveLocation()
        birdnetAggregator.reset()
        v30Aggregator.reset()
        v24ChunkNum = 0
        v30ChunkNum = 0
        v24FamilyShadow.clear(); v24AnchorSpecies.clear()
        v30FamilyShadow.clear(); v30AnchorSpecies.clear()
        _uiState.update {
            it.copy(
                state = DetectionState.ANALYZING,
                birds = emptyList(),
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
            // BirdNET V2.4: 48 kHz, 3s = 144_000 samples, hop = 72_000
            val birdnetChunkSize = BIRDNET_SAMPLE_RATE * BIRDNET_CHUNK_SEC  // 144_000
            val birdnetHop = birdnetChunkSize / 2                           // 72_000
            val birdnetBuf = FloatArray(birdnetChunkSize)
            var birdnetFilled = 0

            // V3.0: needs 32 kHz, 5s = 160_000 samples
            // We record at 48 kHz, so accumulate 48_000 * 5 = 240_000, then resample
            val v30AccumSize = BIRDNET_SAMPLE_RATE * V30_CHUNK_SEC         // 240_000
            val v30Hop = v30AccumSize / 2                                   // 120_000
            val v30Buf = FloatArray(v30AccumSize)
            var v30Filled = 0

            audioRecorder.rawSamplesFlow(BIRDNET_SAMPLE_RATE)
                .catch { e ->
                    Log.e(TAG, "Audio recording failed", e)
                    _uiState.update { it.copy(state = DetectionState.IDLE) }
                }
                .collect { samples ->
                    // Feed into BirdNET V2.4 accumulator
                    var srcIdx = 0
                    while (srcIdx < samples.size) {
                        val toCopy = minOf(samples.size - srcIdx, birdnetChunkSize - birdnetFilled)
                        samples.copyInto(birdnetBuf, birdnetFilled, srcIdx, srcIdx + toCopy)
                        birdnetFilled += toCopy
                        srcIdx += toCopy

                        if (birdnetFilled == birdnetChunkSize) {
                            val chunk = birdnetBuf.copyOf()
                            launch(Dispatchers.Default) { runBirdNetInference(chunk) }

                            birdnetBuf.copyInto(birdnetBuf, 0, birdnetHop, birdnetChunkSize)
                            birdnetFilled = birdnetChunkSize - birdnetHop
                        }
                    }

                    // Feed into V3.0 accumulator (same 48 kHz data, will resample when full)
                    if (v30Classifier != null) {
                        srcIdx = 0
                        while (srcIdx < samples.size) {
                            val toCopy = minOf(samples.size - srcIdx, v30AccumSize - v30Filled)
                            samples.copyInto(v30Buf, v30Filled, srcIdx, srcIdx + toCopy)
                            v30Filled += toCopy
                            srcIdx += toCopy

                            if (v30Filled == v30AccumSize) {
                                val raw48k = v30Buf.copyOf()
                                launch(Dispatchers.Default) { runV30Inference(raw48k) }

                                v30Buf.copyInto(v30Buf, 0, v30Hop, v30AccumSize)
                                v30Filled = v30AccumSize - v30Hop
                            }
                        }
                    }
                }
        }
    }

    // Mutex per classifier — ONNX/TFLite sessions are NOT thread-safe.
    // With 50% overlap, a new chunk arrives before the previous inference finishes.
    // Mutex serializes access to avoid concurrent inference calls.
    private val birdnetMutex = Mutex()
    private val v30Mutex = Mutex()

    private suspend fun runBirdNetInference(chunk: FloatArray) {
        val classifier = birdnetClassifier ?: return
        val processor = birdnetProcessor ?: return
        birdnetMutex.withLock {
            val chunkNum = ++v24ChunkNum
            try {
                val processed = processor.process(chunk)
                if (processed == null) {
                    birdnetAggregator.addChunkResults(null)
                    Log.d(TAG_V24, "═══ V2.4 #$chunkNum: SKIP (${processor.statsLine()}) ═══")
                    return
                }

                val detections = classifier.classify(processed.samples, sessionLocation)
                birdnetAggregator.addChunkResults(detections)
                val confirmed = birdnetAggregator.getConfirmedDetections()
                    .associate { it.scientificName to it }

                val now = System.currentTimeMillis()
                updateShadowMaps(detections, now, v24FamilyShadow, v24AnchorSpecies)
                val shadowedFamilies = v24FamilyShadow
                    .filter { (_, ts) -> now - ts <= FAMILY_SHADOW_MS }
                    .keys.toSet()
                val recentAnchorSpecies = v24AnchorSpecies
                    .filter { (_, ts) -> now - ts <= FAMILY_SHADOW_MS }
                    .keys.toSet()
                val filtered = filterDetections(detections, confirmed, shadowedFamilies, recentAnchorSpecies)

                val sb = StringBuilder()
                sb.appendLine("═══ V2.4 #$chunkNum ═══════════════════════")
                sb.appendLine(" Pre: rms=%.4f peak=%.4f".format(processed.rms, processed.peak))
                sb.appendLine(" Raw top-5: ${detections.take(5).joinToString(" | ") {
                    "${it.commonName} %.1f%%".format(it.confidence * 100) }}")
                sb.appendLine(" Geo: MetaProfile ${if (classifier.metaProfile != null) "ON" else "OFF"}")
                sb.appendLine(" Shadow: ${shadowedFamilies.size} families")
                sb.appendLine(" Aggregator: ${confirmed.size} confirmed" +
                    if (confirmed.isNotEmpty()) " (${confirmed.values.joinToString(", ") {
                        "${it.scientificName} ×${it.confirmedChunks}" }})" else "")
                val profile = classifier.metaProfile
                for (det in detections.filter { it.confidence >= 0.05f }.take(8)) {
                    val inUI = filtered.any { it.scientificName == det.scientificName }
                    val family = familyTaxonomy.getFamily(det.scientificName)
                    val reason = buildFilterReason(det, confirmed, filtered, family, shadowedFamilies, recentAnchorSpecies)
                    val conf = "%.1f%%".format(det.confidence * 100)
                    val tier = profile?.tierLabel(classifier.metaProfileIndex(det.labelIndex)) ?: "?"
                    sb.appendLine(" ${if (inUI) "✓" else "✗"} ${det.commonName} $conf [$tier] — $reason")
                }
                sb.append("═══════════════════════════════════════")
                Log.d(TAG_V24, sb.toString())

                updateBirdList(filtered, isBirdNet = true)
            } catch (e: Exception) {
                Log.e(TAG_V24, "V2.4 #$chunkNum FAILED", e)
            }
        }
    }

    private suspend fun runV30Inference(raw48k: FloatArray) {
        val classifier = v30Classifier ?: return
        val processor = v30Processor ?: return
        v30Mutex.withLock {
            val chunkNum = ++v30ChunkNum
            try {
                // Resample 48 kHz → 32 kHz
                val resampled = AudioResampler.resample(raw48k, BIRDNET_SAMPLE_RATE, V30_SAMPLE_RATE)

                val processed = processor.process(resampled)
                if (processed == null) {
                    v30Aggregator.addChunkResults(null)
                    Log.d(TAG_V30, "═══ V3.0 #$chunkNum: SKIP (${processor.statsLine()}) ═══")
                    return
                }

                val detections = classifier.classify(processed.samples)
                v30Aggregator.addChunkResults(detections)
                val confirmed = v30Aggregator.getConfirmedDetections()
                    .associate { it.scientificName to it }

                val now = System.currentTimeMillis()
                updateShadowMaps(detections, now, v30FamilyShadow, v30AnchorSpecies)
                val shadowedFamilies = v30FamilyShadow
                    .filter { (_, ts) -> now - ts <= FAMILY_SHADOW_MS }
                    .keys.toSet()
                val recentAnchorSpecies = v30AnchorSpecies
                    .filter { (_, ts) -> now - ts <= FAMILY_SHADOW_MS }
                    .keys.toSet()
                val filtered = filterDetections(detections, confirmed, shadowedFamilies, recentAnchorSpecies)

                val sb = StringBuilder()
                sb.appendLine("═══ V3.0 #$chunkNum ═══════════════════════")
                sb.appendLine(" Resample: ${raw48k.size} @48kHz → ${resampled.size} @32kHz")
                sb.appendLine(" Pre: rms=%.4f peak=%.4f".format(processed.rms, processed.peak))
                sb.appendLine(" Raw top-5: ${detections.take(5).joinToString(" | ") {
                    "${it.commonName} %.1f%%".format(it.confidence * 100) }}")
                sb.appendLine(" Geo: MetaProfile ${if (classifier.metaProfile != null) "ON" else "OFF"}")
                sb.appendLine(" Shadow: ${shadowedFamilies.size} families")
                sb.appendLine(" Aggregator: ${confirmed.size} confirmed" +
                    if (confirmed.isNotEmpty()) " (${confirmed.values.joinToString(", ") {
                        "${it.scientificName} ×${it.confirmedChunks}" }})" else "")
                val profile = classifier.metaProfile
                for (det in detections.filter { it.confidence >= 0.05f }.take(8)) {
                    val inUI = filtered.any { it.scientificName == det.scientificName }
                    val family = familyTaxonomy.getFamily(det.scientificName)
                    val reason = buildFilterReason(det, confirmed, filtered, family, shadowedFamilies, recentAnchorSpecies)
                    val conf = "%.1f%%".format(det.confidence * 100)
                    val tier = profile?.tierLabel(classifier.metaProfileIndex(det.labelIndex)) ?: "?"
                    sb.appendLine(" ${if (inUI) "✓" else "✗"} ${det.commonName} $conf [$tier] — $reason")
                }
                sb.append("═══════════════════════════════════════")
                Log.d(TAG_V30, sb.toString())

                updateBirdList(filtered, isBirdNet = false)
            } catch (e: Exception) {
                Log.e(TAG_V30, "V3.0 #$chunkNum FAILED", e)
            }
        }
    }

    private fun updateBirdList(detections: List<BirdDetection>, isBirdNet: Boolean) {
        if (detections.isEmpty()) return

        val elapsedMs = System.currentTimeMillis() - sessionStartMs
        val detectedAt = formatMmSs(elapsedMs / 1_000)
        var flashId: String? = null
        var isNewBird = false

        _uiState.update { s ->
            val birds = s.birds.toMutableList()
            val newIds = s.newBirdIds.toMutableSet()

            for (det in detections) {
                val conf = (det.confidence * 100).roundToInt()
                val idx = birds.indexOfFirst { it.scientificName == det.scientificName }

                if (idx >= 0) {
                    // Re-detection — flash
                    flashId = birds[idx].id
                    birds[idx] = if (isBirdNet) {
                        birds[idx].copy(
                            v24Confidence = maxOf(birds[idx].v24Confidence ?: 0, conf),
                            detectedAt = detectedAt,
                        )
                    } else {
                        birds[idx].copy(
                            v30Confidence = maxOf(birds[idx].v30Confidence ?: 0, conf),
                            detectedAt = detectedAt,
                        )
                    }
                } else {
                    val id = UUID.randomUUID().toString()
                    flashId = id
                    isNewBird = true
                    newIds.add(id)
                    birds.add(0, DualDetectedBirdUi(
                        id = id,
                        commonName = det.commonName,
                        scientificName = det.scientificName,
                        v24Confidence = if (isBirdNet) conf else null,
                        v30Confidence = if (!isBirdNet) conf else null,
                        detectedAt = detectedAt,
                    ))
                }
            }

            s.copy(birds = birds.take(MAX_DETECTIONS), newBirdIds = newIds)
        }

        flashId?.let { id ->
            clearActiveJob?.cancel()
            _uiState.update {
                it.copy(
                    flashBirdId = id,
                    blipSeq = if (isNewBird) it.blipSeq + 1 else it.blipSeq,
                )
            }
            clearActiveJob = viewModelScope.launch {
                delay(if (isNewBird) AURORA_DURATION_MS else FLASH_DURATION_MS)
                _uiState.update { it.copy(flashBirdId = null) }
                if (isNewBird) {
                    // Remove from newBirdIds after aurora finishes
                    delay(AURORA_DURATION_MS - FLASH_DURATION_MS)
                    _uiState.update { it.copy(newBirdIds = it.newBirdIds - id) }
                }
            }
        }
    }

    private fun updateShadowMaps(
        detections: List<BirdDetection>,
        now: Long,
        familyShadow: HashMap<String, Long>,
        anchorSpecies: HashMap<String, Long>,
    ) {
        for (det in detections) {
            if (det.confidence >= LIVE_MIN_CONFIDENCE) {
                anchorSpecies[det.scientificName] = now
                val family = familyTaxonomy.getFamily(det.scientificName)
                if (family != null) familyShadow[family] = now
            }
        }
    }

    private fun filterDetections(
        detections: List<BirdDetection>,
        confirmedSpecies: Map<String, DetectionAggregator.AggregatedDetection>,
        shadowedFamilies: Set<String>,
        recentAnchorSpecies: Set<String>,
    ): List<BirdDetection> {
        val candidates = mutableListOf<BirdDetection>()

        for (det in detections) {
            val family = familyTaxonomy.getFamily(det.scientificName)
            val familyBlocked = family in shadowedFamilies && det.scientificName !in recentAnchorSpecies
            when {
                det.confidence >= LIVE_MIN_CONFIDENCE -> candidates.add(det)
                !familyBlocked && det.scientificName in confirmedSpecies -> {
                    val agg = confirmedSpecies.getValue(det.scientificName)
                    candidates.add(det.copy(confidence = agg.confidence))
                }
            }
        }

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
            }
        }

        return result
    }

    private fun buildFilterReason(
        det: BirdDetection,
        confirmed: Map<String, DetectionAggregator.AggregatedDetection>,
        filtered: List<BirdDetection>,
        family: String?,
        shadowedFamilies: Set<String>,
        recentAnchorSpecies: Set<String>,
    ): String {
        val isAnchor = det.confidence >= LIVE_MIN_CONFIDENCE
        val isConfirmed = det.scientificName in confirmed
        val isFamilyShadowed = family != null && family in shadowedFamilies
        val isAnchorSpecies = det.scientificName in recentAnchorSpecies
        val inUI = filtered.any { it.scientificName == det.scientificName }

        return when {
            isAnchor && inUI -> "anchor (>=${(LIVE_MIN_CONFIDENCE * 100).toInt()}%), family=$family"
            isAnchor && !inUI -> "anchor but family-deduped ($family)"
            !isAnchor && isFamilyShadowed && !isAnchorSpecies && isConfirmed ->
                "family-shadowed ($family had recent anchor)"
            !isAnchor && isConfirmed && inUI -> {
                val agg = confirmed[det.scientificName]!!
                val exempt = if (isFamilyShadowed) ", shadow-exempt" else ""
                "aggregator-confirmed (x${agg.confirmedChunks}), family=$family$exempt"
            }
            !isAnchor && isConfirmed && !inUI -> "confirmed but family-deduped ($family)"
            else -> "below threshold / not confirmed"
        }
    }

    private fun resolveLocation(): LocationMeta? {
        val hasPermission = context.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
        if (!hasPermission) return null
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val loc = lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            ?: lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            ?: return null
        val week = LocalDate.now().get(WeekFields.ISO.weekOfWeekBasedYear())
        val lo = week - LIVE_WEEK_WINDOW
        val hi = week + LIVE_WEEK_WINDOW
        val weekRange = if (lo < 1 || hi > 52) 1..52 else lo..hi
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
                _uiState.update { it.copy(sessionTimer = formatMmSs(elapsedMs / 1_000)) }
            }
        }
    }

    private fun formatMmSs(totalSec: Long): String =
        "%02d:%02d".format(totalSec / 60, totalSec % 60)

    override fun onCleared() {
        onStop()
    }

    companion object {
        private const val TAG = "DualDetectionVM"
        private const val TAG_V24 = "DualV24"
        private const val TAG_V30 = "DualV30"
        private const val MAX_DETECTIONS = 100
        private const val LIVE_WEEK_WINDOW = 4
        private const val LIVE_MIN_CONFIDENCE = 0.80f
        // Shadow duration covers the longest aggregator window: 8 chunks × 2.5s hop (V3.0) = 20s
        private const val V30_CHUNK_SEC = 5
        private const val FAMILY_SHADOW_MS = DetectionAggregator.DEFAULT_WINDOW_SIZE * V30_CHUNK_SEC * 1_000L / 2
        private const val AGGREGATOR_THRESHOLD = 0.10f
        private const val AGGREGATOR_CONFIRMATION = 2

        private const val BIRDNET_SAMPLE_RATE = 48_000
        private const val BIRDNET_CHUNK_SEC = 3
        private const val V30_SAMPLE_RATE = 32_000

        private const val FLASH_DURATION_MS = 1_400L
        private const val AURORA_DURATION_MS = 6_000L
        private const val LURE_DURATION_MS = 8_000L
    }
}
