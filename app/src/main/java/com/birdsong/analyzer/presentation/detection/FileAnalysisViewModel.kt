package com.birdsong.analyzer.presentation.detection

import android.content.Context
import android.net.Uri
import android.text.format.Formatter
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.model.FileAnalysisEntity
import com.birdsong.analyzer.data.model.FileDetectionEntity
import com.birdsong.analyzer.data.repository.FileAnalysisRepository
import com.birdsong.analyzer.data.repository.GeoRepository
import com.birdsong.analyzer.ml.BirdDetectionPipeline
import com.birdsong.analyzer.ml.BoundingBox
import com.birdsong.analyzer.ml.ClassifierFactory
import com.birdsong.analyzer.ml.MetaProfile
import com.birdsong.analyzer.ml.MetaProfileBuilder
import com.birdsong.analyzer.ml.TimelineBuilder
import com.birdsong.analyzer.ml.TimelineSegment
import com.birdsong.analyzer.ml.IncrementalWaveformBuilder
import com.birdsong.analyzer.ml.WaveformData
import com.birdsong.analyzer.ml.WaveformExtractor
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.withLock
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import javax.inject.Inject
import kotlin.math.max
import kotlin.math.roundToInt

data class FileAnalysisHistoryItem(
    val id: String,
    val fileName: String,
    val date: String,
    val speciesCount: Int,
    val regionLabel: String,
    val durationLabel: String,
)

enum class FileAnalysisState { IDLE, ANALYZING, PAUSED, DONE, ERROR }

data class ModelProgress(
    val chunksProcessed: Int = 0,
    val totalChunks: Int = 0,
    val lastProcessedTimeSec: Float = 0f,
)

data class FileTimelineBirdUi(
    val id: String,
    val commonName: String,
    val scientificName: String,
    val startTimeSec: Float,
    val endTimeSec: Float,
    val timeRange: String,
    val v24Confidence: Int?,
    val v30Confidence: Int?,
)

data class SpeciesSegmentUi(
    val startSec: Float,
    val endSec: Float,
    val timeRange: String,
)

data class FileSpeciesSummary(
    val scientificName: String,
    val commonName: String,
    val maxV24Confidence: Int?,
    val maxV30Confidence: Int?,
    val detectionCount: Int,
    val segments: List<SpeciesSegmentUi>,
)

data class FileAnalysisUiState(
    val state: FileAnalysisState = FileAnalysisState.IDLE,
    val fileName: String = "",
    val fileDurationSec: Float = 0f,
    val fileSizeLabel: String = "",
    val v24Progress: ModelProgress = ModelProgress(),
    val v30Progress: ModelProgress = ModelProgress(),
    val timelineBirds: List<FileTimelineBirdUi> = emptyList(),
    val speciesSummaries: List<FileSpeciesSummary> = emptyList(),
    val selectedSpecies: String? = null,
    val waveformAmplitudes: FloatArray? = null,
    val waveformProgress: Float = 0f,
    val progressLabel: String = "",
    val v30Available: Boolean = false,
    val geoLabel: String = "—",
    val geoConfigured: Boolean = false,
    val hasWaveformData: Boolean = false,
    val errorMessage: String = "",
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is FileAnalysisUiState) return false
        return state == other.state && fileName == other.fileName &&
            fileDurationSec == other.fileDurationSec && fileSizeLabel == other.fileSizeLabel &&
            v24Progress == other.v24Progress && v30Progress == other.v30Progress &&
            timelineBirds == other.timelineBirds && speciesSummaries == other.speciesSummaries &&
            selectedSpecies == other.selectedSpecies &&
            (waveformAmplitudes === other.waveformAmplitudes ||
                waveformAmplitudes != null && other.waveformAmplitudes != null &&
                waveformAmplitudes.contentEquals(other.waveformAmplitudes)) &&
            waveformProgress == other.waveformProgress &&
            progressLabel == other.progressLabel &&
            v30Available == other.v30Available && geoLabel == other.geoLabel &&
            geoConfigured == other.geoConfigured &&
            hasWaveformData == other.hasWaveformData &&
            errorMessage == other.errorMessage
    }

    override fun hashCode(): Int = state.hashCode() * 31 + fileName.hashCode()
}

@HiltViewModel
class FileAnalysisViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val pipeline: BirdDetectionPipeline,
    private val classifierFactory: ClassifierFactory,
    private val metaProfileBuilder: MetaProfileBuilder,
    private val geoRepository: GeoRepository,
    private val fileAnalysisRepository: FileAnalysisRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(FileAnalysisUiState(
        v30Available = classifierFactory.isBirdNetV30Available(),
    ))
    val uiState: StateFlow<FileAnalysisUiState> = _uiState.asStateFlow()

    val recentAnalyses: StateFlow<List<FileAnalysisHistoryItem>> =
        fileAnalysisRepository.getAllSummaries()
            .map { summaries ->
                summaries.map { s ->
                    FileAnalysisHistoryItem(
                        id = s.id,
                        fileName = s.fileName,
                        date = formatDate(s.createdAt),
                        speciesCount = s.speciesCount,
                        regionLabel = s.regionLabel ?: "—",
                        durationLabel = formatMmSs(s.durationSec),
                    )
                }
            }
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    private var metaProfileJob: Job? = null
    private var analysisJob: Job? = null

    @Volatile
    private var cachedMetaProfile: MetaProfile? = null

    private val v24Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val v30Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val recordsMutex = Mutex()

    private var v24ChunkDuration = 0f
    private var v30ChunkDuration = 0f

    private val _isPaused = MutableStateFlow(false)
    private var pendingTimelineRebuild = false

    private var currentUri: Uri? = null
    private var currentFileSize: Long = 0L
    private var waveformBuilder: IncrementalWaveformBuilder? = null
    private var analysisStartTimeMs: Long = 0L

    init {
        buildMetaProfileAsync()
        observeGeo()
    }

    private fun buildMetaProfileAsync() {
        metaProfileJob = viewModelScope.launch {
            try {
                val geo = geoRepository.resolveCurrentGeo() ?: return@launch
                Log.d(TAG, "Building MetaProfile for ${geo.nameEn} (${geo.code})")
                val bbox = BoundingBox(geo.minLat!!, geo.maxLat!!, geo.minLon!!, geo.maxLon!!)
                cachedMetaProfile = metaProfileBuilder.build(bbox, geo.bufferDeg)
                Log.d(TAG, "MetaProfile ready")
            } catch (e: Exception) {
                Log.e(TAG, "MetaProfile build failed", e)
            }
        }
    }

    private fun observeGeo() {
        viewModelScope.launch {
            geoRepository.currentSelectionDisplay.collect { label ->
                _uiState.update { it.copy(geoLabel = label) }
            }
        }
        viewModelScope.launch {
            geoRepository.countryCode.collect { code ->
                _uiState.update { it.copy(geoConfigured = code.isNotEmpty()) }
            }
        }
    }

    fun selectFile(uri: Uri, fileName: String) {
        analysisJob?.cancel()
        _isPaused.value = false
        currentUri = uri
        waveformBuilder = null

        viewModelScope.launch {
            try {
                val (duration, fileSize) = withContext(Dispatchers.IO) {
                    WaveformExtractor.extractDuration(context, uri) to
                        WaveformExtractor.extractFileSize(context, uri)
                }
                currentFileSize = fileSize
                _uiState.update {
                    FileAnalysisUiState(
                        state = FileAnalysisState.IDLE,
                        fileName = fileName,
                        fileDurationSec = duration,
                        fileSizeLabel = Formatter.formatFileSize(context, fileSize),
                        v30Available = classifierFactory.isBirdNetV30Available(),
                        geoLabel = it.geoLabel,
                        geoConfigured = it.geoConfigured,
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "selectFile failed", e)
                _uiState.update {
                    it.copy(
                        state = FileAnalysisState.ERROR,
                        errorMessage = e.message ?: "Failed to read file",
                    )
                }
            }
        }
    }

    fun startAnalysis() {
        val uri = currentUri ?: return
        analysisJob?.cancel()
        _isPaused.value = false

        analysisJob = viewModelScope.launch {
            v24Records.clear()
            v30Records.clear()
            v24ChunkDuration = 0f
            v30ChunkDuration = 0f

            analysisStartTimeMs = System.currentTimeMillis()
            val v30Available = classifierFactory.isBirdNetV30Available()
            _uiState.update {
                it.copy(
                    state = FileAnalysisState.ANALYZING,
                    v30Available = v30Available,
                    timelineBirds = emptyList(),
                    speciesSummaries = emptyList(),
                    selectedSpecies = null,
                    waveformAmplitudes = null,
                    waveformProgress = 0f,
                )
            }

            val v24Classifier = classifierFactory.createBirdNet()
            val v30Classifier = if (v30Available) classifierFactory.createBirdNetV30() else null
            var v24Job: Job? = null
            var v30Job: Job? = null

            try {
                metaProfileJob?.join()
                cachedMetaProfile?.let { mp ->
                    v24Classifier.metaProfile = mp
                    v30Classifier?.metaProfile = mp
                }

                v24ChunkDuration = v24Classifier.chunkDurationSeconds.toFloat()
                v30ChunkDuration = v30Classifier?.chunkDurationSeconds?.toFloat() ?: 0f

                val fileDuration = _uiState.value.fileDurationSec
                val estimatedChunks = if (v24ChunkDuration > 0f) {
                    kotlin.math.ceil(fileDuration / v24ChunkDuration).toInt().coerceAtLeast(1)
                } else 1
                val builder = IncrementalWaveformBuilder(
                    totalChunks = estimatedChunks,
                )
                waveformBuilder = builder
                var waveformChunkCount = 0

                // Show empty waveform immediately so UI has the full-width frame
                _uiState.update { it.copy(waveformAmplitudes = FloatArray(400)) }

                val v24Processor = classifierFactory.createProcessor(v24Classifier)
                val v30Processor = v30Classifier?.let { classifierFactory.createProcessor(it) }

                v24Job = launch {
                    pipeline.analyzeFileDetailed(
                        context = context,
                        uri = uri,
                        classifier = v24Classifier,
                        processor = v24Processor,
                        classifierFactory = classifierFactory,
                        numWorkers = 1,
                        pauseState = _isPaused,
                        onProgress = { progress ->
                            _uiState.update {
                                it.copy(
                                    v24Progress = ModelProgress(
                                        chunksProcessed = progress.processedChunks,
                                        totalChunks = progress.totalChunks,
                                        lastProcessedTimeSec = progress.lastProcessedTimeSec,
                                    ),
                                )
                            }
                            updateWaveformProgress()
                        },
                        onChunkResult = { record ->
                            recordsMutex.withLock { v24Records.add(record) }
                            pendingTimelineRebuild = true
                        },
                        onRawChunk = { _, _, samples ->
                            builder.addChunk(samples)
                            waveformChunkCount++
                            if (waveformChunkCount % WAVEFORM_SNAPSHOT_INTERVAL == 0) {
                                val waveformSnapshot = builder.snapshot()
                                _uiState.update { it.copy(waveformAmplitudes = waveformSnapshot) }
                                if (pendingTimelineRebuild) {
                                    pendingTimelineRebuild = false
                                    viewModelScope.launch { rebuildTimeline() }
                                }
                            }
                        },
                    )
                }

                v30Job = if (v30Classifier != null && v30Processor != null) {
                    launch {
                        pipeline.analyzeFileDetailed(
                            context = context,
                            uri = uri,
                            classifier = v30Classifier,
                            processor = v30Processor,
                            classifierFactory = classifierFactory,
                            numWorkers = 1,
                            pauseState = _isPaused,
                            onProgress = { progress ->
                                _uiState.update {
                                    it.copy(
                                        v30Progress = ModelProgress(
                                            chunksProcessed = progress.processedChunks,
                                            totalChunks = progress.totalChunks,
                                            lastProcessedTimeSec = progress.lastProcessedTimeSec,
                                        ),
                                    )
                                }
                                updateWaveformProgress()
                            },
                            onChunkResult = { record ->
                                recordsMutex.withLock { v30Records.add(record) }
                                pendingTimelineRebuild = true
                            },
                        )
                    }
                } else null

                v24Job.join()
                v30Job?.join()

                rebuildTimeline()
                _uiState.update {
                    it.copy(
                        state = FileAnalysisState.DONE,
                        waveformProgress = 1f,
                        waveformAmplitudes = builder.build(),
                    )
                }
                Log.d(TAG, "File analysis done: ${_uiState.value.timelineBirds.size} timeline segments")

                saveToHistory()
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) {
                    // Cancelled by user — keep results if any
                    if (_uiState.value.timelineBirds.isNotEmpty()) {
                        _uiState.update { it.copy(state = FileAnalysisState.DONE) }
                        viewModelScope.launch { saveToHistory() }
                    } else {
                        _uiState.update { it.copy(state = FileAnalysisState.IDLE) }
                    }
                } else {
                    Log.e(TAG, "File analysis failed", e)
                    _uiState.update {
                        it.copy(
                            state = FileAnalysisState.ERROR,
                            errorMessage = e.message ?: "Unknown error",
                        )
                    }
                }
            } finally {
                // Wait for workers to finish native calls before closing classifiers
                withContext(NonCancellable) {
                    v24Job?.join()
                    v30Job?.join()
                }
                v24Classifier.close()
                v30Classifier?.close()
            }
        }
    }

    fun pauseAnalysis() {
        _isPaused.value = true
        _uiState.update { it.copy(state = FileAnalysisState.PAUSED) }
    }

    fun resumeAnalysis() {
        _isPaused.value = false
        _uiState.update { it.copy(state = FileAnalysisState.ANALYZING) }
    }

    fun cancelAnalysis() {
        _isPaused.value = false
        analysisJob?.cancel()
    }

    fun selectSpecies(scientificName: String?) {
        val current = _uiState.value.selectedSpecies
        val newSelection = if (current == scientificName) null else scientificName
        _uiState.update { it.copy(selectedSpecies = newSelection) }
    }

    fun loadFromHistory(analysisId: String) {
        viewModelScope.launch {
            try {
                val analysis = fileAnalysisRepository.getAnalysisById(analysisId) ?: return@launch
                val detections = fileAnalysisRepository.getDetectionsForAnalysis(analysisId)

                val amplitudes = analysis.waveformData?.let { bytes ->
                    WaveformData.fromByteArray(bytes, analysis.durationSec, analysis.fileSizeBytes)
                        .amplitudes
                }

                val summaries = buildSpeciesSummaries(detections.map { det ->
                    TimelineSegment(
                        scientificName = det.scientificName,
                        commonName = det.commonName,
                        startTimeSec = det.startTimeSec,
                        endTimeSec = det.endTimeSec,
                        v24Confidence = det.v24Confidence,
                        v30Confidence = det.v30Confidence,
                    )
                })

                val birds = detections.mapIndexed { idx, det ->
                    FileTimelineBirdUi(
                        id = "${det.scientificName}_$idx",
                        commonName = det.commonName,
                        scientificName = det.scientificName,
                        startTimeSec = det.startTimeSec,
                        endTimeSec = det.endTimeSec,
                        timeRange = formatTimeRange(det.startTimeSec, det.endTimeSec),
                        v24Confidence = det.v24Confidence?.let { (it * 100).roundToInt() },
                        v30Confidence = det.v30Confidence?.let { (it * 100).roundToInt() },
                    )
                }

                _uiState.update {
                    FileAnalysisUiState(
                        state = FileAnalysisState.DONE,
                        fileName = analysis.fileName,
                        fileDurationSec = analysis.durationSec,
                        fileSizeLabel = Formatter.formatFileSize(context, analysis.fileSizeBytes),
                        waveformAmplitudes = amplitudes,
                        waveformProgress = 1f,
                        timelineBirds = birds,
                        speciesSummaries = summaries,
                        v30Available = analysis.v30Available,
                        geoLabel = analysis.regionLabel ?: "—",
                        geoConfigured = it.geoConfigured,
                        hasWaveformData = analysis.waveformData != null && amplitudes == null,
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "loadFromHistory failed", e)
            }
        }
    }

    fun loadWaveform() {
        viewModelScope.launch {
            try {
                val state = _uiState.value
                if (state.state != FileAnalysisState.DONE) return@launch
                val analysis = fileAnalysisRepository.getAllSummaries()
                    .first().firstOrNull { it.fileName == state.fileName } ?: return@launch
                val full = fileAnalysisRepository.getAnalysisById(analysis.id) ?: return@launch
                val amplitudes = full.waveformData?.let { bytes ->
                    WaveformData.fromByteArray(bytes, full.durationSec, full.fileSizeBytes).amplitudes
                }
                _uiState.update { it.copy(waveformAmplitudes = amplitudes, hasWaveformData = false) }
            } catch (e: Exception) {
                Log.e(TAG, "loadWaveform failed", e)
            }
        }
    }

    private fun updateWaveformProgress() {
        val state = _uiState.value
        val fileDuration = state.fileDurationSec
        if (fileDuration <= 0f) return

        // End of analyzed region = last chunk position + chunk duration
        val v24EndSec = state.v24Progress.lastProcessedTimeSec + v24ChunkDuration
        val v30EndSec = if (state.v30Available && v30ChunkDuration > 0f) {
            state.v30Progress.lastProcessedTimeSec + v30ChunkDuration
        } else fileDuration // no V3.0 → not a bottleneck

        // Progress = slowest model's position / file duration
        val processedSec = kotlin.math.min(v24EndSec, v30EndSec)
        val progress = (processedSec / fileDuration).coerceIn(0f, 1f)
        val label = "${(progress * 100).roundToInt()}%"
        _uiState.update { it.copy(waveformProgress = progress, progressLabel = label) }
    }

    private suspend fun rebuildTimeline() {
        val (v24Snapshot, v30Snapshot) = recordsMutex.withLock {
            ArrayList(v24Records) to ArrayList(v30Records)
        }

        val v24Partial = if (v24Snapshot.isNotEmpty()) {
            BirdDetectionPipeline.FileAnalysisResult(
                chunkRecords = v24Snapshot,
                chunkDurationSec = v24ChunkDuration,
            )
        } else null

        val v30Partial = if (v30Snapshot.isNotEmpty()) {
            BirdDetectionPipeline.FileAnalysisResult(
                chunkRecords = v30Snapshot,
                chunkDurationSec = v30ChunkDuration,
            )
        } else null

        val segments = TimelineBuilder.build(v24Partial, v30Partial)
            .filter { seg ->
                val v24 = seg.v24Confidence
                val v30 = seg.v30Confidence
                (v24 != null && v24 >= HIGH_CONFIDENCE) ||
                    (v30 != null && v30 >= HIGH_CONFIDENCE) ||
                    (v24 != null && v24 >= MIN_CONFIDENCE && v30 != null && v30 >= MIN_CONFIDENCE)
            }

        val birds = segments.mapIndexed { idx, seg ->
            FileTimelineBirdUi(
                id = "${seg.scientificName}_$idx",
                commonName = seg.commonName,
                scientificName = seg.scientificName,
                startTimeSec = seg.startTimeSec,
                endTimeSec = seg.endTimeSec,
                timeRange = formatTimeRange(seg.startTimeSec, seg.endTimeSec),
                v24Confidence = seg.v24Confidence?.let { (it * 100).roundToInt() },
                v30Confidence = seg.v30Confidence?.let { (it * 100).roundToInt() },
            )
        }

        val summaries = buildSpeciesSummaries(segments)

        _uiState.update { it.copy(timelineBirds = birds, speciesSummaries = summaries) }
    }

    private fun buildSpeciesSummaries(
        segments: List<TimelineSegment>,
    ): List<FileSpeciesSummary> {
        return segments.groupBy { it.scientificName }.map { (sciName, segs) ->
            val first = segs.first()
            FileSpeciesSummary(
                scientificName = sciName,
                commonName = first.commonName,
                maxV24Confidence = segs.mapNotNull { it.v24Confidence }
                    .maxOrNull()?.let { (it * 100).roundToInt() },
                maxV30Confidence = segs.mapNotNull { it.v30Confidence }
                    .maxOrNull()?.let { (it * 100).roundToInt() },
                detectionCount = segs.size,
                segments = segs.map { seg ->
                    SpeciesSegmentUi(
                        startSec = seg.startTimeSec,
                        endSec = seg.endTimeSec,
                        timeRange = formatTimeRange(seg.startTimeSec, seg.endTimeSec),
                    )
                },
            )
        }
    }

    private suspend fun saveToHistory() {
        try {
            val state = _uiState.value
            val uri = currentUri ?: return
            val analysisId = UUID.randomUUID().toString()

            val geoLabel = state.geoLabel.takeIf { it != "—" }
            val regionCode = geoRepository.regionCode.first()
                ?: geoRepository.countryCode.first().takeIf { it.isNotEmpty() }

            val waveformAmplitudes = waveformBuilder?.build()
            val waveformBytes = waveformAmplitudes?.let {
                WaveformData(it, state.fileDurationSec, currentFileSize).toByteArray()
            }

            val analysisDuration = System.currentTimeMillis() - analysisStartTimeMs

            val entity = FileAnalysisEntity(
                id = analysisId,
                fileName = state.fileName,
                fileUri = uri.toString(),
                durationSec = state.fileDurationSec,
                fileSizeBytes = currentFileSize,
                regionCode = regionCode,
                regionLabel = geoLabel,
                v30Available = state.v30Available,
                waveformData = waveformBytes,
                createdAt = System.currentTimeMillis(),
                speciesCount = state.speciesSummaries.size,
                analysisDurationMs = analysisDuration,
            )

            val detections = state.timelineBirds.map { bird ->
                FileDetectionEntity(
                    id = UUID.randomUUID().toString(),
                    analysisId = analysisId,
                    scientificName = bird.scientificName,
                    commonName = bird.commonName,
                    startTimeSec = bird.startTimeSec,
                    endTimeSec = bird.endTimeSec,
                    v24Confidence = bird.v24Confidence?.let { it / 100f },
                    v30Confidence = bird.v30Confidence?.let { it / 100f },
                )
            }

            fileAnalysisRepository.saveAnalysis(entity, detections)
            Log.d(TAG, "Saved analysis $analysisId with ${detections.size} detections")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save analysis to history", e)
        }
    }

    private fun formatTimeRange(startSec: Float, endSec: Float): String {
        return "${formatMmSs(startSec)} – ${formatMmSs(endSec)}"
    }

    private fun formatMmSs(totalSec: Float): String {
        val sec = totalSec.toInt()
        return "%d:%02d".format(sec / 60, sec % 60)
    }

    fun deleteFromHistory(id: String) {
        viewModelScope.launch {
            fileAnalysisRepository.deleteAnalysis(id)
        }
    }

    private fun formatDate(epochMs: Long): String {
        val fmt = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
        return fmt.format(Date(epochMs))
    }

    companion object {
        private const val TAG = "FileAnalysisVM"
        private const val HIGH_CONFIDENCE = 0.8f
        private const val MIN_CONFIDENCE = 0.4f
        private const val WAVEFORM_SNAPSHOT_INTERVAL = 5
    }
}
