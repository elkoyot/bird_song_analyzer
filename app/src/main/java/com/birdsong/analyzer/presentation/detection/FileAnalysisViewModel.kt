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
import com.birdsong.analyzer.ml.IncrementalWaveformBuilder
import com.birdsong.analyzer.ml.MetaProfile
import com.birdsong.analyzer.ml.MetaProfileBuilder
import com.birdsong.analyzer.ml.SpectrogramComputer
import com.birdsong.analyzer.ml.TimelineBuilder
import com.birdsong.analyzer.ml.TimelineSegment
import com.birdsong.analyzer.ml.WaveformData
import com.birdsong.analyzer.ml.WaveformExtractor
import com.birdsong.analyzer.service.AudioPlaybackManager
import com.birdsong.analyzer.service.PlaybackState
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.UUID
import javax.inject.Inject
import kotlin.math.ceil
import kotlin.math.min
import kotlin.math.roundToInt

// ── Phase ──

enum class FileAnalysisPhase { IDLE, READY, ANALYZING, PAUSED, DONE, ERROR }

// ── Split UI state models ──

data class ModelProgress(
    val chunksProcessed: Int = 0,
    val totalChunks: Int = 0,
    val lastProcessedTimeSec: Float = 0f,
)

data class FileAnalysisCoreState(
    val phase: FileAnalysisPhase = FileAnalysisPhase.IDLE,
    val fileName: String = "",
    val fileDurationSec: Float = 0f,
    val fileSizeLabel: String = "",
    val fileDurationLabel: String = "",
    val v30Available: Boolean = false,
    val geoLabel: String = "\u2014",
    val geoConfigured: Boolean = false,
    val errorMessage: String = "",
)

data class AnalysisProgressState(
    val v24Progress: ModelProgress = ModelProgress(),
    val v30Progress: ModelProgress = ModelProgress(),
    val progress: Float = 0f,
    val elapsedSec: Int = 0,
)

data class SpectrogramUiState(
    val columns: List<FloatArray> = emptyList(),
    val birdMarkers: List<BirdMarker> = emptyList(),
    val highlightedSpecies: String? = null,
)

data class TimelineUiState(
    val timelineBirds: List<FileTimelineBirdUi> = emptyList(),
    val speciesSummaries: List<FileSpeciesSummary> = emptyList(),
)

data class FilePlaybackUiState(
    val isPlaying: Boolean = false,
    val position: Float = 0f,
    val positionLabel: String = "0:00",
)

// ── Shared UI models ──

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

data class BirdMarker(
    val scientificName: String,
    val position: Float,
    val confidence: Float,
)

// ── ViewModel ──

@HiltViewModel
class FileAnalysisViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val pipeline: BirdDetectionPipeline,
    private val classifierFactory: ClassifierFactory,
    private val metaProfileBuilder: MetaProfileBuilder,
    private val geoRepository: GeoRepository,
    private val fileAnalysisRepository: FileAnalysisRepository,
    val playbackManager: AudioPlaybackManager,
) : ViewModel() {

    // ── Split state flows ──

    private val _coreState = MutableStateFlow(FileAnalysisCoreState(
        v30Available = classifierFactory.isBirdNetV30Available(),
    ))
    val coreState: StateFlow<FileAnalysisCoreState> = _coreState.asStateFlow()

    private val _progressState = MutableStateFlow(AnalysisProgressState())
    val progressState: StateFlow<AnalysisProgressState> = _progressState.asStateFlow()

    private val _spectrogramState = MutableStateFlow(SpectrogramUiState())
    val spectrogramState: StateFlow<SpectrogramUiState> = _spectrogramState.asStateFlow()

    private val _timelineState = MutableStateFlow(TimelineUiState())
    val timelineState: StateFlow<TimelineUiState> = _timelineState.asStateFlow()

    private val _playbackUiState = MutableStateFlow(FilePlaybackUiState())
    val playbackUiState: StateFlow<FilePlaybackUiState> = _playbackUiState.asStateFlow()

    // ── Internal state ──

    private var metaProfileJob: Job? = null
    private var analysisJob: Job? = null
    private var elapsedJob: Job? = null

    @Volatile
    private var cachedMetaProfile: MetaProfile? = null

    private val v24Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val v30Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val recordsMutex = Mutex()

    private var v24ChunkDuration = 0f
    private var v30ChunkDuration = 0f

    private val _isPaused = MutableStateFlow(false)
    private var pendingTimelineRebuild = false
    private var v24ProgressCount = 0

    private var currentUri: Uri? = null
    private var currentFileSize: Long = 0L
    private var waveformBuilder: IncrementalWaveformBuilder? = null
    private var analysisStartTimeMs: Long = 0L

    init {
        buildMetaProfileAsync()
        observeGeo()
        observePlayback()
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
                _coreState.update { it.copy(geoLabel = label) }
            }
        }
        viewModelScope.launch {
            geoRepository.countryCode.collect { code ->
                _coreState.update { it.copy(geoConfigured = code.isNotEmpty()) }
            }
        }
    }

    private fun observePlayback() {
        viewModelScope.launch {
            playbackManager.state.collect { state ->
                _playbackUiState.update { it.copy(isPlaying = state == PlaybackState.PLAYING) }
            }
        }
        viewModelScope.launch {
            playbackManager.positionMs.collect { posMs ->
                val dur = playbackManager.durationMs.value
                if (dur > 0) {
                    _playbackUiState.update {
                        it.copy(
                            position = (posMs.toFloat() / dur).coerceIn(0f, 1f),
                            positionLabel = formatMmSs(posMs / 1000f),
                        )
                    }
                }
            }
        }
    }

    // ── File selection ──

    fun selectFile(uri: Uri, fileName: String) {
        analysisJob?.cancel()
        _isPaused.value = false
        playbackManager.release()
        currentUri = uri
        waveformBuilder = null

        viewModelScope.launch {
            try {
                val (duration, fileSize) = withContext(Dispatchers.IO) {
                    WaveformExtractor.extractDuration(context, uri) to
                        WaveformExtractor.extractFileSize(context, uri)
                }
                currentFileSize = fileSize
                _coreState.update {
                    FileAnalysisCoreState(
                        phase = FileAnalysisPhase.READY,
                        fileName = fileName,
                        fileDurationSec = duration,
                        fileSizeLabel = Formatter.formatFileSize(context, fileSize),
                        fileDurationLabel = formatMmSs(duration),
                        v30Available = classifierFactory.isBirdNetV30Available(),
                        geoLabel = it.geoLabel,
                        geoConfigured = it.geoConfigured,
                    )
                }
                _progressState.value = AnalysisProgressState()
                _spectrogramState.value = SpectrogramUiState()
                _timelineState.value = TimelineUiState()
                _playbackUiState.value = FilePlaybackUiState()
            } catch (e: Exception) {
                Log.e(TAG, "selectFile failed", e)
                _coreState.update {
                    it.copy(
                        phase = FileAnalysisPhase.ERROR,
                        errorMessage = e.message ?: "Failed to read file",
                    )
                }
            }
        }
    }

    fun resetFile() {
        analysisJob?.cancel()
        _isPaused.value = false
        playbackManager.release()
        stopElapsedTimer()
        currentUri = null
        _coreState.update {
            FileAnalysisCoreState(
                v30Available = classifierFactory.isBirdNetV30Available(),
                geoLabel = it.geoLabel,
                geoConfigured = it.geoConfigured,
            )
        }
        _progressState.value = AnalysisProgressState()
        _spectrogramState.value = SpectrogramUiState()
        _timelineState.value = TimelineUiState()
        _playbackUiState.value = FilePlaybackUiState()
    }

    // ── Analysis ──

    fun startAnalysis() {
        val uri = currentUri ?: return
        analysisJob?.cancel()
        _isPaused.value = false
        playbackManager.release()

        analysisJob = viewModelScope.launch {
            v24Records.clear()
            v30Records.clear()
            v24ChunkDuration = 0f
            v30ChunkDuration = 0f
            v24ProgressCount = 0
            pendingTimelineRebuild = false

            analysisStartTimeMs = System.currentTimeMillis()
            val v30Available = classifierFactory.isBirdNetV30Available()
            _coreState.update {
                it.copy(phase = FileAnalysisPhase.ANALYZING, v30Available = v30Available)
            }
            _progressState.value = AnalysisProgressState()
            _spectrogramState.value = SpectrogramUiState()
            _timelineState.value = TimelineUiState()
            startElapsedTimer()

            val v24Classifier = classifierFactory.createBirdNet()
            val v30Classifier = if (v30Available) classifierFactory.createBirdNetV30() else null

            try {
                metaProfileJob?.join()
                cachedMetaProfile?.let { mp ->
                    v24Classifier.metaProfile = mp
                    v30Classifier?.metaProfile = mp
                }

                v24ChunkDuration = v24Classifier.chunkDurationSeconds.toFloat()
                v30ChunkDuration = v30Classifier?.chunkDurationSeconds?.toFloat() ?: 0f

                val fileDuration = _coreState.value.fileDurationSec
                val estimatedChunks = if (v24ChunkDuration > 0f) {
                    ceil(fileDuration / v24ChunkDuration).toInt().coerceAtLeast(1)
                } else 1

                // 1. Async spectrogram/waveform via Channel (doesn't block producer)
                val spectrogramChannel = Channel<FloatArray>(Channel.UNLIMITED)
                val wfBuilder = IncrementalWaveformBuilder(totalChunks = estimatedChunks)
                waveformBuilder = wfBuilder

                val specJob = launch(Dispatchers.Default) {
                    val specComputer = SpectrogramComputer()
                    var rawChunkCount = 0
                    for (samples in spectrogramChannel) {
                        wfBuilder.addChunk(samples)
                        specComputer.addChunk(samples)
                        rawChunkCount++
                        if (rawChunkCount % SPECTROGRAM_SNAPSHOT_INTERVAL == 0) {
                            _spectrogramState.update { it.copy(columns = specComputer.snapshot()) }
                        }
                    }
                    _spectrogramState.update { it.copy(columns = specComputer.build()) }
                }

                // 2. V2.4 streaming analysis (numWorkers=2, decodeChunked — no OOM)
                val v24Processor = classifierFactory.createProcessor(v24Classifier)
                val v24Job = launch {
                    pipeline.analyzeFileDetailed(
                        context = context,
                        uri = uri,
                        classifier = v24Classifier,
                        processor = v24Processor,
                        classifierFactory = classifierFactory,
                        numWorkers = NUM_WORKERS,
                        pauseState = _isPaused,
                        onProgress = { progress ->
                            _progressState.update {
                                it.copy(
                                    v24Progress = ModelProgress(
                                        chunksProcessed = progress.processedChunks,
                                        totalChunks = progress.totalChunks,
                                        lastProcessedTimeSec = progress.lastProcessedTimeSec,
                                    ),
                                )
                            }
                            updateProgress()
                            // Throttled timeline rebuild
                            v24ProgressCount++
                            if (pendingTimelineRebuild && v24ProgressCount % TIMELINE_REBUILD_INTERVAL == 0) {
                                pendingTimelineRebuild = false
                                viewModelScope.launch { rebuildTimeline() }
                            }
                        },
                        onChunkResult = { record ->
                            Log.d(TAG, "V2.4 chunk#${record.chunkIndex} @%.1fs: ${record.detections.size} det".format(record.startTimeSec))
                            recordsMutex.withLock { v24Records.add(record) }
                            pendingTimelineRebuild = true
                        },
                        onRawChunk = { _, _, samples ->
                            spectrogramChannel.trySend(samples)
                        },
                    )
                }

                // 3. V3.0 streaming analysis (numWorkers=2, separate decode)
                val v30Job = if (v30Classifier != null) {
                    val v30Processor = classifierFactory.createProcessor(v30Classifier)
                    launch {
                        pipeline.analyzeFileDetailed(
                            context = context,
                            uri = uri,
                            classifier = v30Classifier,
                            processor = v30Processor,
                            classifierFactory = classifierFactory,
                            numWorkers = NUM_WORKERS,
                            pauseState = _isPaused,
                            onProgress = { progress ->
                                _progressState.update {
                                    it.copy(
                                        v30Progress = ModelProgress(
                                            chunksProcessed = progress.processedChunks,
                                            totalChunks = progress.totalChunks,
                                            lastProcessedTimeSec = progress.lastProcessedTimeSec,
                                        ),
                                    )
                                }
                                updateProgress()
                            },
                            onChunkResult = { record ->
                                Log.d(TAG, "V3.0 chunk#${record.chunkIndex} @%.1fs: ${record.detections.size} det".format(record.startTimeSec))
                                recordsMutex.withLock { v30Records.add(record) }
                                pendingTimelineRebuild = true
                            },
                        )
                    }
                } else null

                v24Job.join()
                v30Job?.join()
                spectrogramChannel.close()
                specJob.join()

                stopElapsedTimer()
                rebuildTimeline()
                _coreState.update { it.copy(phase = FileAnalysisPhase.DONE) }
                _progressState.update { it.copy(progress = 1f) }
                Log.d(TAG, "File analysis done: ${_timelineState.value.timelineBirds.size} timeline segments")
            } catch (e: Exception) {
                stopElapsedTimer()
                if (e is kotlinx.coroutines.CancellationException) {
                    if (_timelineState.value.timelineBirds.isNotEmpty()) {
                        _coreState.update { it.copy(phase = FileAnalysisPhase.DONE) }
                    } else {
                        _coreState.update { it.copy(phase = FileAnalysisPhase.IDLE) }
                    }
                } else {
                    Log.e(TAG, "File analysis failed", e)
                    _coreState.update {
                        it.copy(
                            phase = FileAnalysisPhase.ERROR,
                            errorMessage = e.message ?: "Unknown error",
                        )
                    }
                }
            } finally {
                withContext(NonCancellable) {
                    v24Classifier.close()
                    v30Classifier?.close()
                }
            }
        }
    }

    fun pauseAnalysis() {
        _isPaused.value = true
        _coreState.update { it.copy(phase = FileAnalysisPhase.PAUSED) }
    }

    fun resumeAnalysis() {
        _isPaused.value = false
        _coreState.update { it.copy(phase = FileAnalysisPhase.ANALYZING) }
    }

    fun stopAnalysis() {
        _isPaused.value = false
        stopElapsedTimer()
        analysisJob?.cancel()
    }

    // ── Playback ──

    fun togglePlayback() {
        val uri = currentUri ?: return
        when (playbackManager.state.value) {
            PlaybackState.IDLE -> playbackManager.play(uri)
            PlaybackState.PLAYING -> playbackManager.pause()
            PlaybackState.PAUSED -> playbackManager.resume()
        }
    }

    fun seekPlayback(fraction: Float) {
        playbackManager.seekToFraction(fraction.coerceIn(0f, 1f))
    }

    // ── Species highlight ──

    fun highlightSpecies(scientificName: String?) {
        val current = _spectrogramState.value.highlightedSpecies
        val next = if (current == scientificName) null else scientificName
        _spectrogramState.update { it.copy(highlightedSpecies = next) }
        if (next != null) {
            val bird = _timelineState.value.timelineBirds.firstOrNull { it.scientificName == next }
            if (bird != null && _coreState.value.fileDurationSec > 0f) {
                val fraction = bird.startTimeSec / _coreState.value.fileDurationSec
                _playbackUiState.update { it.copy(position = fraction.coerceIn(0f, 1f)) }
            }
        }
    }

    // ── Save / Discard ──

    fun saveAnalysis() {
        viewModelScope.launch {
            saveToHistory()
            resetFile()
        }
    }

    fun discardAnalysis() {
        playbackManager.release()
        resetFile()
    }

    // ── Load from history ──

    fun loadFromHistory(analysisId: String) {
        viewModelScope.launch {
            try {
                val analysis = fileAnalysisRepository.getAnalysisById(analysisId) ?: return@launch
                val detections = fileAnalysisRepository.getDetectionsForAnalysis(analysisId)

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

                val markers = buildBirdMarkers(birds, analysis.durationSec)

                if (analysis.fileUri.isNotEmpty()) {
                    try { currentUri = Uri.parse(analysis.fileUri) } catch (_: Exception) {}
                }

                _coreState.update {
                    FileAnalysisCoreState(
                        phase = FileAnalysisPhase.DONE,
                        fileName = analysis.fileName,
                        fileDurationSec = analysis.durationSec,
                        fileSizeLabel = Formatter.formatFileSize(context, analysis.fileSizeBytes),
                        fileDurationLabel = formatMmSs(analysis.durationSec),
                        v30Available = analysis.v30Available,
                        geoLabel = analysis.regionLabel ?: "\u2014",
                        geoConfigured = it.geoConfigured,
                    )
                }
                _progressState.update { it.copy(progress = 1f) }
                _spectrogramState.value = SpectrogramUiState(birdMarkers = markers)
                _timelineState.value = TimelineUiState(
                    timelineBirds = birds,
                    speciesSummaries = summaries,
                )
            } catch (e: Exception) {
                Log.e(TAG, "loadFromHistory failed", e)
            }
        }
    }

    // ── Elapsed timer ──

    private fun startElapsedTimer() {
        stopElapsedTimer()
        elapsedJob = viewModelScope.launch {
            while (true) {
                delay(1000L)
                if (_coreState.value.phase == FileAnalysisPhase.ANALYZING) {
                    _progressState.update { it.copy(elapsedSec = it.elapsedSec + 1) }
                }
            }
        }
    }

    private fun stopElapsedTimer() {
        elapsedJob?.cancel()
        elapsedJob = null
    }

    // ── Progress ──

    private fun updateProgress() {
        val core = _coreState.value
        val prog = _progressState.value
        val fileDuration = core.fileDurationSec
        if (fileDuration <= 0f) return

        val v24EndSec = prog.v24Progress.lastProcessedTimeSec + v24ChunkDuration
        val v30EndSec = if (core.v30Available && v30ChunkDuration > 0f) {
            prog.v30Progress.lastProcessedTimeSec + v30ChunkDuration
        } else fileDuration

        val processedSec = min(v24EndSec, v30EndSec)
        val progress = (processedSec / fileDuration).coerceIn(0f, 1f)
        _progressState.update { it.copy(progress = progress) }
    }

    // ── Timeline ──

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

        val allSegments = TimelineBuilder.build(v24Partial, v30Partial)

        val segments = allSegments.filter { seg ->
            val v24 = seg.v24Confidence
            val v30 = seg.v30Confidence
            (v24 != null && v24 >= HIGH_CONFIDENCE) ||
                (v30 != null && v30 >= HIGH_CONFIDENCE) ||
                (v24 != null && v24 >= MIN_CONFIDENCE && v30 != null && v30 >= MIN_CONFIDENCE)
        }

        Log.d(TAG, "Timeline: ${allSegments.size} raw -> ${segments.size} pass filter")

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
        val markers = buildBirdMarkers(birds, _coreState.value.fileDurationSec)

        _timelineState.value = TimelineUiState(
            timelineBirds = birds,
            speciesSummaries = summaries,
        )
        _spectrogramState.update { it.copy(birdMarkers = markers) }
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

    private fun buildBirdMarkers(birds: List<FileTimelineBirdUi>, durationSec: Float): List<BirdMarker> {
        if (durationSec <= 0f) return emptyList()
        return birds.map { bird ->
            val bestConf = maxOf(
                bird.v24Confidence ?: 0,
                bird.v30Confidence ?: 0,
            ) / 100f
            BirdMarker(
                scientificName = bird.scientificName,
                position = (bird.startTimeSec / durationSec).coerceIn(0f, 1f),
                confidence = bestConf,
            )
        }
    }

    // ── Save to Room ──

    private suspend fun saveToHistory() {
        try {
            val core = _coreState.value
            val timeline = _timelineState.value
            val uri = currentUri ?: return
            val analysisId = UUID.randomUUID().toString()

            val geoLabel = core.geoLabel.takeIf { it != "\u2014" }
            val regionCode = geoRepository.regionCode.first()
                ?: geoRepository.countryCode.first().takeIf { it.isNotEmpty() }

            val waveformAmplitudes = waveformBuilder?.build()
            val waveformBytes = waveformAmplitudes?.let {
                WaveformData(it, core.fileDurationSec, currentFileSize).toByteArray()
            }

            val analysisDuration = System.currentTimeMillis() - analysisStartTimeMs

            val entity = FileAnalysisEntity(
                id = analysisId,
                fileName = core.fileName,
                fileUri = uri.toString(),
                durationSec = core.fileDurationSec,
                fileSizeBytes = currentFileSize,
                regionCode = regionCode,
                regionLabel = geoLabel,
                v30Available = core.v30Available,
                waveformData = waveformBytes,
                createdAt = System.currentTimeMillis(),
                speciesCount = timeline.speciesSummaries.size,
                analysisDurationMs = analysisDuration,
            )

            val detections = timeline.timelineBirds.map { bird ->
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

    // ── Helpers ──

    private fun formatTimeRange(startSec: Float, endSec: Float): String {
        return "${formatMmSs(startSec)} \u2013 ${formatMmSs(endSec)}"
    }

    private fun formatMmSs(totalSec: Float): String {
        val sec = totalSec.toInt()
        return "%d:%02d".format(sec / 60, sec % 60)
    }

    override fun onCleared() {
        super.onCleared()
        playbackManager.release()
    }

    companion object {
        private const val TAG = "FileAnalysisVM"
        private const val HIGH_CONFIDENCE = 0.5f
        private const val MIN_CONFIDENCE = 0.4f
        private const val SPECTROGRAM_SNAPSHOT_INTERVAL = 5
        private const val TIMELINE_REBUILD_INTERVAL = 5
        private const val NUM_WORKERS = 2
    }
}
