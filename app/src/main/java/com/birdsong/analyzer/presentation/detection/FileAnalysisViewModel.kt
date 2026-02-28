package com.birdsong.analyzer.presentation.detection

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.PreferencesRepository
import com.birdsong.analyzer.ml.BirdDetectionPipeline
import com.birdsong.analyzer.ml.ClassifierFactory
import com.birdsong.analyzer.ml.CountryConfig
import com.birdsong.analyzer.ml.CountryConfigLoader
import com.birdsong.analyzer.ml.MetaProfile
import com.birdsong.analyzer.ml.MetaProfileBuilder
import com.birdsong.analyzer.ml.TimelineBuilder
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import javax.inject.Inject
import kotlin.math.roundToInt

enum class FileAnalysisState { IDLE, ANALYZING, DONE, ERROR }

data class ModelProgress(
    val chunksProcessed: Int = 0,
    val totalChunks: Int = 0,
)

data class FileTimelineBirdUi(
    val id: String,
    val commonName: String,
    val scientificName: String,
    val timeRange: String,
    val v24Confidence: Int?,
    val v30Confidence: Int?,
)

data class FileAnalysisUiState(
    val state: FileAnalysisState = FileAnalysisState.IDLE,
    val fileName: String = "",
    val v24Progress: ModelProgress = ModelProgress(),
    val v30Progress: ModelProgress = ModelProgress(),
    val timelineBirds: List<FileTimelineBirdUi> = emptyList(),
    val v30Available: Boolean = false,
    val errorMessage: String = "",
)

@HiltViewModel
class FileAnalysisViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val pipeline: BirdDetectionPipeline,
    private val classifierFactory: ClassifierFactory,
    private val metaProfileBuilder: MetaProfileBuilder,
    private val prefsRepo: PreferencesRepository,
    private val countries: List<CountryConfig>,
) : ViewModel() {

    private val _uiState = MutableStateFlow(FileAnalysisUiState(
        v30Available = classifierFactory.isBirdNetV30Available(),
    ))
    val uiState: StateFlow<FileAnalysisUiState> = _uiState.asStateFlow()

    private var metaProfileJob: Job? = null
    private var analysisJob: Job? = null

    @Volatile
    private var cachedMetaProfile: MetaProfile? = null

    private val v24Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val v30Records = mutableListOf<BirdDetectionPipeline.ChunkDetectionRecord>()
    private val recordsMutex = Mutex()

    private var v24ChunkDuration = 0f
    private var v30ChunkDuration = 0f

    init {
        buildMetaProfileAsync()
    }

    private fun buildMetaProfileAsync() {
        metaProfileJob = viewModelScope.launch {
            try {
                val code = prefsRepo.countryCode.first()
                val region = prefsRepo.regionCode.first()
                val config = CountryConfigLoader.findByCode(countries, code, region) ?: return@launch
                Log.d(TAG, "Building MetaProfile for ${config.nameEn} (${config.code})")
                cachedMetaProfile = metaProfileBuilder.build(config.bbox, config.bufferDeg)
                Log.d(TAG, "MetaProfile ready")
            } catch (e: Exception) {
                Log.e(TAG, "MetaProfile build failed", e)
            }
        }
    }

    fun analyzeFile(uri: Uri, fileName: String) {
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch {
            v24Records.clear()
            v30Records.clear()
            v24ChunkDuration = 0f
            v30ChunkDuration = 0f

            val v30Available = classifierFactory.isBirdNetV30Available()
            _uiState.update {
                FileAnalysisUiState(
                    state = FileAnalysisState.ANALYZING,
                    fileName = fileName,
                    v30Available = v30Available,
                )
            }

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

                val v24Processor = classifierFactory.createProcessor(v24Classifier)
                val v30Processor = v30Classifier?.let { classifierFactory.createProcessor(it) }

                val v24Job = launch {
                    pipeline.analyzeFileDetailed(
                        context = context,
                        uri = uri,
                        classifier = v24Classifier,
                        processor = v24Processor,
                        classifierFactory = classifierFactory,
                        numWorkers = 1,
                        onProgress = { progress ->
                            _uiState.update {
                                it.copy(v24Progress = ModelProgress(
                                    chunksProcessed = progress.processedChunks,
                                    totalChunks = progress.totalChunks,
                                ))
                            }
                        },
                        onChunkResult = { record ->
                            recordsMutex.withLock { v24Records.add(record) }
                            rebuildTimeline()
                        },
                    )
                }

                val v30Job = if (v30Classifier != null && v30Processor != null) {
                    launch {
                        pipeline.analyzeFileDetailed(
                            context = context,
                            uri = uri,
                            classifier = v30Classifier,
                            processor = v30Processor,
                            classifierFactory = classifierFactory,
                            numWorkers = 1,
                            onProgress = { progress ->
                                _uiState.update {
                                    it.copy(v30Progress = ModelProgress(
                                        chunksProcessed = progress.processedChunks,
                                        totalChunks = progress.totalChunks,
                                    ))
                                }
                            },
                            onChunkResult = { record ->
                                recordsMutex.withLock { v30Records.add(record) }
                                rebuildTimeline()
                            },
                        )
                    }
                } else null

                v24Job.join()
                v30Job?.join()

                rebuildTimeline()
                _uiState.update { it.copy(state = FileAnalysisState.DONE) }
                Log.d(TAG, "File analysis done: ${_uiState.value.timelineBirds.size} timeline segments")
            } catch (e: Exception) {
                Log.e(TAG, "File analysis failed", e)
                _uiState.update {
                    it.copy(
                        state = FileAnalysisState.ERROR,
                        errorMessage = e.message ?: "Unknown error",
                    )
                }
            } finally {
                v24Classifier.close()
                v30Classifier?.close()
            }
        }
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
                // Show if at least one model is highly confident
                (v24 != null && v24 >= HIGH_CONFIDENCE) ||
                    (v30 != null && v30 >= HIGH_CONFIDENCE) ||
                    // Or both models detected above minimum threshold
                    (v24 != null && v24 >= MIN_CONFIDENCE && v30 != null && v30 >= MIN_CONFIDENCE)
            }
        val birds = segments.mapIndexed { idx, seg ->
            FileTimelineBirdUi(
                id = "${seg.scientificName}_$idx",
                commonName = seg.commonName,
                scientificName = seg.scientificName,
                timeRange = formatTimeRange(seg.startTimeSec, seg.endTimeSec),
                v24Confidence = seg.v24Confidence?.let { (it * 100).roundToInt() },
                v30Confidence = seg.v30Confidence?.let { (it * 100).roundToInt() },
            )
        }

        _uiState.update { it.copy(timelineBirds = birds) }
    }

    private fun formatTimeRange(startSec: Float, endSec: Float): String {
        return "${formatMmSs(startSec)} – ${formatMmSs(endSec)}"
    }

    private fun formatMmSs(totalSec: Float): String {
        val sec = totalSec.toInt()
        return "%d:%02d".format(sec / 60, sec % 60)
    }

    companion object {
        private const val TAG = "FileAnalysisVM"
        private const val HIGH_CONFIDENCE = 0.8f
        private const val MIN_CONFIDENCE = 0.4f
    }
}
