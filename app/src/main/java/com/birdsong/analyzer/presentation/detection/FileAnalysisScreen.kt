package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileAnalysisScreen(
    uiState: FileAnalysisUiState = FileAnalysisUiState(),
    onSelectFile: () -> Unit = {},
    onStartAnalysis: () -> Unit = {},
    onPause: () -> Unit = {},
    onResume: () -> Unit = {},
    onCancel: () -> Unit = {},
    onSelectSpecies: (String?) -> Unit = {},
    onSpeciesClick: (scientificName: String, commonName: String) -> Unit = { _, _ -> },
    onLoadWaveform: () -> Unit = {},
    onPickLocation: () -> Unit = {},
    onBack: () -> Unit = {},
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.file_analysis_title)) },
            navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.cd_back))
                }
            },
        )

        // File info + geo (compact row)
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (uiState.fileName.isNotEmpty()) {
                Column(modifier = Modifier.weight(1f, fill = false)) {
                    Text(
                        text = uiState.fileName,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    if (uiState.fileDurationSec > 0f) {
                        Text(
                            text = "${formatDuration(uiState.fileDurationSec)} · ${uiState.fileSizeLabel}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Spacer(modifier = Modifier.width(8.dp))
            }

            if (uiState.geoConfigured) {
                AssistChip(
                    onClick = onPickLocation,
                    label = { Text(uiState.geoLabel, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                    leadingIcon = {
                        Icon(Icons.Default.LocationOn, contentDescription = null,
                            modifier = Modifier.height(18.dp))
                    },
                )
            } else {
                AssistChip(
                    onClick = onPickLocation,
                    label = { Text(stringResource(R.string.file_analysis_geo_warning)) },
                    leadingIcon = {
                        Icon(Icons.Default.Warning, contentDescription = null,
                            modifier = Modifier.height(18.dp))
                    },
                    colors = AssistChipDefaults.assistChipColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        labelColor = MaterialTheme.colorScheme.onErrorContainer,
                        leadingIconContentColor = MaterialTheme.colorScheme.onErrorContainer,
                    ),
                )
            }

            // V3.0 unavailable notice
            if (!uiState.v30Available &&
                uiState.state in listOf(FileAnalysisState.ANALYZING, FileAnalysisState.DONE, FileAnalysisState.PAUSED)
            ) {
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.dual_detection_v30_unavailable),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        // Select file button (only in IDLE or initial)
        if (uiState.state == FileAnalysisState.IDLE || uiState.state == FileAnalysisState.ERROR) {
            Button(
                onClick = onSelectFile,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
            ) {
                Icon(Icons.Default.AudioFile, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.file_analysis_select))
            }
        }

        // Load waveform button (history items without preloaded waveform)
        if (uiState.hasWaveformData && uiState.waveformAmplitudes == null) {
            OutlinedButton(
                onClick = onLoadWaveform,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
            ) {
                Text(stringResource(R.string.file_analysis_show_waveform))
            }
        }

        // Waveform
        if (uiState.waveformAmplitudes != null) {
            val markers = if (uiState.selectedSpecies != null) {
                uiState.speciesSummaries
                    .find { it.scientificName == uiState.selectedSpecies }
                    ?.segments?.map { seg ->
                        WaveformMarker(
                            startSec = seg.startSec,
                            endSec = seg.endSec,
                            label = seg.timeRange,
                        )
                    } ?: emptyList()
            } else emptyList()

            WaveformView(
                amplitudes = uiState.waveformAmplitudes,
                durationSec = uiState.fileDurationSec,
                progress = uiState.waveformProgress,
                progressLabel = uiState.progressLabel,
                markers = markers,
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp),
            )

            // Detection count for selected species
            if (uiState.selectedSpecies != null) {
                val selected = uiState.speciesSummaries
                    .find { it.scientificName == uiState.selectedSpecies }
                if (selected != null) {
                    Text(
                        text = stringResource(
                            R.string.file_analysis_detections_badge,
                            selected.detectionCount,
                        ),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(horizontal = 16.dp),
                    )
                }
            }
        }

        // Controls
        when (uiState.state) {
            FileAnalysisState.IDLE -> {
                if (uiState.fileName.isNotEmpty()) {
                    Button(
                        onClick = onStartAnalysis,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                    ) {
                        Text(stringResource(R.string.file_analysis_start))
                    }
                }
            }
            FileAnalysisState.ANALYZING, FileAnalysisState.PAUSED -> {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (uiState.state == FileAnalysisState.PAUSED) {
                        Button(
                            onClick = onResume,
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(stringResource(R.string.btn_resume))
                        }
                    } else {
                        OutlinedButton(
                            onClick = onPause,
                            modifier = Modifier.weight(1f),
                        ) {
                            Text(stringResource(R.string.btn_pause))
                        }
                    }
                    OutlinedButton(
                        onClick = onCancel,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = MaterialTheme.colorScheme.error,
                        ),
                    ) {
                        Text(stringResource(R.string.btn_cancel))
                    }
                }
            }
            else -> {}
        }

        // Error
        if (uiState.state == FileAnalysisState.ERROR) {
            Text(
                text = stringResource(R.string.file_analysis_error, uiState.errorMessage),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // Species summary header
        if (uiState.speciesSummaries.isNotEmpty()) {
            Text(
                text = stringResource(R.string.file_analysis_species_count, uiState.speciesSummaries.size),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // Species summary list
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (uiState.speciesSummaries.isEmpty() && uiState.state == FileAnalysisState.IDLE
                && uiState.fileName.isEmpty()
            ) {
                item {
                    Text(
                        text = stringResource(R.string.file_analysis_idle),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 32.dp, horizontal = 4.dp),
                    )
                }
            }
            if (uiState.speciesSummaries.isEmpty() && uiState.state == FileAnalysisState.DONE) {
                item {
                    Text(
                        text = stringResource(R.string.detection_no_results),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 32.dp, horizontal = 4.dp),
                    )
                }
            }
            items(uiState.speciesSummaries, key = { it.scientificName }) { summary ->
                SpeciesSummaryCard(
                    summary = summary,
                    isSelected = summary.scientificName == uiState.selectedSpecies,
                    onTap = { onSelectSpecies(summary.scientificName) },
                    onDetailClick = { onSpeciesClick(summary.scientificName, summary.commonName) },
                )
            }
        }
    }
}

@Composable
private fun SpeciesSummaryCard(
    summary: FileSpeciesSummary,
    isSelected: Boolean,
    onTap: () -> Unit,
    onDetailClick: () -> Unit,
) {
    val containerColor = if (isSelected) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.surface
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onTap),
        elevation = CardDefaults.cardElevation(defaultElevation = if (isSelected) 2.dp else 1.dp),
        colors = CardDefaults.cardColors(containerColor = containerColor),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = summary.commonName,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                ConfidenceLabels(summary.maxV24Confidence, summary.maxV30Confidence)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = summary.scientificName,
                    style = MaterialTheme.typography.labelSmall,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                IconButton(onClick = onDetailClick, modifier = Modifier.height(32.dp).width(32.dp)) {
                    Icon(
                        Icons.Default.ChevronRight,
                        contentDescription = stringResource(R.string.cd_details),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

private fun formatDuration(totalSec: Float): String {
    val sec = totalSec.toInt()
    return "%d:%02d".format(sec / 60, sec % 60)
}

// --- Previews ---

private val previewSummaries = listOf(
    FileSpeciesSummary(
        scientificName = "Parus major",
        commonName = "Great Tit",
        maxV24Confidence = 92,
        maxV30Confidence = 78,
        detectionCount = 3,
        segments = listOf(
            SpeciesSegmentUi(0f, 3f, "0:00 – 0:03"),
            SpeciesSegmentUi(15f, 20f, "0:15 – 0:20"),
            SpeciesSegmentUi(36f, 42f, "0:36 – 0:42"),
        ),
    ),
    FileSpeciesSummary(
        scientificName = "Fringilla coelebs",
        commonName = "Chaffinch",
        maxV24Confidence = 85,
        maxV30Confidence = null,
        detectionCount = 1,
        segments = listOf(SpeciesSegmentUi(21f, 35f, "0:21 – 0:35")),
    ),
)

private val previewWaveform = FloatArray(400) { i ->
    val t = i / 400f
    (kotlin.math.sin(t * 20f).toFloat() * 0.5f + 0.5f) * 0.8f + 0.1f
}

@Preview(showBackground = true, showSystemUi = true, name = "Idle - No File")
@Composable
private fun PreviewIdleNoFile() {
    BirdSongTheme {
        FileAnalysisScreen()
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Idle - File Ready")
@Composable
private fun PreviewIdleReady() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.IDLE,
                fileName = "recording_2026-03-04.ogg",
                fileDurationSec = 45f,
                fileSizeLabel = "320 KB",
                waveformAmplitudes = previewWaveform,
                geoLabel = "Belarus · Minsk",
                geoConfigured = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing")
@Composable
private fun PreviewAnalyzing() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.ANALYZING,
                fileName = "recording_2026-03-04.ogg",
                fileDurationSec = 45f,
                fileSizeLabel = "320 KB",
                waveformAmplitudes = previewWaveform,
                waveformProgress = 0.4f,
                v24Progress = ModelProgress(12, 30),
                v30Progress = ModelProgress(5, 15),
                v30Available = true,
                speciesSummaries = previewSummaries.take(1),
                geoLabel = "Belarus · Minsk",
                geoConfigured = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Done")
@Composable
private fun PreviewDone() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.DONE,
                fileName = "recording_2026-03-04.ogg",
                fileDurationSec = 45f,
                fileSizeLabel = "320 KB",
                waveformAmplitudes = previewWaveform,
                waveformProgress = 1f,
                v30Available = true,
                speciesSummaries = previewSummaries,
                selectedSpecies = "Parus major",
                geoLabel = "Belarus · Minsk",
                geoConfigured = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "No Geo")
@Composable
private fun PreviewNoGeo() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.IDLE,
                geoConfigured = false,
            ),
        )
    }
}
