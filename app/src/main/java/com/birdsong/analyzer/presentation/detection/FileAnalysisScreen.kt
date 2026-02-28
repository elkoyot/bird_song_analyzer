package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
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
    onBack: () -> Unit = {},
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.file_analysis_title)) },
            navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
            },
        )

        // Select file button
        if (uiState.state != FileAnalysisState.ANALYZING) {
            Button(
                onClick = onSelectFile,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            ) {
                Icon(Icons.Default.AudioFile, contentDescription = null)
                Spacer(modifier = Modifier.height(8.dp))
                Text(stringResource(R.string.file_analysis_select))
            }
        }

        // Progress
        if (uiState.state == FileAnalysisState.ANALYZING) {
            val v24 = uiState.v24Progress
            val v30 = uiState.v30Progress
            val totalProcessed = v24.chunksProcessed + v30.chunksProcessed
            val totalChunks = v24.totalChunks + v30.totalChunks

            if (totalChunks > 0) {
                LinearProgressIndicator(
                    progress = { totalProcessed.toFloat() / totalChunks },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                )
            } else {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                )
            }

            val progressText = if (uiState.v30Available) {
                "V2.4: ${v24.chunksProcessed}/${v24.totalChunks} | V3.0: ${v30.chunksProcessed}/${v30.totalChunks}"
            } else {
                "V2.4: ${v24.chunksProcessed}/${v24.totalChunks}"
            }
            Text(
                text = progressText,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        // Error
        if (uiState.state == FileAnalysisState.ERROR) {
            Text(
                text = stringResource(R.string.file_analysis_error, uiState.errorMessage),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
            )
        }

        // File name
        if (uiState.fileName.isNotEmpty()) {
            Text(
                text = uiState.fileName,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // V3.0 unavailable notice (during analysis and after)
        if ((uiState.state == FileAnalysisState.ANALYZING || uiState.state == FileAnalysisState.DONE)
            && !uiState.v30Available
        ) {
            Text(
                text = stringResource(R.string.dual_detection_v30_unavailable),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // Detected birds header
        if (uiState.timelineBirds.isNotEmpty()) {
            Text(
                text = stringResource(R.string.detection_detected_count, uiState.timelineBirds.size),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // Timeline bird list or idle/done empty text
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (uiState.timelineBirds.isEmpty() && uiState.state == FileAnalysisState.IDLE) {
                item {
                    Text(
                        text = stringResource(R.string.file_analysis_idle),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 32.dp, horizontal = 4.dp),
                    )
                }
            }
            if (uiState.timelineBirds.isEmpty() && uiState.state == FileAnalysisState.DONE) {
                item {
                    Text(
                        text = stringResource(R.string.detection_no_results),
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 32.dp, horizontal = 4.dp),
                    )
                }
            }
            items(uiState.timelineBirds, key = { it.id }) { bird ->
                TimelineBirdCard(bird = bird)
            }
        }
    }
}

@Composable
private fun TimelineBirdCard(bird: FileTimelineBirdUi) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
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
                    text = bird.commonName,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                ConfidenceLabels(bird.v24Confidence, bird.v30Confidence)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = bird.scientificName,
                    style = MaterialTheme.typography.labelSmall,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                Text(
                    text = bird.timeRange,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

// --- Previews ---

private val previewTimeline = listOf(
    FileTimelineBirdUi("1", "Great Tit", "Parus major", "0:00 – 0:20", v24Confidence = 92, v30Confidence = 78),
    FileTimelineBirdUi("2", "Chaffinch", "Fringilla coelebs", "0:21 – 0:35", v24Confidence = 85, v30Confidence = null),
    FileTimelineBirdUi("3", "Great Tit", "Parus major", "0:36 – 0:52", v24Confidence = 88, v30Confidence = 71),
    FileTimelineBirdUi("4", "Eurasian Blue Tit", "Cyanistes caeruleus", "0:40 – 0:55", v24Confidence = null, v30Confidence = 65),
)

@Preview(showBackground = true, showSystemUi = true, name = "Idle")
@Composable
private fun PreviewIdle() {
    BirdSongTheme {
        FileAnalysisScreen()
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing - Both models")
@Composable
private fun PreviewAnalyzingBoth() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.ANALYZING,
                fileName = "recording_2026-02-28.ogg",
                v24Progress = ModelProgress(chunksProcessed = 12, totalChunks = 48),
                v30Progress = ModelProgress(chunksProcessed = 5, totalChunks = 24),
                timelineBirds = previewTimeline.take(2),
                v30Available = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing - V2.4 only")
@Composable
private fun PreviewAnalyzingV24Only() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.ANALYZING,
                fileName = "recording_2026-02-28.ogg",
                v24Progress = ModelProgress(chunksProcessed = 12, totalChunks = 48),
                v30Available = false,
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
                fileName = "recording_2026-02-28.ogg",
                timelineBirds = previewTimeline,
                v30Available = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Done - V3.0 Unavailable")
@Composable
private fun PreviewDoneNoV30() {
    BirdSongTheme {
        FileAnalysisScreen(
            uiState = FileAnalysisUiState(
                state = FileAnalysisState.DONE,
                fileName = "recording_2026-02-28.ogg",
                timelineBirds = previewTimeline.map { it.copy(v30Confidence = null) },
                v30Available = false,
            ),
        )
    }
}
