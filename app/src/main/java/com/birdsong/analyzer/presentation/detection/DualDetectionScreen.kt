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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
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
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh
import com.birdsong.analyzer.presentation.theme.ConfidenceLow
import com.birdsong.analyzer.presentation.theme.ConfidenceMedium

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DualDetectionScreen(
    uiState: DualDetectionUiState = DualDetectionUiState(),
    onStart: () -> Unit = {},
    onPause: () -> Unit = {},
    onResume: () -> Unit = {},
    onStop: () -> Unit = {},
    onReset: () -> Unit = {},
    onBack: (() -> Unit)? = null,
    onSelectFile: (() -> Unit)? = null,
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.dual_detection_title)) },
            navigationIcon = {
                if (onBack != null) {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            },
        )

        // Reuse shared composables from LiveDetectionScreen
        StatusBar(state = uiState.state, timer = uiState.sessionTimer)

        if (uiState.state == DetectionState.ANALYZING || uiState.state == DetectionState.PAUSED) {
            AudioLevelBar(level = uiState.audioLevel)
        }

        ControlPanel(
            state = uiState.state,
            onStart = onStart,
            onPause = onPause,
            onResume = onResume,
            onStop = onStop,
            idleExtra = if (onSelectFile != null) {
                {
                    OutlinedButton(
                        onClick = onSelectFile,
                        modifier = Modifier.weight(1f),
                    ) {
                        Text(stringResource(R.string.file_analysis_select))
                    }
                }
            } else null,
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Display filter:
        // 1. At least one model ≥80% (high confidence anchor)
        // 2. Both models ≥40% (medium agreement)
        // 3. Both models see the species (cross-model confirmation at any confidence)
        val visibleBirds = uiState.birds.filter { bird ->
            val v24 = bird.v24Confidence
            val v30 = bird.v30Confidence
            (v24 != null && v24 >= 80) || (v30 != null && v30 >= 80) ||
                (v24 != null && v30 != null)
        }

        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            if (visibleBirds.isEmpty()) {
                item {
                    EmptyColumnText(uiState.state)
                }
            }
            if (!uiState.v30Available && uiState.state != DetectionState.IDLE) {
                item {
                    Text(
                        text = stringResource(R.string.dual_detection_v30_unavailable),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 4.dp),
                    )
                }
            }
            items(visibleBirds, key = { it.id }) { bird ->
                DualBirdCard(bird = bird)
            }
        }

        // Reset button
        if (visibleBirds.isNotEmpty() && uiState.state == DetectionState.ANALYZING) {
            OutlinedButton(
                onClick = onReset,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.btn_reset))
            }
        }
    }
}

@Composable
private fun EmptyColumnText(state: DetectionState) {
    Text(
        text = when (state) {
            DetectionState.IDLE -> stringResource(R.string.detection_idle)
            DetectionState.PREPARING -> stringResource(R.string.detection_preparing)
            else -> stringResource(R.string.detection_no_results)
        },
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(vertical = 16.dp, horizontal = 4.dp),
    )
}

@Composable
private fun DualBirdCard(bird: DualDetectedBirdUi) {
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
                    text = bird.detectedAt,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
internal fun ConfidenceLabels(v24: Int?, v30: Int?) {
    val style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    val dash = MaterialTheme.colorScheme.onSurfaceVariant
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("v2.4: ", style = style, color = dash)
        Text(
            text = v24?.let { "$it%" } ?: "—",
            style = style,
            color = v24?.let { confidenceColor(it) } ?: dash,
        )
        Text("  |  ", style = style, color = dash)
        Text("v3.0: ", style = style, color = dash)
        Text(
            text = v30?.let { "$it%" } ?: "—",
            style = style,
            color = v30?.let { confidenceColor(it) } ?: dash,
        )
    }
}

internal fun confidenceColor(percent: Int) = when {
    percent >= 80 -> ConfidenceHigh
    percent >= 40 -> ConfidenceMedium
    else -> ConfidenceLow
}

// --- Previews ---

private val previewBirds = listOf(
    DualDetectedBirdUi("b1", "Great Tit", "Parus major", v24Confidence = 92, v30Confidence = 78, detectedAt = "05:32"),
    DualDetectedBirdUi("b2", "Chaffinch", "Fringilla coelebs", v24Confidence = 85, v30Confidence = null, detectedAt = "04:18"),
    DualDetectedBirdUi("b3", "Eurasian Blue Tit", "Cyanistes caeruleus", v24Confidence = null, v30Confidence = 65, detectedAt = "04:20"),
)

@Preview(showBackground = true, showSystemUi = true, name = "Dual - Analyzing")
@Composable
private fun PreviewAnalyzing() {
    BirdSongTheme {
        DualDetectionScreen(
            uiState = DualDetectionUiState(
                state = DetectionState.ANALYZING,
                sessionTimer = "00:05:32",
                hasGps = true,
                birds = previewBirds,
                v30Available = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Dual - V3.0 Unavailable")
@Composable
private fun PreviewV30Unavailable() {
    BirdSongTheme {
        DualDetectionScreen(
            uiState = DualDetectionUiState(
                state = DetectionState.ANALYZING,
                sessionTimer = "00:02:10",
                birds = previewBirds.map { it.copy(v30Confidence = null) },
                v30Available = false,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Dual - Idle")
@Composable
private fun PreviewIdle() {
    BirdSongTheme {
        DualDetectionScreen(
            uiState = DualDetectionUiState(v30Available = true),
        )
    }
}
