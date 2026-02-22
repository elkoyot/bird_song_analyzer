package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FiberManualRecord
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh

// --- UI State ---

enum class DetectionState { IDLE, ANALYZING, PAUSED, STOPPED }

data class DetectedBirdUi(
    val id: String,
    val commonName: String,
    val scientificName: String,
    val confidence: Int,
    val detectedAt: String,
    val durationSec: String,
)

data class LiveDetectionUiState(
    val state: DetectionState = DetectionState.IDLE,
    val sessionTimer: String = "00:00:00",
    val hasGps: Boolean = false,
    val audioLevel: Float = 0f,  // RMS 0..1, updated ~10x/sec during recording
    val detectedBirds: List<DetectedBirdUi> = emptyList(),
)

// --- Screen ---

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LiveDetectionScreen(
    onBirdClick: (String) -> Unit = {},
    uiState: LiveDetectionUiState = LiveDetectionUiState(),
    onStart: () -> Unit = {},
    onPause: () -> Unit = {},
    onResume: () -> Unit = {},
    onStop: () -> Unit = {},
    onReset: () -> Unit = {},
    onTestSample: () -> Unit = {},
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.detection_title)) },
            actions = {
                if (uiState.hasGps) {
                    Icon(
                        Icons.Default.LocationOn,
                        contentDescription = "GPS",
                        tint = ConfidenceHigh,
                        modifier = Modifier.padding(end = 12.dp),
                    )
                }
            },
        )

        // Status + Timer
        StatusBar(state = uiState.state, timer = uiState.sessionTimer)

        // Audio level meter — visible during recording
        if (uiState.state == DetectionState.ANALYZING || uiState.state == DetectionState.PAUSED) {
            AudioLevelBar(level = uiState.audioLevel)
        }

        // Controls
        ControlPanel(
            state = uiState.state,
            onStart = onStart,
            onPause = onPause,
            onResume = onResume,
            onStop = onStop,
        )

        // Debug: test with audio file
        if (uiState.state == DetectionState.IDLE || uiState.state == DetectionState.STOPPED) {
            OutlinedButton(
                onClick = onTestSample,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp),
            ) {
                Text("Test Sample")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Detected birds header
        if (uiState.detectedBirds.isNotEmpty()) {
            Text(
                text = stringResource(R.string.detection_detected_count, uiState.detectedBirds.size),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
            )
        }

        // Bird list
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (uiState.detectedBirds.isEmpty()) {
                item {
                    Text(
                        text = when (uiState.state) {
                            DetectionState.IDLE -> stringResource(R.string.detection_idle)
                            DetectionState.ANALYZING -> stringResource(R.string.detection_no_results)
                            else -> stringResource(R.string.detection_no_results)
                        },
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 32.dp),
                    )
                }
            }
            items(uiState.detectedBirds, key = { it.id }) { bird ->
                DetectedBirdCard(bird = bird, onClick = { onBirdClick(bird.id) })
            }
        }

        // Reset button
        if (uiState.detectedBirds.isNotEmpty() && uiState.state == DetectionState.ANALYZING) {
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
private fun StatusBar(state: DetectionState, timer: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (state == DetectionState.ANALYZING) {
            Icon(
                Icons.Default.FiberManualRecord,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.error,
                modifier = Modifier.size(12.dp),
            )
            Spacer(modifier = Modifier.width(8.dp))
        }
        Text(
            text = when (state) {
                DetectionState.IDLE -> stringResource(R.string.detection_idle)
                DetectionState.ANALYZING -> stringResource(R.string.detection_analyzing)
                DetectionState.PAUSED -> stringResource(R.string.detection_paused)
                DetectionState.STOPPED -> stringResource(R.string.detection_stopped)
            },
            style = MaterialTheme.typography.bodyLarge,
        )
        Spacer(modifier = Modifier.weight(1f))
        if (state != DetectionState.IDLE) {
            Text(
                text = timer,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium,
            )
        }
    }
}

@Composable
private fun ControlPanel(
    state: DetectionState,
    onStart: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        when (state) {
            DetectionState.IDLE, DetectionState.STOPPED -> {
                Button(
                    onClick = onStart,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                    ),
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.btn_start))
                }
            }
            DetectionState.ANALYZING -> {
                FilledTonalButton(
                    onClick = onPause,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.Pause, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(stringResource(R.string.btn_pause))
                }
                Button(
                    onClick = onStop,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(Icons.Default.Stop, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(stringResource(R.string.btn_stop))
                }
            }
            DetectionState.PAUSED -> {
                FilledTonalButton(
                    onClick = onResume,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(stringResource(R.string.btn_resume))
                }
                Button(
                    onClick = onStop,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(Icons.Default.Stop, contentDescription = null)
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(stringResource(R.string.btn_stop))
                }
            }
        }
    }
}

@Composable
private fun AudioLevelBar(level: Float) {
    // Map RMS to visual: RMS 0..0.3 → bar 0..1 (log-ish scale for better visibility)
    val dbfs = if (level > 1e-6f) (20 * kotlin.math.log10(level)).coerceAtLeast(-60f) else -60f
    val normalized = ((dbfs + 60f) / 60f).coerceIn(0f, 1f)
    val animatedLevel by animateFloatAsState(targetValue = normalized, label = "level")

    // Color: green → yellow → red
    val barColor = when {
        animatedLevel < 0.4f -> lerp(Color(0xFF4CAF50), Color(0xFFFFC107), animatedLevel / 0.4f)
        else -> lerp(Color(0xFFFFC107), Color(0xFFF44336), ((animatedLevel - 0.4f) / 0.6f).coerceAtMost(1f))
    }

    val shape = RoundedCornerShape(4.dp)

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = "MIC",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(modifier = Modifier.width(8.dp))
        Box(
            modifier = Modifier
                .weight(1f)
                .height(8.dp)
                .clip(shape)
                .background(MaterialTheme.colorScheme.surfaceVariant),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(fraction = animatedLevel)
                    .clip(shape)
                    .background(barColor),
            )
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = if (level > 1e-6f) "${dbfs.toInt()} dB" else "---",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun DetectedBirdCard(bird: DetectedBirdUi, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = bird.commonName,
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    text = bird.scientificName,
                    style = MaterialTheme.typography.bodySmall,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${bird.detectedAt} • ${bird.durationSec}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                text = "${bird.confidence}%",
                style = MaterialTheme.typography.headlineLarge,
                color = ConfidenceHigh,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

// --- Previews ---

private val previewBirds = listOf(
    DetectedBirdUi("1", "Great Tit", "Parus major", 92, "05:32", "05:29 – 05:32"),
    DetectedBirdUi("2", "Chaffinch", "Fringilla coelebs", 85, "04:18", "04:15 – 04:18"),
    DetectedBirdUi("3", "Song Thrush", "Turdus philomelos", 88, "02:45", "02:42 – 02:45"),
)

@Preview(showBackground = true, showSystemUi = true, name = "Idle")
@Composable
private fun PreviewIdle() {
    BirdSongTheme {
        LiveDetectionScreen(uiState = LiveDetectionUiState())
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing")
@Composable
private fun PreviewAnalyzing() {
    BirdSongTheme {
        LiveDetectionScreen(
            uiState = LiveDetectionUiState(
                state = DetectionState.ANALYZING,
                sessionTimer = "00:05:32",
                hasGps = true,
                detectedBirds = previewBirds,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Paused")
@Composable
private fun PreviewPaused() {
    BirdSongTheme {
        LiveDetectionScreen(
            uiState = LiveDetectionUiState(
                state = DetectionState.PAUSED,
                sessionTimer = "00:08:15",
                hasGps = true,
                detectedBirds = previewBirds,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing - Dark")
@Composable
private fun PreviewAnalyzingDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        LiveDetectionScreen(
            uiState = LiveDetectionUiState(
                state = DetectionState.ANALYZING,
                sessionTimer = "00:03:47",
                hasGps = true,
                detectedBirds = previewBirds.take(2),
            ),
        )
    }
}
