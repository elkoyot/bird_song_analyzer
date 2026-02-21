package com.birdsong.analyzer.presentation.detail

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.PauseCircle
import androidx.compose.material.icons.filled.PlayCircle
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh

data class DetailUiState(
    val commonName: String = "",
    val scientificName: String = "",
    val confidence: Int = 0,
    val detectedAt: String = "",
    val durationSec: String = "",
    val latitude: Double? = null,
    val longitude: Double? = null,
    val isPlaying: Boolean = false,
    val playbackProgress: Float = 0f,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailScreen(
    onBack: () -> Unit = {},
    uiState: DetailUiState = DetailUiState(),
    onPlayPause: () -> Unit = {},
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(uiState.commonName) },
            navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
            },
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
        ) {
            // Species name
            Text(
                text = uiState.commonName,
                style = MaterialTheme.typography.headlineLarge,
            )
            Text(
                text = uiState.scientificName,
                style = MaterialTheme.typography.titleMedium,
                fontStyle = FontStyle.Italic,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = stringResource(R.string.detail_confidence, uiState.confidence),
                style = MaterialTheme.typography.titleLarge,
                color = ConfidenceHigh,
                fontWeight = FontWeight.Bold,
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Audio player
            AudioPlayerCard(
                isPlaying = uiState.isPlaying,
                progress = uiState.playbackProgress,
                duration = uiState.durationSec,
                onPlayPause = onPlayPause,
            )

            Spacer(modifier = Modifier.height(24.dp))

            // Metadata
            Text(
                text = stringResource(R.string.detail_detected_at, uiState.detectedAt),
                style = MaterialTheme.typography.bodyLarge,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = stringResource(R.string.detail_duration, uiState.durationSec),
                style = MaterialTheme.typography.bodyLarge,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = if (uiState.latitude != null && uiState.longitude != null) {
                    stringResource(
                        R.string.detail_location,
                        "%.4f°N, %.4f°E".format(uiState.latitude, uiState.longitude),
                    )
                } else {
                    stringResource(R.string.detail_no_location)
                },
                style = MaterialTheme.typography.bodyLarge,
            )

            Spacer(modifier = Modifier.height(32.dp))

            // Description placeholder
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                ),
            ) {
                Text(
                    text = stringResource(R.string.detail_description_placeholder),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(16.dp),
                )
            }
        }
    }
}

@Composable
private fun AudioPlayerCard(
    isPlaying: Boolean,
    progress: Float,
    duration: String,
    onPlayPause: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            IconButton(onClick = onPlayPause) {
                Icon(
                    imageVector = if (isPlaying) Icons.Default.PauseCircle else Icons.Default.PlayCircle,
                    contentDescription = if (isPlaying) "Pause" else "Play",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.height(48.dp),
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "${duration} sec",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

// --- Previews ---

@Preview(showBackground = true, showSystemUi = true, name = "Detail")
@Composable
private fun PreviewDetail() {
    BirdSongTheme {
        DetailScreen(
            uiState = DetailUiState(
                commonName = "Great Tit",
                scientificName = "Parus major",
                confidence = 92,
                detectedAt = "14:32",
                durationSec = "8.5",
                latitude = 53.9045,
                longitude = 27.5615,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Detail - No GPS, Dark")
@Composable
private fun PreviewDetailDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        DetailScreen(
            uiState = DetailUiState(
                commonName = "Chaffinch",
                scientificName = "Fringilla coelebs",
                confidence = 85,
                detectedAt = "14:30",
                durationSec = "5.2",
                isPlaying = true,
                playbackProgress = 0.4f,
            ),
        )
    }
}
