package com.birdsong.analyzer.presentation.history

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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh

data class HistoryItemUi(
    val id: String,
    val commonName: String,
    val scientificName: String,
    val confidence: Int,
    val date: String,
    val time: String,
    val hasLocation: Boolean,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(
    onObservationClick: (String) -> Unit = {},
    observations: List<HistoryItemUi> = emptyList(),
    onDelete: (String) -> Unit = {},
) {
    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = { Text(stringResource(R.string.history_title)) },
        )

        if (observations.isEmpty()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(
                    text = stringResource(R.string.history_empty),
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            LazyColumn(
                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(observations, key = { it.id }) { item ->
                    HistoryCard(
                        item = item,
                        onClick = { onObservationClick(item.id) },
                        onDelete = { onDelete(item.id) },
                    )
                }
            }
        }
    }
}

@Composable
private fun HistoryCard(
    item: HistoryItemUi,
    onClick: () -> Unit,
    onDelete: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 16.dp, top = 12.dp, bottom = 12.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = item.commonName,
                        style = MaterialTheme.typography.titleMedium,
                    )
                    Text(
                        text = "  ${item.confidence}%",
                        style = MaterialTheme.typography.labelLarge,
                        color = ConfidenceHigh,
                    )
                }
                Text(
                    text = item.scientificName,
                    style = MaterialTheme.typography.bodySmall,
                    fontStyle = FontStyle.Italic,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${item.date} • ${item.time}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            IconButton(onClick = onDelete) {
                Icon(
                    Icons.Default.Delete,
                    contentDescription = "Delete",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

// --- Previews ---

private val previewHistory = listOf(
    HistoryItemUi("1", "Great Tit", "Parus major", 92, "2026-02-19", "14:32", true),
    HistoryItemUi("2", "Chaffinch", "Fringilla coelebs", 85, "2026-02-19", "14:30", true),
    HistoryItemUi("3", "Song Thrush", "Turdus philomelos", 88, "2026-02-18", "09:45", false),
    HistoryItemUi("4", "Eurasian Blackbird", "Turdus merula", 91, "2026-02-17", "07:20", true),
)

@Preview(showBackground = true, showSystemUi = true, name = "History")
@Composable
private fun PreviewHistory() {
    BirdSongTheme {
        HistoryScreen(observations = previewHistory)
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "History - Empty")
@Composable
private fun PreviewHistoryEmpty() {
    BirdSongTheme {
        HistoryScreen()
    }
}
