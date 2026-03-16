package com.birdsong.analyzer.presentation.history

import android.text.format.Formatter
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.data.local.FileAnalysisSummary
import com.birdsong.analyzer.presentation.theme.HubColors
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen(
    analyses: List<FileAnalysisSummary>,
    onAnalysisClick: (String) -> Unit,
    onDelete: (String) -> Unit,
    onBack: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(HubColors.Bg),
    ) {
        // Header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(HubColors.BgEl)
                    .border(1.dp, HubColors.Border, RoundedCornerShape(12.dp))
                    .clickable(onClick = onBack)
                    .padding(horizontal = 14.dp, vertical = 8.dp),
            ) {
                Text(
                    "\u2190 \u041D\u0430\u0437\u0430\u0434",
                    color = HubColors.TextSecondary,
                    fontSize = 13.sp,
                )
            }
            Column {
                Text(
                    "\u0418\u0441\u0442\u043E\u0440\u0438\u044F \u0430\u043D\u0430\u043B\u0438\u0437\u043E\u0432",
                    color = HubColors.TextPrimary,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "${analyses.size} \u0437\u0430\u043F\u0438\u0441\u0435\u0439",
                    color = HubColors.TextMuted,
                    fontSize = 11.sp,
                )
            }
        }

        if (analyses.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("\uD83D\uDDC2", fontSize = 48.sp)
                    Spacer(Modifier.height(12.dp))
                    Text(
                        "\u0418\u0441\u0442\u043E\u0440\u0438\u044F \u043F\u0443\u0441\u0442\u0430",
                        color = HubColors.TextMuted,
                        fontSize = 15.sp,
                    )
                }
            }
        } else {
            LazyColumn(
                contentPadding = PaddingValues(horizontal = 18.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                items(analyses, key = { it.id }) { item ->
                    HistoryCard(
                        item = item,
                        onClick = { onAnalysisClick(item.id) },
                        onDelete = { onDelete(item.id) },
                    )
                }
            }
        }
    }
}

@Composable
private fun HistoryCard(
    item: FileAnalysisSummary,
    onClick: () -> Unit,
    onDelete: () -> Unit,
) {
    val context = LocalContext.current
    val storageSizeLabel = Formatter.formatFileSize(
        context,
        200L + item.waveformSize + item.detectionCount * 100L,
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(HubColors.BgCard)
            .border(1.dp, HubColors.Border, RoundedCornerShape(18.dp))
            .clickable(onClick = onClick),
    ) {
        // Top gradient line
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .align(Alignment.TopCenter)
                .padding(horizontal = 40.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(Color.Transparent, HubColors.Blue.copy(alpha = 0.27f), Color.Transparent),
                    ),
                ),
        )

        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(42.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(HubColors.Blue.copy(alpha = 0.13f)),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83D\uDCC1", fontSize = 18.sp)
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    item.fileName,
                    color = HubColors.TextPrimary,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "\uD83D\uDCCD ${item.regionLabel ?: "\u2014"} \u00b7 ${formatDate(item.createdAt)}",
                    color = HubColors.TextSecondary,
                    fontSize = 12.sp,
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Tag("${item.speciesCount} \u0432\u0438\u0434\u043E\u0432", HubColors.Green)
                    Tag(formatDuration(item.durationSec), HubColors.TextSecondary)
                    Tag(storageSizeLabel, HubColors.TextSecondary)
                }
            }
            Box(
                modifier = Modifier
                    .size(32.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(HubColors.BgEl)
                    .clickable(onClick = onDelete),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83D\uDDD1", fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun Tag(label: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(color.copy(alpha = 0.09f))
            .padding(horizontal = 8.dp, vertical = 3.dp),
    ) {
        Text(
            label,
            color = color,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.3.sp,
        )
    }
}

private fun formatDuration(totalSec: Float): String {
    val sec = totalSec.toInt()
    return "%d:%02d".format(sec / 60, sec % 60)
}

private fun formatDate(epochMs: Long): String {
    val fmt = SimpleDateFormat("dd MMM, HH:mm", Locale.getDefault())
    return fmt.format(Date(epochMs))
}
