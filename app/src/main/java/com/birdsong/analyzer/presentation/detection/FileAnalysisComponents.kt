package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.presentation.theme.HubColors

// ── DropZone ─────────────────────────────────────────────────────────────────

@Composable
fun DropZone(onPick: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .border(
                width = 2.dp,
                color = HubColors.BgEl2,
                shape = RoundedCornerShape(24.dp),
            )
            .background(HubColors.BgCard)
            .clickable(onClick = onPick)
            .padding(vertical = 32.dp, horizontal = 20.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(HubColors.BgEl)
                .border(1.dp, HubColors.Border, RoundedCornerShape(20.dp)),
            contentAlignment = Alignment.Center,
        ) {
            Text("\uD83C\uDFB5", fontSize = 32.sp)
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "\u0412\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0430\u0443\u0434\u0438\u043E\u0444\u0430\u0439\u043B",
                color = HubColors.TextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = "MP3 \u00b7 WAV \u00b7 M4A \u00b7 FLAC \u00b7 OGG",
                color = HubColors.TextMuted,
                fontSize = 12.sp,
            )
        }
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(12.dp))
                .background(HubColors.Accent)
                .padding(horizontal = 24.dp, vertical = 10.dp),
        ) {
            Text(
                text = "\u041E\u0442\u043A\u0440\u044B\u0442\u044C \u0444\u0430\u0439\u043B",
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp,
            )
        }
    }
}

// ── FileCard ─────────────────────────────────────────────────────────────────

@Composable
fun FileCard(
    fileName: String,
    fileSize: String,
    phase: FileAnalysisPhase,
    progress: Float = 0f,
    elapsedSec: Int = 0,
    speciesCount: Int = 0,
    onClose: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    val isDone = phase == FileAnalysisPhase.DONE
    val isAnalyzing = phase == FileAnalysisPhase.ANALYZING
    val isPaused = phase == FileAnalysisPhase.PAUSED
    val isActive = isAnalyzing || isPaused

    val borderColor = when {
        isDone -> HubColors.Green.copy(alpha = 0.2f)
        isAnalyzing -> HubColors.Accent.copy(alpha = 0.27f)
        else -> HubColors.Border
    }
    val bgColor = if (isDone) HubColors.Green.copy(alpha = 0.03f) else HubColors.BgCard

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(18.dp))
            .padding(12.dp),
    ) {
        // File info row
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .clip(RoundedCornerShape(11.dp))
                    .background(HubColors.Blue.copy(alpha = 0.13f)),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83C\uDFB5", fontSize = 20.sp)
            }
            Spacer(Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = fileName,
                    color = HubColors.TextPrimary,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 13.sp,
                    maxLines = 1,
                )
                Text(text = fileSize, color = HubColors.TextMuted, fontSize = 11.sp)
            }
            if (onClose != null && (phase == FileAnalysisPhase.READY || isDone)) {
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(HubColors.BgEl)
                        .clickable(onClick = onClose),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("\u00D7", color = HubColors.TextMuted, fontSize = 14.sp)
                }
            }
        }

        // Progress section
        if (isActive) {
            Spacer(Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(HubColors.Accent),
                    )
                    Spacer(Modifier.width(5.dp))
                    Text(
                        text = if (isAnalyzing) "\u0410\u043D\u0430\u043B\u0438\u0437\u0438\u0440\u0443\u044E..."
                               else "\u23F8 \u041F\u0430\u0443\u0437\u0430",
                        color = HubColors.Accent,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Text(
                    text = "${formatElapsed(elapsedSec)} \u00b7 ${(progress * 100).toInt().coerceAtMost(100)}%",
                    color = HubColors.TextMuted,
                    fontSize = 11.sp,
                )
            }
            Spacer(Modifier.height(6.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(HubColors.BgEl),
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(progress.coerceIn(0f, 1f))
                        .height(3.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(HubColors.Accent),
                )
            }
        }

        // Done header
        if (isDone) {
            Spacer(Modifier.height(10.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = "\u2713 \u0410\u043D\u0430\u043B\u0438\u0437 \u0437\u0430\u0432\u0435\u0440\u0448\u0451\u043D",
                    color = HubColors.Green,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    text = "\u00b7 $speciesCount \u0432\u0438\u0434\u043E\u0432",
                    color = HubColors.TextMuted,
                    fontSize = 11.sp,
                )
            }
        }
    }
}

// ── SpectrogramView ──────────────────────────────────────────────────────────

@Composable
fun SpectrogramView(
    columns: List<FloatArray>,
    markers: List<BirdMarker> = emptyList(),
    highlightedSpecies: String? = null,
    playhead: Float? = null,
    isAnalyzing: Boolean = false,
    onSeek: ((Float) -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    val accent = HubColors.Accent

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(64.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(HubColors.Bg)
            .then(
                if (onSeek != null) {
                    Modifier.pointerInput(Unit) {
                        detectTapGestures { offset ->
                            onSeek(offset.x / size.width)
                        }
                    }
                } else Modifier,
            ),
    ) {
        if (columns.isEmpty()) return@Canvas

        val colCount = columns.size
        val colWidth = size.width / colCount
        val numBins = columns.firstOrNull()?.size ?: 32

        // Draw spectrogram tiles
        for (x in 0 until colCount) {
            val col = columns[x]
            for (y in 0 until col.size) {
                val v = col[y]
                val r = (v * 80f).toInt().coerceIn(0, 255)
                val g = (v * 180f).toInt().coerceIn(0, 255)
                val b = (v * 200f).toInt().coerceIn(0, 255)
                val a = (0.3f + v * 0.7f).coerceIn(0f, 1f)
                drawRect(
                    color = Color(r, g, b, (a * 255).toInt()),
                    topLeft = Offset(
                        x * colWidth,
                        size.height - (y + 1) * (size.height / numBins),
                    ),
                    size = Size(colWidth + 1f, size.height / numBins + 1f),
                )
            }
        }

        // Bird markers
        for (marker in markers) {
            val mx = marker.position * size.width
            val col = when {
                marker.confidence >= 0.75f -> HubColors.Green
                marker.confidence >= 0.35f -> HubColors.Yellow
                else -> HubColors.Red
            }
            val isHighlighted = marker.scientificName == highlightedSpecies
            drawLine(
                color = if (isHighlighted) accent else col,
                start = Offset(mx, 0f),
                end = Offset(mx, size.height),
                strokeWidth = if (isHighlighted) 2.dp.toPx() else 1.5.dp.toPx(),
                pathEffect = if (isHighlighted) null else PathEffect.dashPathEffect(floatArrayOf(3f, 3f)),
            )
        }

        // Playhead
        if (playhead != null) {
            val px = playhead * size.width
            drawLine(
                color = accent.copy(alpha = 0.9f),
                start = Offset(px, 0f),
                end = Offset(px, size.height),
                strokeWidth = 1.5.dp.toPx(),
            )
        }
    }
}

// ── PlaybackControls ─────────────────────────────────────────────────────────

@Composable
fun PlaybackControls(
    isPlaying: Boolean,
    position: Float,
    positionLabel: String,
    durationLabel: String,
    onToggle: () -> Unit,
    onSeek: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(10.dp))
                .background(HubColors.Accent)
                .clickable(onClick = onToggle)
                .padding(horizontal = 16.dp, vertical = 7.dp),
        ) {
            Text(
                text = if (isPlaying) "\u23F8 \u041F\u0430\u0443\u0437\u0430" else "\u25B6 \u0421\u043B\u0443\u0448\u0430\u0442\u044C",
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp,
            )
        }
        Column(modifier = Modifier.weight(1f)) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .clip(RoundedCornerShape(3.dp))
                    .background(HubColors.BgEl),
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(position.coerceIn(0f, 1f))
                        .height(3.dp)
                        .clip(RoundedCornerShape(3.dp))
                        .background(HubColors.Accent),
                )
            }
            Spacer(Modifier.height(3.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(positionLabel, color = HubColors.TextMuted, fontSize = 10.sp)
                Text(durationLabel, color = HubColors.TextMuted, fontSize = 10.sp)
            }
        }
    }
}

// ── FileBirdResultItem ───────────────────────────────────────────────────────

@Composable
fun FileBirdResultItem(
    bird: FileTimelineBirdUi,
    isHighlighted: Boolean,
    isDone: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val best = maxOf(bird.v24Confidence ?: 0, bird.v30Confidence ?: 0) / 100f
    val col = when {
        best >= 0.75f -> HubColors.Green
        best >= 0.35f -> HubColors.Yellow
        else -> HubColors.Red
    }

    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }
    val alpha by animateFloatAsState(if (visible) 1f else 0f, tween(300), label = "birdAlpha")

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(if (isHighlighted) col.copy(alpha = 0.07f) else HubColors.BgCard)
            .border(
                1.5.dp,
                if (isHighlighted) col.copy(alpha = 0.4f) else HubColors.Border,
                RoundedCornerShape(16.dp),
            )
            .clickable(onClick = onClick)
            .padding(11.dp),
        contentAlignment = Alignment.TopStart,
    ) {
        if (isHighlighted) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .align(Alignment.TopCenter)
                    .background(
                        Brush.horizontalGradient(
                            listOf(Color.Transparent, col, Color.Transparent),
                        ),
                    ),
            )
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(11.dp),
        ) {
            // Bird icon placeholder
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(col.copy(alpha = 0.1f))
                    .border(1.dp, col.copy(alpha = 0.2f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83D\uDC26", fontSize = 20.sp)
            }
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        bird.commonName,
                        color = HubColors.TextPrimary,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp,
                    )
                    if (isHighlighted) {
                        Spacer(Modifier.width(6.dp))
                        Box(
                            modifier = Modifier
                                .size(5.dp)
                                .clip(RoundedCornerShape(2.5.dp))
                                .background(col),
                        )
                    }
                }
                Text(
                    bird.scientificName,
                    color = HubColors.TextMuted,
                    fontSize = 11.sp,
                    fontStyle = FontStyle.Italic,
                )
                Spacer(Modifier.height(5.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    ConfBar(best, modifier = Modifier.weight(1f))
                    Text(
                        "${(best * 100).toInt()}%",
                        color = col,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.width(32.dp),
                    )
                }
            }
            if (isDone) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(bird.timeRange, color = HubColors.TextMuted, fontSize = 10.sp)
                    Text("\u203A", color = HubColors.TextMuted, fontSize = 14.sp)
                }
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

private fun formatElapsed(sec: Int): String =
    "%02d:%02d".format(sec / 60, sec % 60)
