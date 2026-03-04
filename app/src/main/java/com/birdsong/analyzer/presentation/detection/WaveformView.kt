package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.presentation.theme.BirdSongTheme

data class WaveformMarker(
    val startSec: Float,
    val endSec: Float,
    val label: String,
    val color: Color = Color.Unspecified,
)

@Composable
fun WaveformView(
    amplitudes: FloatArray,
    durationSec: Float,
    progress: Float = 0f,
    progressLabel: String = "",
    markers: List<WaveformMarker> = emptyList(),
    modifier: Modifier = Modifier,
) {
    val colorScheme = MaterialTheme.colorScheme
    val barColor = colorScheme.surfaceVariant
    val progressBarColor = colorScheme.primary.copy(alpha = 0.12f)
    val progressTextColor = colorScheme.primary
    val markerFallbackColor = colorScheme.tertiary.copy(alpha = 0.3f)
    val labelStyle = TextStyle(
        color = colorScheme.onSurface,
        fontSize = 9.sp,
    )
    val progressLabelStyle = TextStyle(
        color = progressTextColor,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
    )
    val textMeasurer = rememberTextMeasurer()

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(120.dp),
    ) {
        if (amplitudes.isEmpty() || durationSec <= 0f) return@Canvas

        val barCount = amplitudes.size
        val barWidth = size.width / barCount
        val maxBarHeight = size.height * 0.85f
        val barTop = size.height - maxBarHeight

        val clampedProgress = progress.coerceIn(0f, 1f)
        val progressX = clampedProgress * size.width

        // Green progress background
        if (clampedProgress in 0.001f..0.999f) {
            drawRect(
                color = progressBarColor,
                topLeft = Offset.Zero,
                size = Size(progressX, size.height),
            )
        }

        // Waveform bars
        for (i in 0 until barCount) {
            val x = i * barWidth
            val barH = amplitudes[i].coerceIn(0f, 1f) * maxBarHeight
            val y = size.height - barH
            drawRect(
                color = barColor,
                topLeft = Offset(x, y),
                size = Size(barWidth.coerceAtLeast(1f), barH),
            )
        }

        // Progress label centered
        if (clampedProgress in 0.001f..0.999f && progressLabel.isNotEmpty()) {
            val labelResult = textMeasurer.measure(progressLabel, progressLabelStyle)
            val labelX = (size.width - labelResult.size.width) / 2
            val labelY = (size.height - labelResult.size.height) / 2
            drawText(labelResult, topLeft = Offset(labelX, labelY))
        }

        // Markers
        for (marker in markers) {
            val startX = (marker.startSec / durationSec) * size.width
            val endX = (marker.endSec / durationSec) * size.width
            val markerColor = if (marker.color == Color.Unspecified) {
                markerFallbackColor
            } else {
                marker.color.copy(alpha = 0.3f)
            }
            drawRect(
                color = markerColor,
                topLeft = Offset(startX, barTop),
                size = Size((endX - startX).coerceAtLeast(2f), maxBarHeight),
            )
            drawMarkerLabel(textMeasurer, marker.label, startX, barTop - 2.dp.toPx(), labelStyle)
        }
    }
}

private fun DrawScope.drawMarkerLabel(
    textMeasurer: TextMeasurer,
    text: String,
    x: Float,
    y: Float,
    style: TextStyle,
) {
    val result = textMeasurer.measure(text, style)
    val clampedX = x.coerceIn(0f, (size.width - result.size.width).coerceAtLeast(0f))
    val clampedY = (y - result.size.height).coerceAtLeast(0f)
    drawText(result, topLeft = Offset(clampedX, clampedY))
}

@Preview(showBackground = true)
@Composable
private fun WaveformPreview() {
    BirdSongTheme {
        val amplitudes = FloatArray(400) { i ->
            val t = i / 400f
            (kotlin.math.sin(t * 20f).toFloat() * 0.5f + 0.5f) * 0.8f + 0.1f
        }
        WaveformView(
            amplitudes = amplitudes,
            durationSec = 30f,
            progress = 0.6f,
            progressLabel = "60% · ~12s",
            markers = listOf(
                WaveformMarker(3f, 6f, "0:03–0:06"),
                WaveformMarker(15f, 20f, "0:15–0:20"),
            ),
        )
    }
}
