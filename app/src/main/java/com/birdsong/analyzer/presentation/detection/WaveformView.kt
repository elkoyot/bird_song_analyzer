package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
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
    val bgColor = Color(0xFF0A0A0A)
    val waveformFill = Color(0xFF2196F3)
    val waveformContour = Color(0xFF64B5F6)
    val markerFallbackColor = Color(0x50FFCC00)
    val labelStyle = TextStyle(color = Color.White.copy(alpha = 0.9f), fontSize = 9.sp)
    val progressLabelStyle = TextStyle(
        color = Color.White,
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
    )
    val textMeasurer = rememberTextMeasurer()

    val animatedProgress by animateFloatAsState(
        targetValue = progress.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 300, easing = LinearEasing),
        label = "waveform_progress",
    )

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(120.dp),
    ) {
        if (amplitudes.isEmpty() || durationSec <= 0f) return@Canvas

        val barCount = amplitudes.size
        val barWidth = size.width / barCount
        val centerY = size.height / 2f
        val halfHeight = size.height * 0.44f
        val progressX = animatedProgress * size.width

        // Build waveform paths once
        val fillPath = buildMirroredFill(amplitudes, barCount, barWidth, centerY, halfHeight)
        val topContour = buildContour(amplitudes, barCount, barWidth, centerY, halfHeight, top = true)
        val bottomContour = buildContour(amplitudes, barCount, barWidth, centerY, halfHeight, top = false)

        // --- Layer 1: Black background (full width) ---
        drawRect(color = bgColor, topLeft = Offset.Zero, size = size)

        // --- Layer 2: Blue waveform clipped to progress ---
        clipRect(right = progressX) {
            drawPath(fillPath, waveformFill)
            drawPath(topContour, waveformContour, style = Stroke(width = 1.5f))
            drawPath(bottomContour, waveformContour, style = Stroke(width = 1.5f))
        }

        // --- Layer 3: Progress label ---
        if (animatedProgress in 0.001f..0.999f && progressLabel.isNotEmpty()) {
            val labelResult = textMeasurer.measure(progressLabel, progressLabelStyle)
            val labelX = (size.width - labelResult.size.width) / 2
            val labelY = (size.height - labelResult.size.height) / 2
            drawText(labelResult, topLeft = Offset(labelX, labelY))
        }

        // --- Layer 4: Markers ---
        val markerTop = size.height * 0.08f
        val markerHeight = size.height * 0.84f
        for (marker in markers) {
            val startX = (marker.startSec / durationSec) * size.width
            val endX = (marker.endSec / durationSec) * size.width
            val markerColor = if (marker.color == Color.Unspecified) {
                markerFallbackColor
            } else {
                marker.color.copy(alpha = 0.25f)
            }
            drawRect(
                color = markerColor,
                topLeft = Offset(startX, markerTop),
                size = Size((endX - startX).coerceAtLeast(2f), markerHeight),
            )
            drawMarkerLabel(textMeasurer, marker.label, startX, markerTop - 2.dp.toPx(), labelStyle)
        }
    }
}

private fun buildMirroredFill(
    amplitudes: FloatArray,
    barCount: Int,
    barWidth: Float,
    centerY: Float,
    halfHeight: Float,
): Path = Path().apply {
    moveTo(0f, centerY)
    for (i in 0 until barCount) {
        lineTo(i * barWidth + barWidth / 2, centerY - amplitudes[i].coerceIn(0f, 1f) * halfHeight)
    }
    lineTo(barCount * barWidth, centerY)
    for (i in barCount - 1 downTo 0) {
        lineTo(i * barWidth + barWidth / 2, centerY + amplitudes[i].coerceIn(0f, 1f) * halfHeight)
    }
    close()
}

private fun buildContour(
    amplitudes: FloatArray,
    barCount: Int,
    barWidth: Float,
    centerY: Float,
    halfHeight: Float,
    top: Boolean,
): Path = Path().apply {
    val sign = if (top) -1f else 1f
    moveTo(barWidth / 2, centerY + sign * amplitudes[0].coerceIn(0f, 1f) * halfHeight)
    for (i in 1 until barCount) {
        lineTo(i * barWidth + barWidth / 2, centerY + sign * amplitudes[i].coerceIn(0f, 1f) * halfHeight)
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

@Preview(showBackground = true, name = "Analyzing 40%")
@Composable
private fun WaveformAnalyzingPreview() {
    BirdSongTheme {
        val amplitudes = FloatArray(400) { i ->
            val t = i / 400f
            (kotlin.math.sin(t * 20f).toFloat() * 0.5f + 0.5f) * 0.8f + 0.1f
        }
        WaveformView(
            amplitudes = amplitudes,
            durationSec = 30f,
            progress = 0.4f,
            progressLabel = "40%",
            markers = listOf(WaveformMarker(3f, 6f, "0:03-0:06")),
        )
    }
}

@Preview(showBackground = true, name = "Done 100%")
@Composable
private fun WaveformDonePreview() {
    BirdSongTheme {
        val amplitudes = FloatArray(400) { i ->
            val t = i / 400f
            (kotlin.math.sin(t * 20f).toFloat() * 0.5f + 0.5f) * 0.8f + 0.1f
        }
        WaveformView(
            amplitudes = amplitudes,
            durationSec = 30f,
            progress = 1f,
        )
    }
}
