package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.presentation.theme.HubColors
import kotlinx.coroutines.delay
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

// ── WaveBars ──────────────────────────────────────────────────────────────────

@Composable
fun WaveBars(
    active: Boolean,
    color: Color = HubColors.Accent,
    count: Int = 40,
    modifier: Modifier = Modifier,
) {
    var heights by remember { mutableStateOf(FloatArray(count) { 0.12f }) }

    LaunchedEffect(active) {
        if (!active) {
            heights = FloatArray(count) { 0.12f }
            return@LaunchedEffect
        }
        while (true) {
            delay(33L)
            val h = heights.copyOf()
            for (i in 0 until count) {
                val v = h[i]
                h[i] = (v + (0.08f + Math.random().toFloat() * 0.88f - v) * 0.25f).coerceIn(0.05f, 1f)
            }
            heights = h
        }
    }

    Canvas(modifier = modifier) {
        val gap = 2.dp.toPx()
        val barW = (size.width - gap * (count - 1)) / count
        heights.forEachIndexed { i, h ->
            val barH = maxOf(3.dp.toPx(), h * size.height)
            drawRoundRect(
                color = color.copy(alpha = 0.4f + h * 0.6f),
                topLeft = Offset(i * (barW + gap), (size.height - barH) / 2f),
                size = Size(barW, barH),
                cornerRadius = CornerRadius(barW / 2f),
            )
        }
    }
}

// ── ConfBar ───────────────────────────────────────────────────────────────────

@Composable
fun ConfBar(value: Float, modifier: Modifier = Modifier) {
    val fill = value.coerceIn(0f, 1f)
    val gradColors = listOf(Color(0xFFE05050), Color(0xFFE8C020), Color(0xFF3DBA7E))
    Box(modifier = modifier.height(4.dp).clip(RoundedCornerShape(3.dp))) {
        Box(modifier = Modifier.matchParentSize().background(
            Brush.horizontalGradient(gradColors.map { it.copy(alpha = 0.12f) })
        ))
        Box(modifier = Modifier.fillMaxHeight().fillMaxWidth(fill).background(
            Brush.horizontalGradient(gradColors)
        ))
    }
}

// ── RadarCanvas ───────────────────────────────────────────────────────────────

private data class RadarBlip(
    val id: String,
    val angle: Float,
    val r: Float,
    val born: Long,
    val ripple: Float = 0f,
)

@Composable
fun RadarCanvas(birds: List<DualDetectedBirdUi>, modifier: Modifier = Modifier) {
    var blips by remember { mutableStateOf(emptyList<RadarBlip>()) }

    LaunchedEffect(birds.size) {
        val ids = birds.map { it.id }.toSet()
        blips = blips.filter { it.id in ids } +
            birds.filter { b -> blips.none { it.id == b.id } }.map { b ->
                RadarBlip(
                    id = b.id,
                    angle = (Math.random() * 2.0 * PI).toFloat(),
                    r = 0.38f + Math.random().toFloat() * 0.48f,
                    born = System.currentTimeMillis(),
                )
            }
    }

    LaunchedEffect(Unit) {
        while (true) {
            delay(33L)
            if (blips.isNotEmpty()) {
                blips = blips.map { it.copy(ripple = minOf(1f, it.ripple + 0.022f)) }
            }
        }
    }

    val green = HubColors.Green
    val accent = HubColors.Accent

    Canvas(modifier = modifier) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val maxR = minOf(cx, cy) - 4.dp.toPx()

        listOf(0.35f, 0.58f, 0.82f, 1.0f).forEachIndexed { i, f ->
            drawCircle(
                color = accent.copy(alpha = if (i == 3) 0.18f else 0.10f),
                radius = maxR * f,
                center = Offset(cx, cy),
                style = Stroke(width = (if (i == 3) 1.5f else 1f).dp.toPx()),
            )
        }

        val now = System.currentTimeMillis()
        blips.forEach { b ->
            val age = now - b.born
            val opacity = if (age < 1800L) 1f else maxOf(0f, 1f - (age - 1800L) / 900f)
            if (opacity <= 0f) return@forEach
            val px = cx + cos(b.angle) * b.r * maxR
            val py = cy + sin(b.angle) * b.r * maxR
            if (b.ripple < 1f) {
                drawCircle(
                    color = green.copy(alpha = maxOf(0f, (0.75f - b.ripple * 0.8f)) * opacity),
                    radius = 5.dp.toPx() + b.ripple * 24.dp.toPx(),
                    center = Offset(px, py),
                    style = Stroke(width = 1.5.dp.toPx()),
                )
            }
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(green.copy(alpha = 0.55f * opacity), Color.Transparent),
                    center = Offset(px, py),
                    radius = 12.dp.toPx(),
                ),
                radius = 12.dp.toPx(),
                center = Offset(px, py),
            )
            drawCircle(
                color = Color(0xFF64E6A0).copy(alpha = 0.95f * opacity),
                radius = 3.5.dp.toPx(),
                center = Offset(px, py),
            )
        }
    }
}
