package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
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
import kotlinx.coroutines.launch
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

private val confBarEasing = CubicBezierEasing(0.25f, 0.8f, 0.25f, 1f)

@Composable
fun ConfBar(value: Float, modifier: Modifier = Modifier) {
    val targetFill = value.coerceIn(0.01f, 1f)
    // Animate width: prototype transition "width 1.2s cubic-bezier(0.25,0.8,0.25,1)"
    val animatedFill by animateFloatAsState(
        targetValue = targetFill,
        animationSpec = tween(durationMillis = 1200, easing = confBarEasing),
        label = "confBarFill",
    )
    val gradColors = listOf(Color(0xFFE05050), Color(0xFFE8C020), Color(0xFF3DBA7E))
    val gradientScale = maxOf(2f, 1f / animatedFill)
    Box(modifier = modifier.height(3.dp).clip(RoundedCornerShape(3.dp))) {
        Box(modifier = Modifier.matchParentSize().background(
            Brush.horizontalGradient(gradColors.map { it.copy(alpha = 0.12f) })
        ))
        Canvas(modifier = Modifier.fillMaxHeight().fillMaxWidth(animatedFill)) {
            val totalW = size.width * gradientScale
            drawRect(
                brush = Brush.horizontalGradient(
                    gradColors,
                    startX = 0f,
                    endX = totalW,
                ),
                size = size,
            )
        }
    }
}

// ── AuroraEffect ─────────────────────────────────────────────────────────────
// Per-pixel curtain aurora matching prototype: 9 bands, dual edges,
// noise-based column heights, color cycling via green palette.

private class AuroraBand(
    val xFrac: Float,
    var phase: Float,
    val speed: Float,
    var colorPhase: Float,
    val colorSpeed: Float,
)

@Composable
fun AuroraEffect(modifier: Modifier = Modifier) {
    val bands = remember {
        Array(9) { i ->
            AuroraBand(
                xFrac = i / 8f,
                phase = (Math.random() * 2.0 * PI).toFloat(),
                speed = 0.004f + (Math.random() * 0.006).toFloat(),
                colorPhase = (Math.random() * 2.0 * PI).toFloat(),
                colorSpeed = 0.003f + (Math.random() * 0.004).toFloat(),
            )
        }
    }

    var tick by remember { mutableStateOf(0f) }

    LaunchedEffect(Unit) {
        while (true) {
            delay(33L)
            tick += 0.016f
            bands.forEach { b ->
                b.phase += b.speed
                b.colorPhase += b.colorSpeed
            }
        }
    }

    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        if (w <= 0f || h <= 0f) return@Canvas
        val colW = 2.dp.toPx()

        fun noise(x: Float, t: Float): Float =
            sin(x + t) * 0.5f + sin(x * 2.3f + t * 1.7f) * 0.3f + sin(x * 0.7f + t * 0.5f) * 0.2f

        fun greenPalette(cp: Float): Triple<Int, Int, Int> {
            val gr = maxOf(0f, sin(cp) * 130f + 190f).toInt().coerceIn(0, 255)
            val bl = maxOf(0f, sin(cp + 1.0f) * 110f + 140f).toInt().coerceIn(0, 255)
            val rd = maxOf(0f, sin(cp + 2.8f) * 15f + 8f).toInt().coerceIn(0, 255)
            return Triple(rd, gr, bl)
        }

        // Two edges: top→down (55% height), bottom→up (45% height)
        for (edgeIdx in 0..1) {
            val y0 = if (edgeIdx == 0) 0f else h
            val maxH = if (edgeIdx == 0) h * 0.55f else h * 0.45f
            val flip = edgeIdx == 1

            for (b in bands.indices.reversed()) {
                val band = bands[b]
                val driftX = band.xFrac * w + sin(band.phase * 0.4f) * w * 0.06f
                val bw = w * 0.18f + sin(band.phase * 0.7f + b) * w * 0.06f
                val curtainH = maxH * (0.35f + 0.5f * (0.5f + 0.5f * sin(band.phase + b * 0.9f)))
                val steps = (bw / colW).toInt().coerceAtLeast(1)

                for (s in 0 until steps) {
                    val fx = s.toFloat() / steps
                    val x = driftX - bw / 2f + fx * bw
                    if (x < 0f || x > w) continue

                    val colNoise = noise(fx * PI.toFloat() * 4f, band.phase * 1.2f + b)
                    val colH = curtainH * (0.6f + 0.4f * colNoise)
                    if (colH < 1f) continue

                    val cp = band.colorPhase + fx * 1.2f + b * 0.4f
                    val (rd, gr, bl) = greenPalette(cp)
                    val edgeFade = sin(fx * PI.toFloat())
                    val flicker = 0.55f + 0.35f * sin(band.phase * 3.1f + fx * 7f + b)
                    val alpha = (edgeFade * flicker * 0.72f).coerceIn(0f, 1f)
                    if (alpha < 0.01f) continue

                    val baseColor = Color(rd, gr, bl)
                    val darkColor = Color(
                        (rd * 0.5f).toInt().coerceIn(0, 255),
                        (gr * 0.6f).toInt().coerceIn(0, 255),
                        (bl * 0.8f).toInt().coerceIn(0, 255),
                    )

                    // 3 stacked sections per column (approximates per-column gradient)
                    val s1H = colH * 0.35f
                    val s2H = colH * 0.35f
                    val s3H = colH * 0.30f
                    val a1 = alpha
                    val a2 = (alpha * 0.7f).coerceIn(0f, 1f)
                    val a3 = (alpha * 0.25f).coerceIn(0f, 1f)

                    if (flip) {
                        drawRect(color = baseColor.copy(alpha = a1), topLeft = Offset(x, y0 - s1H), size = Size(colW, s1H))
                        drawRect(color = baseColor.copy(alpha = a2), topLeft = Offset(x, y0 - s1H - s2H), size = Size(colW, s2H))
                        drawRect(color = darkColor.copy(alpha = a3), topLeft = Offset(x, y0 - colH), size = Size(colW, s3H))
                    } else {
                        drawRect(color = baseColor.copy(alpha = a1), topLeft = Offset(x, y0), size = Size(colW, s1H))
                        drawRect(color = baseColor.copy(alpha = a2), topLeft = Offset(x, y0 + s1H), size = Size(colW, s2H))
                        drawRect(color = darkColor.copy(alpha = a3), topLeft = Offset(x, y0 + s1H + s2H), size = Size(colW, s3H))
                    }
                }
            }
        }

        // Soft glow along top edge
        val (gr0, gg0, gb0) = greenPalette(tick * 0.3f)
        val glowH = 8.dp.toPx()
        drawRect(
            brush = Brush.verticalGradient(
                colors = listOf(Color(gr0, gg0, gb0).copy(alpha = 0.18f), Color.Transparent),
                startY = 0f,
                endY = glowH,
            ),
            topLeft = Offset.Zero,
            size = Size(w, glowH),
        )

        // Glow border (rgba(0,255,160,0.28))
        drawRoundRect(
            color = Color(0, 255, 160).copy(alpha = 0.28f),
            topLeft = Offset.Zero,
            size = size,
            cornerRadius = CornerRadius(11.dp.toPx()),
            style = Stroke(width = 1.5.dp.toPx()),
        )
    }
}

// ── RadarCanvas ───────────────────────────────────────────────────────────────

private data class RadarBlip(
    val id: Int,
    val angle: Float,
    val r: Float,
    var ripple: Float = 0f,
    var phase: String = "in",     // "in" or "out"
    var opacity: Float = 0f,
)

@Composable
fun RadarCanvas(
    active: Boolean,
    blipSeq: Int,
    modifier: Modifier = Modifier,
) {
    var blips by remember { mutableStateOf(emptyList<RadarBlip>()) }
    var lastSeq by remember { mutableIntStateOf(0) }
    val scope = rememberCoroutineScope()

    // Add new blip when blipSeq increments
    LaunchedEffect(blipSeq) {
        if (blipSeq > lastSeq && blipSeq > 0) {
            val newBlip = RadarBlip(
                id = blipSeq,
                angle = (Math.random() * 2.0 * PI).toFloat(),
                r = 0.38f + Math.random().toFloat() * 0.55f,
            )
            blips = (blips + newBlip).takeLast(15)
            // Schedule fade-out after 2.5s in separate coroutine
            scope.launch {
                delay(2500L)
                blips = blips.map { if (it.id == newBlip.id) it.copy(phase = "out") else it }
            }
        }
        lastSeq = blipSeq
    }

    // Animate blips
    LaunchedEffect(Unit) {
        while (true) {
            delay(33L)
            if (blips.isNotEmpty()) {
                blips = blips.mapNotNull { b ->
                    val newOpacity = when (b.phase) {
                        "in" -> minOf(1f, b.opacity + 0.06f)
                        else -> maxOf(0f, b.opacity - 0.03f)
                    }
                    if (newOpacity <= 0f && b.phase == "out") null
                    else b.copy(
                        opacity = newOpacity,
                        ripple = if (b.phase == "in") minOf(1f, b.ripple + 0.025f) else b.ripple,
                    )
                }
            }
        }
    }

    val accent = HubColors.Accent

    Canvas(modifier = modifier) {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val maxR = minOf(cx, cy) - 4.dp.toPx()

        // Radial background gradient — only when active (8.5)
        if (active) {
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(accent.copy(alpha = 0.05f), Color.Transparent),
                    center = Offset(cx, cy),
                    radius = maxR,
                ),
                radius = maxR,
                center = Offset(cx, cy),
            )
        }

        // Blips
        blips.forEach { b ->
            val op = b.opacity
            if (op <= 0f) return@forEach
            val px = cx + cos(b.angle) * b.r * maxR
            val py = cy + sin(b.angle) * b.r * maxR

            // Expanding ripple ring (during "in" phase)
            if (b.ripple < 1f && b.phase == "in") {
                val rr = 4.dp.toPx() + b.ripple * 24.dp.toPx()
                val ra = maxOf(0f, (1f - b.ripple) * 0.7f * op)
                drawCircle(
                    color = accent.copy(alpha = ra),
                    radius = rr,
                    center = Offset(px, py),
                    style = Stroke(width = 1.5.dp.toPx()),
                )
            }

            // Dynamic halo glow (8.4) — size varies with opacity
            val haloR = (13f + op * 7f).dp.toPx()
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(accent.copy(alpha = 0.3f * op), Color.Transparent),
                    center = Offset(px, py),
                    radius = haloR,
                ),
                radius = haloR,
                center = Offset(px, py),
            )

            // Dot with 3-stop radial gradient (8.3)
            val dotR = (3.8f + op * 1.2f).dp.toPx()
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color(0xFFFFE478).copy(alpha = 0.95f * op),  // white-gold center
                        accent.copy(alpha = op),                      // accent middle
                        Color(0xFFC8640A).copy(alpha = 0.15f),       // dark orange edge
                    ),
                    center = Offset(px - 0.6.dp.toPx(), py - 0.6.dp.toPx()),
                    radius = dotR,
                ),
                radius = dotR,
                center = Offset(px, py),
            )
        }
    }
}
