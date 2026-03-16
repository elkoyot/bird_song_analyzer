package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.StartOffset
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.draw.scale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size

import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.HubColors
import kotlin.math.roundToInt
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

// ── helpers ───────────────────────────────────────────────────────────────────

internal fun confColor(best: Float) = when {
    best >= 0.75f -> HubColors.Green
    best >= 0.35f -> HubColors.Yellow
    else -> HubColors.Red
}

// Kept for FileAnalysisScreen backward compatibility
internal fun confidenceColor(percent: Int) = when {
    percent >= 75 -> HubColors.Green
    percent >= 35 -> HubColors.Yellow
    else -> HubColors.Red
}

@Composable
internal fun ConfidenceLabels(v24: Int?, v30: Int?) {
    val style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("v2.4: ", style = style, color = muted)
        Text(v24?.let { "$it%" } ?: "\u2014", style = style, color = v24?.let { confidenceColor(it) } ?: muted)
        Text("  |  ", style = style, color = muted)
        Text("v3.0: ", style = style, color = muted)
        Text(v30?.let { "$it%" } ?: "\u2014", style = style, color = v30?.let { confidenceColor(it) } ?: muted)
    }
}

// ── RecordButton ──────────────────────────────────────────────────────────────

@Composable
fun RecordButton(
    state: DetectionState,
    blipSeq: Int,
    sessionTimer: String,
    onPress: () -> Unit,
    onLongPressStop: () -> Unit,
) {
    val isRec = state == DetectionState.ANALYZING
    val isPaused = state == DetectionState.PAUSED
    val isPreparing = state == DetectionState.PREPARING
    val isActive = isRec || isPaused
    val buttonBrush = when {
        isRec -> Brush.radialGradient(
            listOf(Color(0xFFE8504A), Color(0xFFC0392B)),
            center = Offset(36f, 32f),
        )
        isPaused -> Brush.radialGradient(
            listOf(Color(0xFFFFDA6A), Color(0xFFC87800)),
            center = Offset(36f, 32f),
        )
        else -> Brush.radialGradient(
            listOf(Color(0xFFFFD780), HubColors.Accent),
            center = Offset(36f, 32f),
        )
    }
    val waveformIconSize = if (isRec) 19.dp else 22.dp

    val inf = rememberInfiniteTransition()

    val breathScale by inf.animateFloat(
        initialValue = 1f, targetValue = 1.025f,
        animationSpec = infiniteRepeatable(tween(2800, easing = FastOutSlowInEasing), RepeatMode.Reverse),
    )

    val rippleEasing = CubicBezierEasing(0.12f, 0.5f, 0.28f, 1.0f)
    val r0 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(3400, easing = rippleEasing)))
    val r1 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(4300, easing = rippleEasing), initialStartOffset = StartOffset(1000)))
    val r2 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(5200, easing = rippleEasing), initialStartOffset = StartOffset(2000)))

    var holdProgress by remember { mutableFloatStateOf(0f) }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Box(modifier = Modifier.size(170.dp), contentAlignment = Alignment.Center) {
        RadarCanvas(active = isActive, blipSeq = blipSeq, modifier = Modifier.size(190.dp).align(Alignment.Center))
        if (!isRec && !isPaused && !isPreparing) {
            Box(
                modifier = Modifier.size(95.dp).scale(breathScale)
                    .border(1.5.dp, HubColors.Accent.copy(alpha = 0.27f), CircleShape),
            )
            Box(
                modifier = Modifier.size(121.dp).scale(breathScale)
                    .border(1.dp, HubColors.Accent.copy(alpha = 0.12f), CircleShape),
            )
        }
        if (isRec) {
            Box(modifier = Modifier.size(88.dp).scale(0.88f + r0 * 1.42f).alpha(0.5f * (1f - r0)).border(1.1.dp, HubColors.RedHot, CircleShape))
            Box(modifier = Modifier.size(110.dp).scale(0.88f + r1 * 1.42f).alpha(0.5f * (1f - r1)).border(0.9.dp, HubColors.RedHot, CircleShape))
            Box(modifier = Modifier.size(132.dp).scale(0.88f + r2 * 1.42f).alpha(0.5f * (1f - r2)).border(0.7.dp, HubColors.RedHot, CircleShape))
        }
        // Glow behind button
        Canvas(modifier = Modifier.size(110.dp)) {
            val glowColor = when {
                isRec -> HubColors.Red
                isPaused -> HubColors.Accent
                else -> HubColors.Accent
            }
            val glowAlpha = if (isRec) 0.22f else 0.15f
            drawCircle(
                brush = Brush.radialGradient(
                    listOf(glowColor.copy(alpha = glowAlpha), Color.Transparent),
                ),
                radius = size.minDimension / 2f,
            )
        }
        // Long-press hold-progress ring
        if (holdProgress > 0f) {
            Canvas(modifier = Modifier.size(112.dp)) {
                val r = 46.dp.toPx()
                val cx = size.width / 2f
                val cy = size.height / 2f
                // Background track
                drawCircle(
                    color = HubColors.RedHot.copy(alpha = 0.18f),
                    radius = r,
                    center = Offset(cx, cy),
                    style = Stroke(width = 4.5.dp.toPx()),
                )
                // Progress arc
                val sweep = holdProgress * 360f
                drawArc(
                    color = HubColors.RedHot,
                    startAngle = -90f,
                    sweepAngle = sweep,
                    useCenter = false,
                    style = Stroke(width = 4.5.dp.toPx()),
                )
            }
        }
        val btnScale = when {
            holdProgress > 0f -> 0.88f
            isRec -> 0.93f
            else -> 1f
        }
        val animatedScale by animateFloatAsState(btnScale, tween(250), label = "btnScale")
        Box(
            modifier = Modifier
                .size(82.dp)
                .scale(animatedScale)
                .clip(CircleShape)
                .background(buttonBrush)
                .then(
                    if (!isPreparing) {
                        Modifier.pointerInput(isRec, isPaused) {
                            detectTapGestures(
                                onTap = { onPress() },
                                onPress = { _ ->
                                    if (isRec || isPaused) {
                                        coroutineScope {
                                            val job = launch {
                                                val totalMs = 1300L
                                                val stepMs = 16L
                                                val steps = totalMs / stepMs
                                                var step = 0
                                                while (step <= steps) {
                                                    holdProgress = step.toFloat() / steps.toFloat()
                                                    if (holdProgress >= 1f) {
                                                        onLongPressStop()
                                                        break
                                                    }
                                                    delay(stepMs)
                                                    step++
                                                }
                                            }
                                            tryAwaitRelease()
                                            job.cancel()
                                            holdProgress = 0f
                                        }
                                    } else {
                                        tryAwaitRelease()
                                    }
                                },
                            )
                        }
                    } else Modifier,
                ),
            contentAlignment = Alignment.Center,
        ) {
            WaveformIcon(modifier = Modifier.size(waveformIconSize))
        }
        // Timer inside radar area (bottom)
        if (isRec || isPaused) {
            Text(
                text = sessionTimer,
                color = if (holdProgress > 0f) HubColors.RedHot else if (isRec) HubColors.RedHot.copy(alpha = 0.8f) else HubColors.Accent.copy(alpha = 0.8f),
                fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 3.sp,
                modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 2.dp),
            )
        }
    }
    // Status label with blinking dot
    Row(
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth().padding(top = 2.dp),
    ) {
        if (isRec) {
            val blinkInf = rememberInfiniteTransition()
            val blinkA by blinkInf.animateFloat(
                initialValue = 1f, targetValue = 0.1f,
                animationSpec = infiniteRepeatable(tween(1100, easing = FastOutSlowInEasing), RepeatMode.Reverse),
            )
            Box(
                modifier = Modifier.size(5.dp).alpha(blinkA)
                    .clip(CircleShape).background(HubColors.RedHot),
            )
            Spacer(Modifier.width(6.dp))
        }
        Text(
            text = when (state) {
                DetectionState.ANALYZING -> stringResource(R.string.live_listening)
                DetectionState.PAUSED -> stringResource(R.string.live_pause_label)
                DetectionState.PREPARING -> stringResource(R.string.detection_preparing)
                else -> stringResource(R.string.live_start_hint)
            },
            color = when (state) {
                DetectionState.ANALYZING -> HubColors.RedHot
                DetectionState.PAUSED -> HubColors.Yellow
                else -> HubColors.TextSecondary
            },
            fontSize = 11.sp,
            fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
        )
    }
    // Hold hint
    if (isActive) {
        Text(
            text = if (holdProgress > 0f) stringResource(R.string.live_hold_release_cancel)
                   else stringResource(R.string.live_hold_to_stop),
            color = if (holdProgress > 0f) HubColors.RedHot.copy(alpha = 0.7f) else HubColors.TextMuted,
            fontSize = 9.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp),
        )
    }
    } // Column
}

// ── BirdListItem ──────────────────────────────────────────────────────────────

private val birdEmojis = listOf("\uD83D\uDC26", "\uD83D\uDC24", "\uD83E\uDD9C", "\uD83D\uDD4A\uFE0F", "\uD83E\uDD89")

private fun birdEmoji(name: String): String =
    birdEmojis[kotlin.math.abs(name.hashCode()) % birdEmojis.size]

@Composable
fun BirdListItem(
    bird: DualDetectedBirdUi,
    isNew: Boolean,
    isFlash: Boolean,
    isLuring: Boolean,
    isDone: Boolean = false,
    onClick: () -> Unit,
    onLure: () -> Unit,
    onRemove: () -> Unit,
) {
    val rawBest = maxOf(bird.v24Confidence ?: 0, bird.v30Confidence ?: 0) / 100f
    var targetConf by remember { mutableFloatStateOf(rawBest * 0.65f) }
    LaunchedEffect(rawBest) { delay(200); targetConf = rawBest }
    val best = targetConf
    val col = confColor(best)

    var entered by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { entered = true }

    val cardBorder = when {
        isFlash -> Color(0xFF3DBA7E).copy(alpha = 0.7f)
        isLuring -> HubColors.Blue.copy(alpha = 0.35f)
        else -> HubColors.Border
    }

    val density = LocalDensity.current

    // Icon pop animation for flash
    var iconPopTrigger by remember { mutableStateOf(false) }
    val iconPopScale by animateFloatAsState(
        targetValue = if (iconPopTrigger) 1f else 0.72f,
        animationSpec = spring(dampingRatio = 0.45f, stiffness = 800f),
        label = "iconPopSpring",
    )
    LaunchedEffect(isFlash) { if (isFlash) { iconPopTrigger = false; delay(16); iconPopTrigger = true } }

    // Badge pop animation for new species
    var badgePopTrigger by remember { mutableStateOf(false) }
    val badgeScale by animateFloatAsState(
        targetValue = if (badgePopTrigger) 1f else 0.4f,
        animationSpec = spring(dampingRatio = 0.4f, stiffness = 600f),
        label = "badgePop",
    )
    val badgeOffsetY by animateFloatAsState(
        targetValue = if (badgePopTrigger) 0f else -4f,
        animationSpec = tween(400), label = "badgeOffset",
    )
    LaunchedEffect(isNew) { if (isNew) { badgePopTrigger = false; delay(80); badgePopTrigger = true } }

    // Lure pulse animation
    val lurePulseScale = if (isLuring) {
        val inf = rememberInfiniteTransition()
        val s by inf.animateFloat(1f, 1.06f, infiniteRepeatable(tween(1200, easing = FastOutSlowInEasing), RepeatMode.Reverse))
        s
    } else 1f

    // Smooth height entrance
    androidx.compose.animation.AnimatedVisibility(
        visible = entered,
        enter = androidx.compose.animation.expandVertically(
            animationSpec = tween(700, easing = CubicBezierEasing(0.22f, 0.61f, 0.36f, 1f)),
            expandFrom = Alignment.Top,
        ) + androidx.compose.animation.fadeIn(animationSpec = tween(500, delayMillis = 150)),
    ) {
    Box(
        modifier = Modifier.fillMaxWidth()
            // Flash glow: prototype boxShadow 0 0 14px green 0.3, 0 0 36px green 0.1
            .then(
                if (isFlash) Modifier.shadow(
                    elevation = 14.dp,
                    shape = RoundedCornerShape(12.dp),
                    ambientColor = Color(0xFF3DBA7E).copy(alpha = 0.4f),
                    spotColor = Color(0xFF3DBA7E).copy(alpha = 0.4f),
                ) else Modifier,
            )
            .clip(RoundedCornerShape(12.dp))
            .background(HubColors.BgCard)
            .border(1.5.dp, cardBorder, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
    ) {
        // Scan line (flash / luring)
        if (isFlash || isLuring) {
            val scanInf = rememberInfiniteTransition()
            val scanOffset by scanInf.animateFloat(
                -1.1f, 1.1f, infiniteRepeatable(tween(1100, easing = LinearEasing)),
            )
            val scanColor = if (isLuring) HubColors.Blue else col
            Box(
                modifier = Modifier.fillMaxWidth().height(1.5.dp).align(Alignment.TopCenter)
                    .background(
                        Brush.horizontalGradient(
                            listOf(Color.Transparent, scanColor, Color.Transparent),
                            startX = scanOffset * 400f, endX = scanOffset * 400f + 200f,
                        ),
                    ),
            )
        }
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(9.dp),
        ) {
            // Bird icon — 42dp matching prototype
            Box(
                modifier = Modifier.size(42.dp)
                    .scale(when {
                        isFlash -> iconPopScale; isLuring -> lurePulseScale; else -> 1f
                    })
                    .clip(RoundedCornerShape(10.dp))
                    .background(when {
                        isFlash -> col.copy(alpha = 0.16f)
                        isLuring -> HubColors.Blue.copy(alpha = 0.10f)
                        else -> col.copy(alpha = 0.06f)
                    })
                    .border(1.dp, when {
                        isFlash -> col.copy(alpha = 0.27f)
                        isLuring -> HubColors.Blue.copy(alpha = 0.21f)
                        else -> col.copy(alpha = 0.10f)
                    }, RoundedCornerShape(10.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Text(birdEmoji(bird.commonName), fontSize = 20.sp)
            }
            // Text column
            Column(modifier = Modifier.weight(1f)) {
                // Name row + badges
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text(bird.commonName, color = HubColors.TextPrimary, fontWeight = FontWeight.SemiBold, fontSize = 12.sp, letterSpacing = 0.15.sp, lineHeight = 14.sp)
                    if (isNew) {
                        Text(
                            stringResource(R.string.live_new_species), color = Color(0xFF00F5D4),
                            fontSize = 7.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 0.6.sp,
                            modifier = Modifier.scale(badgeScale)
                                .offset { IntOffset(0, with(density) { badgeOffsetY.dp.roundToPx() }) },
                        )
                    }
                    if (isFlash && !isNew) {
                        val blinkNow = rememberInfiniteTransition()
                        val nowAlpha by blinkNow.animateFloat(1f, 0.1f, infiniteRepeatable(tween(700, easing = FastOutSlowInEasing), RepeatMode.Reverse))
                        Text(stringResource(R.string.live_now_badge), color = col, fontSize = 8.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.alpha(nowAlpha))
                    }
                    if (isLuring) {
                        val blinkLure = rememberInfiniteTransition()
                        val lureAlpha by blinkLure.animateFloat(1f, 0.1f, infiniteRepeatable(tween(1200, easing = FastOutSlowInEasing), RepeatMode.Reverse))
                        Text("\u266B", color = HubColors.Blue, fontSize = 8.sp, fontWeight = FontWeight.Bold, modifier = Modifier.alpha(lureAlpha))
                    }
                }
                // Latin name
                Text(bird.scientificName, color = HubColors.TextMuted, fontSize = 9.sp, fontStyle = FontStyle.Italic, letterSpacing = 0.3.sp, lineHeight = 11.sp)
                Spacer(Modifier.height(2.dp))
                // ConfBar row
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                    if (isLuring) {
                        LureWaveBars(modifier = Modifier.weight(1f).height(10.dp))
                    } else {
                        ConfBar(best, modifier = Modifier.weight(1f))
                        Text(
                            "${(best * 100).roundToInt()}%", color = col, fontSize = 10.sp, fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace, textAlign = TextAlign.End, lineHeight = 12.sp,
                            modifier = Modifier.width(28.dp),
                        )
                    }
                }
            }
            // Lure button (hidden when session done)
            if (!isDone) {
                Box(
                    modifier = Modifier.scale(lurePulseScale)
                        .clip(RoundedCornerShape(7.dp))
                        .background(if (isLuring) HubColors.Blue else HubColors.BgEl2)
                        .border(1.dp, if (isLuring) HubColors.Blue.copy(alpha = 0.40f) else HubColors.Border, RoundedCornerShape(7.dp))
                        .clickable(onClick = onLure)
                        .padding(horizontal = 7.dp, vertical = 4.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(if (isLuring) "\u23F9" else "\uD83D\uDCE2", fontSize = 12.sp, color = if (isLuring) Color.White else HubColors.TextSecondary)
                }
            }
        }
    }
    } // AnimatedVisibility
}

// ── LureWaveBars (prototype: 22 fixed-height bars with scaleY animation) ─────

@Composable
private fun LureWaveBars(modifier: Modifier = Modifier) {
    // Pre-computed bar properties matching prototype:
    // height: 2 + abs(sin(i*1.4)) * 7, opacity: 0.55 + abs(sin(i)) * 0.45
    // animation: scaleY(0.35)→scaleY(1), random duration 0.5-1.0s, random delay 0-0.4s
    val barCount = 22
    val barHeights = remember {
        FloatArray(barCount) { i -> 2f + kotlin.math.abs(kotlin.math.sin(i * 1.4f)) * 7f }
    }
    val barOpacities = remember {
        FloatArray(barCount) { i -> 0.55f + kotlin.math.abs(kotlin.math.sin(i.toFloat())) * 0.45f }
    }
    // Each bar has random speed (period 0.5-1.0s) and phase offset (0-0.4s)
    val barPhases = remember { FloatArray(barCount) { (Math.random() * 0.4f).toFloat() } }
    val barSpeeds = remember { FloatArray(barCount) { (0.5f + Math.random().toFloat() * 0.5f) } }

    var tick by remember { mutableFloatStateOf(0f) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(33L)
            tick += 0.033f
        }
    }

    Canvas(modifier = modifier) {
        val barW = 2.dp.toPx()
        val gap = 1.5.dp.toPx()
        val maxH = size.height
        for (i in 0 until barCount) {
            val t = (tick - barPhases[i]).coerceAtLeast(0f)
            // scaleY oscillates between 0.35 and 1.0 (sin wave with bar's period)
            val phase = (t / barSpeeds[i]) * kotlin.math.PI.toFloat()
            val scaleY = 0.35f + 0.65f * kotlin.math.abs(kotlin.math.sin(phase))
            val h = (barHeights[i] / 9f) * maxH * scaleY // normalize to container height
            val x = i * (barW + gap)
            val y = (maxH - h) / 2f
            drawRoundRect(
                color = HubColors.Blue.copy(alpha = barOpacities[i]),
                topLeft = Offset(x, y),
                size = Size(barW, h),
                cornerRadius = CornerRadius(barW / 2f),
            )
        }
    }
}

// ── SessionDoneBlock ──────────────────────────────────────────────────────────

@Composable
fun SessionDoneBlock(
    timer: String,
    birdsCount: Int,
    onSave: () -> Unit,
    onDiscard: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        // Timer (prototype: fontSize 11, mono, letterSpacing 2)
        Text(
            text = timer,
            color = HubColors.TextMuted, fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 2.sp,
        )
        Spacer(Modifier.height(12.dp))
        // Two circle buttons (prototype: save=green 56dp, discard=red 56dp)
        Row(
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Save button — green circle with checkmark
            Box(
                modifier = Modifier.size(56.dp).clip(CircleShape)
                    .background(
                        Brush.radialGradient(
                            listOf(HubColors.Green, Color(0xFF2A9E68)),
                            center = Offset(21f, 19f),
                        ),
                    )
                    .clickable(onClick = onSave),
                contentAlignment = Alignment.Center,
            ) {
                Canvas(modifier = Modifier.size(22.dp)) {
                    val sw = 2.8.dp.toPx()
                    val sc = size.width / 24f
                    val path = Path().apply {
                        moveTo(20 * sc, 6 * sc)
                        lineTo(9 * sc, 17 * sc)
                        lineTo(4 * sc, 12 * sc)
                    }
                    drawPath(path, Color.White, style = Stroke(sw, cap = StrokeCap.Round, join = StrokeJoin.Round))
                }
            }
            // Discard button — red circle with X
            Box(
                modifier = Modifier.size(56.dp).clip(CircleShape)
                    .background(
                        Brush.radialGradient(
                            listOf(HubColors.RedHot, HubColors.RedDark),
                            center = Offset(21f, 19f),
                        ),
                    )
                    .clickable(onClick = onDiscard),
                contentAlignment = Alignment.Center,
            ) {
                Canvas(modifier = Modifier.size(19.dp)) {
                    val sw = 2.8.dp.toPx()
                    val sc = size.width / 24f
                    drawLine(Color.White, Offset(18 * sc, 6 * sc), Offset(6 * sc, 18 * sc), sw, cap = StrokeCap.Round)
                    drawLine(Color.White, Offset(6 * sc, 6 * sc), Offset(18 * sc, 18 * sc), sw, cap = StrokeCap.Round)
                }
            }
        }
    }
}

// ── ResetIcon ─────────────────────────────────────────────────────────────────

@Composable
fun ResetIcon(onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val isHovered by interactionSource.collectIsHoveredAsState()
    val bgColor = if (isHovered) HubColors.Red.copy(alpha = 0.13f) else HubColors.BgEl
    val borderColor = if (isHovered) HubColors.Red.copy(alpha = 0.27f) else HubColors.Border
    val strokeColor = if (isHovered) HubColors.RedHot else HubColors.TextMuted
    Box(
        modifier = Modifier.size(30.dp).clip(RoundedCornerShape(8.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .clickable(interactionSource = interactionSource, indication = null, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Canvas(modifier = Modifier.size(14.dp)) {
            val sw = 2.2.dp.toPx()
            val w = size.width
            val h = size.height
            val sc = w / 24f
            // Lid line: polyline 3,6 -> 5,6 -> 21,6
            drawLine(strokeColor, Offset(3*sc, 6*sc), Offset(21*sc, 6*sc), sw, cap = StrokeCap.Round)
            // Body: M19,6 l-1,14 H6 L5,6
            val bodyPath = Path().apply {
                moveTo(19*sc, 6*sc); lineTo(18*sc, 20*sc); lineTo(6*sc, 20*sc); lineTo(5*sc, 6*sc)
            }
            drawPath(bodyPath, strokeColor, style = Stroke(sw, cap = StrokeCap.Round, join = StrokeJoin.Round))
            // Lines inside: M10,11 v6  M14,11 v6
            drawLine(strokeColor, Offset(10*sc, 11*sc), Offset(10*sc, 17*sc), sw, cap = StrokeCap.Round)
            drawLine(strokeColor, Offset(14*sc, 11*sc), Offset(14*sc, 17*sc), sw, cap = StrokeCap.Round)
            // Handle: M9,6 V4 h6 v2
            val handlePath = Path().apply {
                moveTo(9*sc, 6*sc); lineTo(9*sc, 4*sc); lineTo(15*sc, 4*sc); lineTo(15*sc, 6*sc)
            }
            drawPath(handlePath, strokeColor, style = Stroke(sw, cap = StrokeCap.Round, join = StrokeJoin.Round))
        }
    }
}

// ── WaveformIcon (prototype: IconWaveform — 5 vertical bars) ─────────────────

@Composable
private fun WaveformIcon(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val sw = 2.2.dp.toPx()
        val sc = size.width / 24f
        val col = HubColors.Accent
        // 5 bars at x=4,8,12,16,20 with heights matching prototype
        drawLine(col, Offset(4 * sc, 8 * sc), Offset(4 * sc, 16 * sc), sw, cap = StrokeCap.Round)
        drawLine(col, Offset(8 * sc, 5 * sc), Offset(8 * sc, 19 * sc), sw, cap = StrokeCap.Round)
        drawLine(col, Offset(12 * sc, 3 * sc), Offset(12 * sc, 21 * sc), sw, cap = StrokeCap.Round)
        drawLine(col, Offset(16 * sc, 7 * sc), Offset(16 * sc, 17 * sc), sw, cap = StrokeCap.Round)
        drawLine(col, Offset(20 * sc, 10 * sc), Offset(20 * sc, 14 * sc), sw, cap = StrokeCap.Round)
    }
}
