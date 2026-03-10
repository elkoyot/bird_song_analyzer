package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.StartOffset
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.HubColors
import kotlin.math.roundToInt

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
    birds: List<DualDetectedBirdUi>,
    onPress: () -> Unit,
) {
    val isRec = state == DetectionState.ANALYZING
    val isPaused = state == DetectionState.PAUSED
    val isPreparing = state == DetectionState.PREPARING
    val buttonColor = when {
        isRec -> HubColors.Red
        isPaused -> HubColors.Yellow
        else -> HubColors.Accent
    }
    val icon = when {
        isRec -> "\u23F9"
        isPaused -> "\u25B6"
        else -> "\uD83C\uDF99"
    }

    val inf = rememberInfiniteTransition()

    val breathScale by inf.animateFloat(
        initialValue = 1f, targetValue = 1.12f,
        animationSpec = infiniteRepeatable(tween(2800, easing = FastOutSlowInEasing), RepeatMode.Reverse),
    )

    val r0 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(2200, easing = LinearEasing)))
    val r1 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(2700, easing = LinearEasing), initialStartOffset = StartOffset(500)))
    val r2 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(3200, easing = LinearEasing), initialStartOffset = StartOffset(1000)))
    val r3 by inf.animateFloat(0f, 1f, infiniteRepeatable(tween(3700, easing = LinearEasing), initialStartOffset = StartOffset(1500)))

    Box(modifier = Modifier.size(200.dp), contentAlignment = Alignment.Center) {
        if (isRec) {
            RadarCanvas(birds = birds, modifier = Modifier.matchParentSize())
        }
        if (!isRec && !isPaused && !isPreparing) {
            Box(
                modifier = Modifier.size(120.dp).scale(breathScale)
                    .border(1.5.dp, HubColors.Accent.copy(alpha = 0.25f), CircleShape),
            )
        }
        if (isRec) {
            Box(modifier = Modifier.size(96.dp).scale(1f + r0 * 0.5f).alpha((1f - r0).coerceAtLeast(0f)).border(1.4.dp, HubColors.Red, CircleShape))
            Box(modifier = Modifier.size(118.dp).scale(1f + r1 * 0.5f).alpha((1f - r1).coerceAtLeast(0f)).border(1.2.dp, HubColors.Red, CircleShape))
            Box(modifier = Modifier.size(140.dp).scale(1f + r2 * 0.5f).alpha((1f - r2).coerceAtLeast(0f)).border(1.0.dp, HubColors.Red, CircleShape))
            Box(modifier = Modifier.size(162.dp).scale(1f + r3 * 0.5f).alpha((1f - r3).coerceAtLeast(0f)).border(0.8.dp, HubColors.Red, CircleShape))
        }
        Box(
            modifier = Modifier.size(96.dp).clip(CircleShape)
                .background(buttonColor)
                .then(if (!isPreparing) Modifier.clickable(onClick = onPress) else Modifier),
            contentAlignment = Alignment.Center,
        ) {
            Text(text = icon, fontSize = 28.sp)
        }
    }
}

// ── BirdListItem ──────────────────────────────────────────────────────────────

@Composable
fun BirdListItem(
    bird: DualDetectedBirdUi,
    isActive: Boolean,
    onClick: () -> Unit,
    onLure: () -> Unit,
) {
    val best = maxOf(bird.v24Confidence ?: 0, bird.v30Confidence ?: 0) / 100f
    val col = confColor(best)

    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { visible = true }
    val alpha by animateFloatAsState(if (visible) 1f else 0f, tween(350), label = "cardAlpha")

    val inf = rememberInfiniteTransition()
    val dotAlpha by inf.animateFloat(
        initialValue = 0.2f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(800, easing = FastOutSlowInEasing), RepeatMode.Reverse),
    )

    Box(
        modifier = Modifier.fillMaxWidth().alpha(alpha)
            .clip(RoundedCornerShape(16.dp))
            .background(if (isActive) col.copy(alpha = 0.09f) else HubColors.BgCard)
            .border(1.5.dp, if (isActive) col.copy(alpha = 0.44f) else HubColors.Border, RoundedCornerShape(16.dp))
            .clickable(onClick = onClick),
    ) {
        if (isActive) {
            Box(
                modifier = Modifier.fillMaxWidth().height(2.dp).align(Alignment.TopCenter)
                    .background(
                        androidx.compose.ui.graphics.Brush.horizontalGradient(
                            listOf(Color.Transparent, col.copy(alpha = 0.8f), Color.Transparent),
                        ),
                    ),
            )
        }
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Box(
                modifier = Modifier.size(46.dp).clip(RoundedCornerShape(13.dp))
                    .background(col.copy(alpha = 0.18f))
                    .border(1.dp, col.copy(alpha = 0.33f), RoundedCornerShape(13.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83D\uDC26", fontSize = 22.sp)
            }
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    Text(bird.commonName, color = HubColors.TextPrimary, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                    if (isActive) {
                        Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(col.copy(alpha = dotAlpha)))
                    }
                }
                Text(bird.scientificName, color = HubColors.TextMuted, fontSize = 11.sp, fontStyle = FontStyle.Italic)
                Spacer(Modifier.height(5.dp))
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ConfBar(best, modifier = Modifier.weight(1f))
                    Text(
                        text = "${(best * 100).roundToInt()}%",
                        color = col, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                    )
                }
            }
            Box(
                modifier = Modifier.size(40.dp).clip(RoundedCornerShape(10.dp))
                    .background(HubColors.BgEl)
                    .border(1.dp, HubColors.Border, RoundedCornerShape(10.dp))
                    .clickable(onClick = onLure),
                contentAlignment = Alignment.Center,
            ) {
                Text("\uD83D\uDCE2", fontSize = 16.sp)
            }
        }
    }
}

// ── SessionCompleteBanner ─────────────────────────────────────────────────────

@Composable
fun SessionCompleteBanner(
    timer: String,
    birdsCount: Int,
    regionLabel: String?,
    onNewSession: () -> Unit,
) {
    Box(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 8.dp)
            .background(HubColors.Green.copy(alpha = 0.06f), RoundedCornerShape(16.dp))
            .border(1.dp, HubColors.Green.copy(alpha = 0.33f), RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Box(
                modifier = Modifier.size(36.dp).clip(CircleShape)
                    .background(HubColors.Green.copy(alpha = 0.22f)),
                contentAlignment = Alignment.Center,
            ) {
                Text("\u2713", color = HubColors.Green, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(R.string.live_session_done),
                    color = HubColors.Green, fontWeight = FontWeight.Bold, fontSize = 13.sp,
                )
                Text(
                    text = "$timer \u00b7 $birdsCount ${stringResource(R.string.live_species_unit)} \u00b7 ${regionLabel ?: "\u2014"}",
                    color = HubColors.TextMuted, fontSize = 11.sp,
                )
            }
            Box(
                modifier = Modifier.clip(RoundedCornerShape(10.dp))
                    .background(HubColors.Green.copy(alpha = 0.18f))
                    .border(1.dp, HubColors.Green.copy(alpha = 0.44f), RoundedCornerShape(10.dp))
                    .clickable(onClick = onNewSession)
                    .padding(horizontal = 12.dp, vertical = 7.dp),
            ) {
                Text(
                    text = stringResource(R.string.btn_new_session),
                    color = HubColors.Green, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}
