package com.birdsong.analyzer.presentation.detection

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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.HubColors

@Composable
fun DualDetectionScreen(
    uiState: DualDetectionUiState = DualDetectionUiState(),
    onStart: () -> Unit = {},
    onPause: () -> Unit = {},
    onResume: () -> Unit = {},
    onStop: () -> Unit = {},
    onReset: () -> Unit = {},
    onSave: () -> Unit = {},
    onDiscard: () -> Unit = {},
    onBack: (() -> Unit)? = null,
    onRegionPress: () -> Unit = {},
    onBirdClick: (DualDetectedBirdUi) -> Unit = {},
    onHistory: () -> Unit = {},
) {
    val state = uiState.state
    val isStopped = state == DetectionState.STOPPED
    val isActive = state == DetectionState.ANALYZING || state == DetectionState.PAUSED

    Box(modifier = Modifier.fillMaxSize().background(HubColors.Bg)) {
        Column(modifier = Modifier.fillMaxSize()) {

            // ── Header ────────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                if (onBack != null) {
                    Box(
                        modifier = Modifier.clip(RoundedCornerShape(12.dp))
                            .background(HubColors.BgEl)
                            .border(1.dp, HubColors.Border, RoundedCornerShape(12.dp))
                            .clickable(onClick = onBack)
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                    ) {
                        Text("\u2190 \u041d\u0430\u0437\u0430\u0434", color = HubColors.TextSecondary, fontSize = 13.sp)
                    }
                    Spacer(Modifier.width(8.dp))
                }
                Column {
                    Text(
                        text = stringResource(R.string.live_screen_subtitle),
                        color = HubColors.TextMuted, fontSize = 9.sp, letterSpacing = 1.2.sp,
                    )
                    Text(
                        text = stringResource(R.string.live_screen_app_name),
                        color = HubColors.TextPrimary, fontSize = 15.sp,
                        fontWeight = FontWeight.ExtraBold, letterSpacing = 1.5.sp,
                    )
                }
                Spacer(Modifier.weight(1f))
                RegionChip(uiState.regionLabel, onRegionPress)
            }

            // ── Record button area or session complete ────────────────────
            if (!isStopped) {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    RecordButton(
                        state = state,
                        birds = uiState.birds,
                        onPress = {
                            when (state) {
                                DetectionState.IDLE -> if (uiState.regionLabel == null) onRegionPress() else onStart()
                                DetectionState.ANALYZING -> onPause()
                                DetectionState.PAUSED -> onResume()
                                else -> {}
                            }
                        },
                    )
                }
                Text(
                    text = if (isActive) uiState.sessionTimer else "",
                    color = HubColors.TextMuted, fontSize = 18.sp, fontWeight = FontWeight.Light,
                    letterSpacing = 8.sp, textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    text = when (state) {
                        DetectionState.ANALYZING -> stringResource(R.string.live_listening)
                        DetectionState.PAUSED -> stringResource(R.string.live_pause_label)
                        DetectionState.PREPARING -> stringResource(R.string.detection_preparing)
                        else -> stringResource(R.string.live_start_hint)
                    },
                    color = when (state) {
                        DetectionState.ANALYZING -> HubColors.Red
                        DetectionState.PAUSED -> HubColors.Yellow
                        else -> HubColors.TextMuted
                    },
                    fontSize = 13.sp,
                    fontWeight = if (isActive) FontWeight.SemiBold else FontWeight.Normal,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth().padding(bottom = 14.dp),
                )
            } else {
                SessionCompleteBanner(uiState.sessionTimer, uiState.birds.size, uiState.regionLabel, onDiscard)
            }

            // ── Action buttons ─────────────────────────────────────────────
            if (isActive) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Box(
                        modifier = Modifier.weight(1f).clip(RoundedCornerShape(10.dp))
                            .background(HubColors.BgEl)
                            .border(1.dp, HubColors.Border, RoundedCornerShape(10.dp))
                            .clickable(onClick = onReset).padding(vertical = 10.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "\uD83D\uDDD1 ${stringResource(R.string.btn_reset_short)}",
                            color = HubColors.TextMuted, fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
                        )
                    }
                    Box(
                        modifier = Modifier.weight(1f).clip(RoundedCornerShape(10.dp))
                            .background(HubColors.Red.copy(alpha = 0.18f))
                            .border(1.dp, HubColors.Red.copy(alpha = 0.44f), RoundedCornerShape(10.dp))
                            .clickable(onClick = onStop).padding(vertical = 10.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "\u23F9 ${stringResource(R.string.btn_stop)}",
                            color = HubColors.Red, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        )
                    }
                }
            }

            // ── WaveBars ───────────────────────────────────────────────────
            Box(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 4.dp)
                    .background(HubColors.BgEl, RoundedCornerShape(14.dp))
                    .border(1.dp, HubColors.Border, RoundedCornerShape(14.dp))
                    .padding(horizontal = 14.dp, vertical = 8.dp),
            ) {
                WaveBars(
                    active = state == DetectionState.ANALYZING,
                    modifier = Modifier.fillMaxWidth().height(24.dp),
                )
            }

            // ── Birds header ───────────────────────────────────────────────
            if (uiState.birds.isNotEmpty()) {
                Row(modifier = Modifier.padding(horizontal = 18.dp, vertical = 2.dp)) {
                    Text(
                        text = stringResource(R.string.live_detected_count, uiState.birds.size),
                        color = HubColors.TextMuted, fontSize = 11.sp,
                    )
                    if (isStopped) {
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = "\u00b7 ${stringResource(R.string.live_tap_hint)}",
                            color = HubColors.TextMuted, fontSize = 11.sp,
                        )
                    }
                }
            }

            // ── Idle empty state or bird list ──────────────────────────────
            if (state == DetectionState.IDLE && uiState.birds.isEmpty()) {
                IdleEmptyState(uiState.regionLabel, onRegionPress, modifier = Modifier.weight(1f))
            } else {
                LazyColumn(
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(
                        start = 18.dp, end = 18.dp,
                        top = 4.dp,
                        bottom = if (isStopped) 88.dp else 8.dp,
                    ),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(uiState.birds, key = { it.id }) { bird ->
                        BirdListItem(
                            bird = bird,
                            isActive = bird.id == uiState.activelyDetectedBirdId &&
                                state == DetectionState.ANALYZING,
                            onClick = { onBirdClick(bird) },
                            onLure = {},
                        )
                    }
                }
            }
        }

        // ── Save footer (stopped) ──────────────────────────────────────────
        if (isStopped) {
            Column(
                modifier = Modifier.align(Alignment.BottomCenter)
                    .fillMaxWidth().background(HubColors.Bg),
            ) {
                HorizontalDivider(color = HubColors.Border)
                Row(
                    modifier = Modifier.padding(horizontal = 18.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Box(
                        modifier = Modifier.weight(2f).clip(RoundedCornerShape(14.dp))
                            .background(HubColors.Accent)
                            .clickable(onClick = onSave)
                            .padding(vertical = 13.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = "\uD83D\uDCAE ${stringResource(R.string.btn_save_session)}",
                            color = HubColors.Bg, fontWeight = FontWeight.Bold, fontSize = 14.sp,
                        )
                    }
                    Box(
                        modifier = Modifier.weight(1f).clip(RoundedCornerShape(14.dp))
                            .background(HubColors.BgEl)
                            .border(1.dp, HubColors.Border, RoundedCornerShape(14.dp))
                            .clickable(onClick = onDiscard)
                            .padding(vertical = 13.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = stringResource(R.string.btn_cancel),
                            color = HubColors.TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
            }
        }

        // ── History FAB (idle) ─────────────────────────────────────────────
        if (state == DetectionState.IDLE) {
            Column(
                modifier = Modifier.align(Alignment.BottomEnd)
                    .padding(end = 18.dp, bottom = 16.dp)
                    .size(52.dp).clip(CircleShape)
                    .background(HubColors.BgCard)
                    .border(1.5.dp, HubColors.Border, CircleShape)
                    .clickable(onClick = onHistory),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                Text("\uD83D\uDDC2", fontSize = 16.sp)
                Text(
                    text = stringResource(R.string.live_history_label),
                    color = HubColors.TextMuted, fontSize = 7.sp, fontWeight = FontWeight.SemiBold,
                    letterSpacing = 0.3.sp,
                )
            }
        }
    }
}

// ── RegionChip ────────────────────────────────────────────────────────────────

@Composable
private fun RegionChip(label: String?, onPress: () -> Unit) {
    Box(
        modifier = Modifier.clip(RoundedCornerShape(20.dp))
            .background(HubColors.BgEl)
            .border(
                1.dp,
                if (label != null) HubColors.Border else HubColors.Accent.copy(alpha = 0.66f),
                RoundedCornerShape(20.dp),
            )
            .clickable(onClick = onPress)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Text(
            text = "\uD83D\uDCCD ${label ?: stringResource(R.string.live_select_region)}",
            color = if (label != null) HubColors.TextSecondary else HubColors.Accent,
            fontSize = 11.sp, maxLines = 1,
        )
    }
}

// ── IdleEmptyState ────────────────────────────────────────────────────────────

@Composable
private fun IdleEmptyState(
    regionLabel: String?,
    onRegionPress: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.padding(horizontal = 18.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        if (regionLabel == null) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
                    .background(HubColors.Accent.copy(alpha = 0.06f), RoundedCornerShape(14.dp))
                    .border(1.dp, HubColors.Accent.copy(alpha = 0.33f), RoundedCornerShape(14.dp))
                    .padding(horizontal = 12.dp, vertical = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("\uD83D\uDCCD", fontSize = 20.sp)
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = stringResource(R.string.live_region_hint_title),
                        color = HubColors.Accent, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        text = stringResource(R.string.live_region_hint_desc),
                        color = HubColors.TextMuted, fontSize = 11.sp,
                    )
                }
                Box(
                    modifier = Modifier.clip(RoundedCornerShape(10.dp))
                        .background(HubColors.Accent)
                        .clickable(onClick = onRegionPress)
                        .padding(horizontal = 12.dp, vertical = 7.dp),
                ) {
                    Text(
                        text = stringResource(R.string.live_region_btn),
                        color = HubColors.Bg, fontWeight = FontWeight.Bold, fontSize = 12.sp,
                    )
                }
            }
        }
        Box(
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("\uD83C\uDFA7", fontSize = 36.sp)
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.live_idle_desc),
                    color = HubColors.TextMuted, fontSize = 13.sp,
                    textAlign = TextAlign.Center, lineHeight = 20.sp,
                )
            }
        }
    }
}

// ── Previews ──────────────────────────────────────────────────────────────────

private val previewBirds = listOf(
    DualDetectedBirdUi("1", "Иволга", "Oriolus oriolus", v24Confidence = 94, v30Confidence = 88, detectedAt = "05:32"),
    DualDetectedBirdUi("2", "Зяблик", "Fringilla coelebs", v24Confidence = 78, detectedAt = "04:18"),
    DualDetectedBirdUi("3", "Большая синица", "Parus major", v30Confidence = 61, detectedAt = "03:10"),
)

@Preview(showBackground = true, showSystemUi = true, name = "IDLE")
@Composable
private fun PreviewIdle() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        DualDetectionScreen()
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "ANALYZING")
@Composable
private fun PreviewAnalyzing() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        DualDetectionScreen(
            uiState = DualDetectionUiState(
                state = DetectionState.ANALYZING, sessionTimer = "05:32",
                birds = previewBirds, regionLabel = "Минская обл., BY",
                activelyDetectedBirdId = "1", v30Available = true,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "STOPPED")
@Composable
private fun PreviewStopped() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        DualDetectionScreen(
            uiState = DualDetectionUiState(
                state = DetectionState.STOPPED, sessionTimer = "12:34",
                birds = previewBirds, regionLabel = "Минская обл., BY",
            ),
        )
    }
}
