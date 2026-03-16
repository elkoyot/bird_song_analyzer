package com.birdsong.analyzer.presentation.detection

import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.MenuBook
import androidx.compose.material.icons.outlined.Person
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
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
    onRemoveBird: (String) -> Unit = {},
    onLure: (String) -> Unit = {},
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
                modifier = Modifier.fillMaxWidth().padding(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    AvalgaLogo(modifier = Modifier.size(28.dp))
                    Text(
                        text = stringResource(R.string.live_screen_app_name),
                        color = HubColors.TextPrimary, fontSize = 16.sp,
                        fontWeight = FontWeight.ExtraBold, letterSpacing = 1.4.sp,
                    )
                }
                Spacer(Modifier.weight(1f))
                RegionChip(uiState.regionLabel, isActive, onRegionPress)
            }

            // ── Record button area or session complete ────────────────────
            if (!isStopped) {
                Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    RecordButton(
                        state = state,
                        blipSeq = uiState.blipSeq,
                        sessionTimer = uiState.sessionTimer,
                        onPress = {
                            when (state) {
                                DetectionState.IDLE -> onStart()
                                DetectionState.ANALYZING -> onPause()
                                DetectionState.PAUSED -> onResume()
                                else -> {}
                            }
                        },
                        onLongPressStop = onStop,
                    )
                }
            } else {
                Box(modifier = Modifier.padding(start = 18.dp, end = 18.dp, top = 10.dp, bottom = 4.dp)) {
                    SessionDoneBlock(uiState.sessionTimer, uiState.birds.size, onSave, onDiscard)
                }
            }

            // ── Birds header ───────────────────────────────────────────────
            if (uiState.birds.isNotEmpty()) {
                val newCount = uiState.newBirdIds.size
                Row(
                    modifier = Modifier.fillMaxWidth().defaultMinSize(minHeight = 28.dp)
                        .padding(start = 18.dp, end = 18.dp, top = 2.dp, bottom = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Определено: ",
                        color = HubColors.TextMuted, fontSize = 10.sp,
                    )
                    Text(
                        text = "${uiState.birds.size}",
                        color = HubColors.Green, fontSize = 10.sp, fontWeight = FontWeight.Bold,
                    )
                    if (newCount > 0) {
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.live_new_count, newCount),
                            color = HubColors.Accent, fontSize = 10.sp, fontWeight = FontWeight.SemiBold,
                        )
                    }
                    if (uiState.luringBirdId != null) {
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.live_lure_active),
                            color = HubColors.Blue, fontSize = 10.sp, fontWeight = FontWeight.SemiBold,
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    ResetIcon(onClick = onReset)
                }
            }

            // ── Bird list (or empty area) ─────────────────────────────────
            val listState = rememberLazyListState()
            var prevBirdCount by remember { mutableIntStateOf(0) }
            LaunchedEffect(uiState.birds.size) {
                val count = uiState.birds.size
                if (count > prevBirdCount) {
                    kotlinx.coroutines.delay(100L)
                    listState.animateScrollToItem(0)
                }
                prevBirdCount = count
            }
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(
                    start = 18.dp, end = 18.dp,
                    top = 0.dp,
                    bottom = 12.dp,
                ),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                items(uiState.birds, key = { it.id }) { bird ->
                    BirdListItem(
                        bird = bird,
                        isNew = bird.id in uiState.newBirdIds,
                        isFlash = bird.id == uiState.flashBirdId,
                        isLuring = bird.id == uiState.luringBirdId,
                        isDone = isStopped,
                        onClick = { onBirdClick(bird) },
                        onLure = { onLure(bird.id) },
                        onRemove = { onRemoveBird(bird.id) },
                    )
                }
            }

            // ── BottomNav ──────────────────────────────────────────────────
            BottomNav()
        }

        // ── History FAB (idle) ─────────────────────────────────────────────
        if (state == DetectionState.IDLE) {
            Column(
                modifier = Modifier.align(Alignment.BottomEnd)
                    .padding(end = 18.dp, bottom = 68.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                Box(
                    modifier = Modifier.size(46.dp)
                        .shadow(18.dp, CircleShape, ambientColor = Color.Black.copy(alpha = 0.55f))
                        .clip(CircleShape)
                        .background(HubColors.BgCard)
                        .border(1.5.dp, HubColors.Border, CircleShape)
                        .clickable(onClick = onHistory),
                    contentAlignment = Alignment.Center,
                ) {
                    ClockIcon(modifier = Modifier.size(20.dp), color = HubColors.TextSecondary)
                }
                Text(
                    text = stringResource(R.string.live_history_label).uppercase(),
                    color = HubColors.TextMuted, fontSize = 7.sp, fontWeight = FontWeight.SemiBold,
                    letterSpacing = 0.4.sp,
                )
            }
        }
    }
}

// ── RegionChip ────────────────────────────────────────────────────────────────

@Composable
private fun RegionChip(label: String?, isActive: Boolean, onPress: () -> Unit) {
    Box(
        modifier = Modifier.alpha(if (isActive) 0.45f else 1f)
            .clip(RoundedCornerShape(20.dp))
            .background(HubColors.BgEl)
            .border(1.dp, HubColors.Border, RoundedCornerShape(20.dp))
            .then(if (!isActive) Modifier.clickable(onClick = onPress) else Modifier)
            .padding(horizontal = 11.dp, vertical = 5.dp),
    ) {
        Text(
            text = "\uD83D\uDCCD ${label ?: stringResource(R.string.live_select_region)}",
            color = HubColors.TextSecondary,
            fontSize = 11.sp, maxLines = 1,
        )
    }
}

// ── AvalgaLogo (1.1) ─────────────────────────────────────────────────────────

@Composable
private fun AvalgaLogo(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val sx = size.width / 200f
        val sy = size.height / 200f
        fun poly(pts: List<Pair<Float, Float>>, color: Color) {
            val path = Path().apply {
                moveTo(pts[0].first * sx, pts[0].second * sy)
                for (i in 1 until pts.size) lineTo(pts[i].first * sx, pts[i].second * sy)
                close()
            }
            drawPath(path, color)
        }
        // Tail feathers
        poly(listOf(48f to 130f, 18f to 170f, 38f to 165f, 55f to 145f), Color(0xFFE8A020))
        poly(listOf(55f to 145f, 38f to 165f, 58f to 158f, 68f to 148f), Color(0xFF1A1A1A))
        // Body
        poly(listOf(55f to 90f, 100f to 80f, 110f to 115f, 65f to 130f), Color(0xFFF5C300))
        poly(listOf(65f to 130f, 110f to 115f, 95f to 140f, 60f to 145f), Color(0xFFE8A020))
        poly(listOf(80f to 90f, 110f to 80f, 115f to 100f, 90f to 108f), Color(0xFF1A1A1A))
        // Head
        poly(listOf(110f to 75f, 135f to 60f, 148f to 75f, 130f to 90f, 108f to 88f), Color(0xFFF5C300))
        // Eye
        drawCircle(Color(0xFF1A1A1A), radius = 5f * sx, center = Offset(143f * sx, 65f * sy))
        drawCircle(Color.White, radius = 1.8f * sx, center = Offset(144f * sx, 64f * sy))
        // Beak
        poly(listOf(155f to 65f, 178f to 60f, 172f to 70f, 155f to 70f), Color(0xFFC85A3A))
        // Wing
        poly(listOf(55f to 90f, 48f to 115f, 20f to 88f, 15f to 60f, 45f to 70f), Color(0xFFE8A020))
        poly(listOf(45f to 70f, 15f to 60f, 22f to 42f, 50f to 55f), Color(0xFFF5C300))
        // Crest
        poly(listOf(50f to 35f, 35f to 18f, 42f to 10f, 54f to 28f), Color(0xFFE8A020))
        // Wing shadow
        poly(listOf(48f to 115f, 20f to 88f, 10f to 105f, 35f to 125f), Color(0xFF2D2D2D))
        // Tail accent
        poly(listOf(38f to 135f, 15f to 120f, 22f to 132f, 45f to 140f), Color(0xFFE8A020))
    }
}

// ── BottomNav (prototype: 4 tabs — Домой, Справочник, Профиль, Настройки) ─────

private data class NavItem(val id: String, val label: String)

private val navItems = listOf(
    NavItem("home", "ДОМОЙ"),
    NavItem("reference", "СПРАВОЧНИК"),
    NavItem("profile", "ПРОФИЛЬ"),
    NavItem("settings", "НАСТРОЙКИ"),
)

@Composable
private fun BottomNav() {
    var activeTab by remember { mutableStateOf("home") }
    Row(
        modifier = Modifier.fillMaxWidth()
            .background(HubColors.NavBg)
            .drawBehind {
                drawLine(
                    color = HubColors.NavBorder,
                    start = Offset(0f, 0f),
                    end = Offset(size.width, 0f),
                    strokeWidth = 1.dp.toPx(),
                )
            }
            .padding(bottom = 6.dp),
    ) {
        navItems.forEach { item ->
            val isActive = item.id == activeTab
            val iconColor = if (isActive) HubColors.Accent else HubColors.TextMuted
            val labelColor = if (isActive) HubColors.Accent else HubColors.TextMuted
            Box(
                modifier = Modifier.weight(1f)
                    .clickable { activeTab = item.id },
                contentAlignment = Alignment.Center,
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(top = 10.dp, bottom = 6.dp),
                ) {
                    val icon = when (item.id) {
                        "home" -> Icons.Outlined.Home
                        "reference" -> Icons.Outlined.MenuBook
                        "profile" -> Icons.Outlined.Person
                        "settings" -> Icons.Outlined.Settings
                        else -> Icons.Outlined.Home
                    }
                    Icon(icon, contentDescription = item.label, tint = iconColor, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.height(3.dp))
                    Text(
                        text = item.label,
                        color = labelColor,
                        fontSize = 8.sp,
                        fontWeight = if (isActive) FontWeight.Bold else FontWeight.Normal,
                        letterSpacing = 0.3.sp,
                    )
                }
                if (isActive) {
                    Box(
                        modifier = Modifier.align(Alignment.BottomCenter)
                            .width(18.dp).height(2.dp)
                            .clip(RoundedCornerShape(2.dp))
                            .background(HubColors.Accent),
                    )
                }
            }
        }
    }
}


// ── ClockIcon (SVG-like, matches prototype IconClock) ─────────────────────────

@Composable
private fun ClockIcon(modifier: Modifier = Modifier, color: Color = HubColors.TextMuted) {
    Canvas(modifier = modifier) {
        val sw = 2.dp.toPx()
        val cx = size.width / 2f
        val cy = size.height / 2f
        val r = cx - sw
        val sc = size.width / 24f
        // Circle
        drawCircle(color = color, radius = r, center = Offset(cx, cy), style = Stroke(sw))
        // Hour hand: 12 to 12 (vertical up to center)
        drawLine(color, Offset(12 * sc, 6 * sc), Offset(12 * sc, 12 * sc), sw, cap = StrokeCap.Round)
        // Minute hand: center to 4 o'clock
        drawLine(color, Offset(12 * sc, 12 * sc), Offset(16 * sc, 14 * sc), sw, cap = StrokeCap.Round)
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
                flashBirdId = "1", newBirdIds = setOf("1"), v30Available = true,
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
