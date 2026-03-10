package com.birdsong.analyzer.presentation.detection

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
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
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.HubColors

private data class ModeItem(
    val labelRes: Int,
    val descRes: Int,
    val icon: String,
    val color: Color,
    val gradStart: Color,
    val gradEnd: Color,
    val soon: Boolean = false,
    val action: ActionType,
)

private enum class ActionType { LIVE, FILE, NONE }

private val MODES = listOf(
    ModeItem(
        labelRes = R.string.hub_live_label,
        descRes = R.string.hub_live_desc,
        icon = "🎙",
        color = HubColors.Green,
        gradStart = Color(0xFF0D2E1F),
        gradEnd = Color(0xFF0A1F16),
        action = ActionType.LIVE,
    ),
    ModeItem(
        labelRes = R.string.hub_file_label,
        descRes = R.string.hub_file_desc,
        icon = "📁",
        color = HubColors.Blue,
        gradStart = Color(0xFF0D1E2E),
        gradEnd = Color(0xFF0A1520),
        action = ActionType.FILE,
    ),
    ModeItem(
        labelRes = R.string.hub_trap_label,
        descRes = R.string.hub_trap_desc,
        icon = "🔭",
        color = HubColors.Accent,
        gradStart = Color(0xFF2A1E08),
        gradEnd = Color(0xFF1C1408),
        soon = true,
        action = ActionType.NONE,
    ),
    ModeItem(
        labelRes = R.string.hub_exp_label,
        descRes = R.string.hub_exp_desc,
        icon = "🗺",
        color = HubColors.Purple,
        gradStart = Color(0xFF1A1228),
        gradEnd = Color(0xFF120D1E),
        soon = true,
        action = ActionType.NONE,
    ),
)

@Composable
fun HomeScreen(
    onNavigateToLiveDetection: () -> Unit = {},
    onNavigateToFileAnalysis: () -> Unit = {},
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(HubColors.Bg),
    ) {
        // ── Header ────────────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 20.dp, end = 20.dp, top = 20.dp, bottom = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Image(
                painter = painterResource(R.drawable.ic_avalga_logo),
                contentDescription = null,
                modifier = Modifier.size(32.dp),
            )
            Column {
                Text(
                    text = "AVALGA",
                    color = HubColors.TextPrimary,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.ExtraBold,
                    letterSpacing = 1.5.sp,
                )
                Text(
                    text = "BIRD SOUND ID",
                    color = HubColors.TextMuted,
                    fontSize = 10.sp,
                    letterSpacing = 1.sp,
                )
            }
        }

        // ── Mode cards ────────────────────────────────────────────────────────
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            contentPadding = PaddingValues(bottom = 16.dp),
        ) {
            items(MODES) { mode ->
                val onClick: (() -> Unit)? = when {
                    mode.soon -> null
                    mode.action == ActionType.LIVE -> onNavigateToLiveDetection
                    mode.action == ActionType.FILE -> onNavigateToFileAnalysis
                    else -> null
                }
                ModeCard(mode = mode, onClick = onClick)
            }
        }
    }
}

@Composable
private fun ModeCard(mode: ModeItem, onClick: (() -> Unit)?) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.98f else 1f,
        animationSpec = tween(durationMillis = 150),
        label = "cardScale",
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .scale(scale)
            .clip(RoundedCornerShape(20.dp))
            .background(Brush.linearGradient(colors = listOf(mode.gradStart, mode.gradEnd)))
            .then(
                if (onClick != null) {
                    Modifier.clickable(
                        interactionSource = interactionSource,
                        indication = null,
                        onClick = onClick,
                    )
                } else {
                    Modifier.alpha(0.45f)
                },
            ),
    ) {
        // Top-edge shimmer line
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth(0.8f)
                .height(1.dp)
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(
                            Color.Transparent,
                            mode.color.copy(alpha = 0.5f),
                            Color.Transparent,
                        ),
                    ),
                ),
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Icon box
            Box(
                modifier = Modifier
                    .size(52.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .background(mode.color.copy(alpha = 0.12f))
                    .border(1.dp, mode.color.copy(alpha = 0.22f), RoundedCornerShape(16.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Text(text = mode.icon, fontSize = 24.sp)
            }

            // Labels
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = stringResource(mode.labelRes),
                    color = HubColors.TextPrimary,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.2.sp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = stringResource(mode.descRes),
                    color = mode.color.copy(alpha = 0.6f),
                    fontSize = 12.sp,
                )
            }

            // Trailing: "Скоро" tag or chevron circle
            if (mode.soon) {
                Box(
                    modifier = Modifier
                        .background(
                            HubColors.TextSecondary.copy(alpha = 0.12f),
                            RoundedCornerShape(20.dp),
                        )
                        .padding(horizontal = 8.dp, vertical = 3.dp),
                ) {
                    Text(
                        text = stringResource(R.string.hub_soon),
                        color = HubColors.TextSecondary,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 0.3.sp,
                    )
                }
            } else {
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(mode.color.copy(alpha = 0.12f))
                        .border(1.dp, mode.color.copy(alpha = 0.22f), CircleShape),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "›",
                        color = mode.color,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Mode Hub - Dark")
@Composable
private fun PreviewModeHub() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        HomeScreen()
    }
}
