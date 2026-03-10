package com.birdsong.analyzer.presentation.detail

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.detection.ConfBar
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.HubColors

data class DetailUiState(
    val commonName: String = "",
    val scientificName: String = "",
    val confidence: Int = 0,
    val detectedAt: String = "",
    val durationSec: String = "",
    val latitude: Double? = null,
    val longitude: Double? = null,
    val isPlaying: Boolean = false,
    val playbackProgress: Float = 0f,
)

private fun confColor(pct: Int) = when {
    pct >= 75 -> HubColors.Green
    pct >= 35 -> HubColors.Yellow
    else -> HubColors.Red
}

@Composable
fun DetailScreen(
    uiState: DetailUiState = DetailUiState(),
    onBack: () -> Unit = {},
    onPlayPause: () -> Unit = {},
) {
    val col = confColor(uiState.confidence)
    var tab by remember { mutableStateOf("info") }
    var saved by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    Box(modifier = Modifier.fillMaxSize().background(HubColors.Bg)) {
        Column(modifier = Modifier.fillMaxSize()) {

            // ── Hero ──────────────────────────────────────────────────────
            Box(
                modifier = Modifier.fillMaxWidth().height(200.dp)
                    .background(Brush.linearGradient(listOf(col.copy(alpha = 0.22f), HubColors.BgCard))),
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    modifier = Modifier.size(180.dp).clip(CircleShape)
                        .background(
                            Brush.radialGradient(listOf(col.copy(alpha = 0.22f), Color.Transparent)),
                        ),
                )
                Text("\uD83D\uDC26", fontSize = 88.sp)
                Box(modifier = Modifier.align(Alignment.TopStart).padding(14.dp)) {
                    Box(
                        modifier = Modifier.clip(RoundedCornerShape(12.dp))
                            .background(HubColors.BgEl)
                            .border(1.dp, HubColors.Border, RoundedCornerShape(12.dp))
                            .clickable(onClick = onBack)
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                    ) {
                        Text("\u2190 \u041d\u0430\u0437\u0430\u0434", color = HubColors.TextSecondary, fontSize = 13.sp)
                    }
                }
                Box(
                    modifier = Modifier.align(Alignment.TopEnd).padding(14.dp)
                        .size(38.dp).clip(CircleShape)
                        .background(if (saved) HubColors.Green.copy(alpha = 0.22f) else HubColors.BgEl)
                        .border(1.dp, if (saved) HubColors.Green.copy(alpha = 0.66f) else HubColors.Border, CircleShape)
                        .clickable { saved = !saved },
                    contentAlignment = Alignment.Center,
                ) {
                    Text("\uD83D\uDD16", fontSize = 16.sp)
                }
            }

            // ── Scrollable content ────────────────────────────────────────
            Column(
                modifier = Modifier.weight(1f).verticalScroll(scrollState)
                    .padding(bottom = 80.dp),
            ) {
                NameBlock(uiState, col, tab, onTabChange = { tab = it })
                Spacer(Modifier.height(14.dp))
                when (tab) {
                    "info" -> InfoTab()
                    "audio" -> AudioTab()
                    "map" -> MapTab()
                }
            }
        }

        // ── Sticky footer ─────────────────────────────────────────────────
        Column(
            modifier = Modifier.align(Alignment.BottomCenter)
                .fillMaxWidth().background(HubColors.Bg),
        ) {
            HorizontalDivider(color = HubColors.Border)
            Row(
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FooterBtn(
                    text = stringResource(R.string.detail_ebird),
                    color = HubColors.Blue, modifier = Modifier.weight(1f),
                ) {
                    val uri = Uri.parse("https://ebird.org/search?q=${Uri.encode(uiState.scientificName)}")
                    context.startActivity(Intent(Intent.ACTION_VIEW, uri))
                }
                FooterBtn(
                    text = stringResource(R.string.detail_inaturalist),
                    color = HubColors.Green, modifier = Modifier.weight(1f),
                ) {
                    val uri = Uri.parse("https://www.inaturalist.org/search?q=${Uri.encode(uiState.scientificName)}")
                    context.startActivity(Intent(Intent.ACTION_VIEW, uri))
                }
                FooterBtn(
                    text = if (saved) stringResource(R.string.detail_saved) else stringResource(R.string.detail_add_to_list),
                    color = if (saved) HubColors.Green else HubColors.Bg,
                    bgColor = if (saved) HubColors.Green.copy(alpha = 0.18f) else HubColors.Accent,
                    borderColor = if (saved) HubColors.Green.copy(alpha = 0.44f) else Color.Transparent,
                    modifier = Modifier.weight(1f),
                ) { saved = !saved }
            }
        }
    }
}

@Composable
private fun NameBlock(
    uiState: DetailUiState,
    col: Color,
    tab: String,
    onTabChange: (String) -> Unit,
) {
    Column(modifier = Modifier.fillMaxWidth().background(HubColors.Bg).padding(horizontal = 20.dp)) {
        Spacer(Modifier.height(14.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = uiState.commonName.ifBlank { "\u2014" },
                    color = HubColors.TextPrimary, fontSize = 24.sp,
                    fontWeight = FontWeight.ExtraBold, lineHeight = 28.sp,
                )
                Text(
                    text = uiState.scientificName,
                    color = HubColors.TextMuted, fontSize = 13.sp, fontStyle = FontStyle.Italic,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
            Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(5.dp)) {
                StatusTag(stringResource(R.string.detail_placeholder_status), HubColors.Green)
                StatusTag(stringResource(R.string.detail_placeholder_rarity), col)
            }
        }
        if (uiState.confidence > 0) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp)
                    .background(HubColors.BgCard, RoundedCornerShape(12.dp))
                    .border(1.dp, col.copy(alpha = 0.33f), RoundedCornerShape(12.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text(
                    text = stringResource(R.string.detail_accuracy_label),
                    color = HubColors.TextMuted, fontSize = 11.sp,
                )
                ConfBar(uiState.confidence / 100f, modifier = Modifier.weight(1f))
                Text(
                    text = "${uiState.confidence}%",
                    color = col, fontWeight = FontWeight.ExtraBold, fontSize = 14.sp,
                )
            }
        }
        Row(
            modifier = Modifier.fillMaxWidth().padding(top = 14.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            listOf("info" to R.string.detail_tab_info, "audio" to R.string.detail_tab_audio, "map" to R.string.detail_tab_map)
                .forEach { (id, strId) ->
                    Box(
                        modifier = Modifier.weight(1f).clip(RoundedCornerShape(12.dp))
                            .background(if (tab == id) HubColors.Accent else HubColors.BgCard)
                            .clickable { onTabChange(id) }
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = stringResource(strId),
                            color = if (tab == id) HubColors.Bg else HubColors.TextSecondary,
                            fontSize = 12.sp,
                            fontWeight = if (tab == id) FontWeight.Bold else FontWeight.Normal,
                        )
                    }
                }
        }
    }
}

@Composable
private fun InfoTab() {
    Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)) {
        Box(
            modifier = Modifier.fillMaxWidth()
                .background(HubColors.BgCard, RoundedCornerShape(16.dp))
                .border(1.dp, HubColors.Border, RoundedCornerShape(16.dp))
                .padding(16.dp),
        ) {
            Text(
                text = stringResource(R.string.detail_data_soon),
                color = HubColors.TextSecondary, fontSize = 14.sp, lineHeight = 24.sp,
            )
        }
    }
}

@Composable
private fun AudioTab() {
    Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)) {
        Box(
            modifier = Modifier.fillMaxWidth()
                .background(HubColors.BgCard, RoundedCornerShape(16.dp))
                .border(1.dp, HubColors.Border, RoundedCornerShape(16.dp))
                .padding(16.dp),
        ) {
            Text(
                text = stringResource(R.string.detail_audio_soon),
                color = HubColors.TextSecondary, fontSize = 14.sp, lineHeight = 24.sp,
            )
        }
    }
}

@Composable
private fun MapTab() {
    Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 14.dp)) {
        Box(
            modifier = Modifier.fillMaxWidth()
                .background(HubColors.BgCard, RoundedCornerShape(16.dp))
                .border(1.dp, HubColors.Border, RoundedCornerShape(16.dp))
                .padding(16.dp),
        ) {
            Text(
                text = stringResource(R.string.detail_map_soon),
                color = HubColors.TextSecondary, fontSize = 14.sp, lineHeight = 24.sp,
            )
        }
    }
}

@Composable
private fun StatusTag(label: String, color: Color) {
    Box(
        modifier = Modifier.clip(RoundedCornerShape(20.dp))
            .background(color.copy(alpha = 0.18f))
            .border(1.dp, color.copy(alpha = 0.33f), RoundedCornerShape(20.dp))
            .padding(horizontal = 8.dp, vertical = 3.dp),
    ) {
        Text(text = label, color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.3.sp)
    }
}

@Composable
private fun FooterBtn(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    bgColor: Color = HubColors.BgEl,
    borderColor: Color = HubColors.Border,
    onClick: () -> Unit,
) {
    Box(
        modifier = modifier.clip(RoundedCornerShape(12.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(vertical = 11.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text = text, color = color, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    }
}

// ── Previews ──────────────────────────────────────────────────────────────────

@Preview(showBackground = true, showSystemUi = true, name = "Detail - Info")
@Composable
private fun PreviewDetail() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        DetailScreen(
            uiState = DetailUiState(
                commonName = "Иволга",
                scientificName = "Oriolus oriolus",
                confidence = 94,
            ),
        )
    }
}
