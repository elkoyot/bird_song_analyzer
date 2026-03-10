package com.birdsong.analyzer.presentation.detection

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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.presentation.theme.HubColors

@Composable
fun FileAnalysisScreen(
    coreState: FileAnalysisCoreState = FileAnalysisCoreState(),
    progressState: AnalysisProgressState = AnalysisProgressState(),
    spectrogramState: SpectrogramUiState = SpectrogramUiState(),
    timelineState: TimelineUiState = TimelineUiState(),
    playbackState: FilePlaybackUiState = FilePlaybackUiState(),
    onSelectFile: () -> Unit = {},
    onStartAnalysis: () -> Unit = {},
    onPause: () -> Unit = {},
    onResume: () -> Unit = {},
    onStop: () -> Unit = {},
    onTogglePlayback: () -> Unit = {},
    onSeekPlayback: (Float) -> Unit = {},
    onHighlightSpecies: (String?) -> Unit = {},
    onSpeciesClick: (scientificName: String, commonName: String) -> Unit = { _, _ -> },
    onSave: () -> Unit = {},
    onDiscard: () -> Unit = {},
    onResetFile: () -> Unit = {},
    onPickLocation: () -> Unit = {},
    onHistory: () -> Unit = {},
    onBack: () -> Unit = {},
) {
    val phase = coreState.phase
    val isAnalyzing = phase == FileAnalysisPhase.ANALYZING
    val isPaused = phase == FileAnalysisPhase.PAUSED
    val isDone = phase == FileAnalysisPhase.DONE
    val isActive = isAnalyzing || isPaused

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(HubColors.Bg),
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Header ──
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Back button
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .background(HubColors.BgEl)
                            .border(1.dp, HubColors.Border, RoundedCornerShape(12.dp))
                            .clickable(onClick = onBack)
                            .padding(horizontal = 14.dp, vertical = 8.dp),
                    ) {
                        Text(
                            "\u2190 \u041D\u0430\u0437\u0430\u0434",
                            color = HubColors.TextSecondary,
                            fontSize = 13.sp,
                        )
                    }
                    Spacer(Modifier.width(8.dp))
                    Column {
                        Text(
                            "\u0410\u043D\u0430\u043B\u0438\u0437 \u0444\u0430\u0439\u043B\u0430",
                            color = HubColors.TextMuted,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.SemiBold,
                            letterSpacing = 1.2.sp,
                        )
                        Text(
                            "AVALGA",
                            color = HubColors.TextPrimary,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.ExtraBold,
                        )
                    }
                }
                // Region button
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(HubColors.BgEl)
                        .border(
                            1.dp,
                            if (coreState.geoConfigured) HubColors.Border
                            else HubColors.Accent.copy(alpha = 0.4f),
                            RoundedCornerShape(20.dp),
                        )
                        .clickable(onClick = onPickLocation)
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                ) {
                    Text(
                        text = "\uD83D\uDCCD ${if (coreState.geoConfigured) coreState.geoLabel else "\u0412\u044B\u0431\u0440\u0430\u0442\u044C \u0440\u0435\u0433\u0438\u043E\u043D"}",
                        color = if (coreState.geoConfigured) HubColors.TextSecondary else HubColors.Accent,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }

            // ── Content ──
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 18.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                // IDLE: DropZone
                if (phase == FileAnalysisPhase.IDLE) {
                    item(key = "dropzone") {
                        DropZone(onPick = onSelectFile)
                    }
                }

                // File card (READY / ANALYZING / PAUSED / DONE)
                if (phase != FileAnalysisPhase.IDLE && phase != FileAnalysisPhase.ERROR) {
                    item(key = "filecard") {
                        FileCard(
                            fileName = coreState.fileName,
                            fileSize = coreState.fileSizeLabel,
                            phase = phase,
                            progress = progressState.progress,
                            elapsedSec = progressState.elapsedSec,
                            speciesCount = timelineState.speciesSummaries.size,
                            onClose = if (phase == FileAnalysisPhase.READY || isDone) {
                                { onResetFile() }
                            } else null,
                        )
                    }
                }

                // Start button (READY)
                if (phase == FileAnalysisPhase.READY) {
                    item(key = "start") {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(14.dp))
                                .background(
                                    Brush.linearGradient(
                                        listOf(HubColors.Accent, HubColors.Accent.copy(alpha = 0.8f)),
                                    ),
                                )
                                .clickable(onClick = onStartAnalysis)
                                .padding(vertical = 14.dp),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text(
                                text = if (coreState.geoConfigured) "\u25B6 \u041D\u0430\u0447\u0430\u0442\u044C \u0430\u043D\u0430\u043B\u0438\u0437"
                                       else "\uD83D\uDCCD \u0421\u043D\u0430\u0447\u0430\u043B\u0430 \u0432\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0433\u0438\u043E\u043D",
                                color = Color.Black,
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp,
                            )
                        }
                    }
                }

                // Spectrogram (ANALYZING / PAUSED / DONE)
                if (isActive || isDone) {
                    item(key = "spectrogram") {
                        SpectrogramView(
                            columns = spectrogramState.columns,
                            markers = if (isDone) spectrogramState.birdMarkers else emptyList(),
                            highlightedSpecies = spectrogramState.highlightedSpecies,
                            playhead = playbackState.position.takeIf { isActive || isDone },
                            isAnalyzing = isAnalyzing,
                            onSeek = if (isDone) onSeekPlayback else null,
                        )
                    }
                }

                // Control buttons (ANALYZING / PAUSED)
                if (isActive) {
                    item(key = "controls") {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            if (isAnalyzing) {
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(HubColors.Yellow.copy(alpha = 0.1f))
                                        .border(1.dp, HubColors.Yellow.copy(alpha = 0.27f), RoundedCornerShape(12.dp))
                                        .clickable(onClick = onPause)
                                        .padding(vertical = 11.dp),
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Text(
                                        "\u23F8 \u041F\u0430\u0443\u0437\u0430",
                                        color = HubColors.Yellow,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 13.sp,
                                    )
                                }
                            }
                            if (isPaused) {
                                Box(
                                    modifier = Modifier
                                        .weight(2f)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(
                                            Brush.linearGradient(
                                                listOf(HubColors.Accent, HubColors.Accent.copy(alpha = 0.8f)),
                                            ),
                                        )
                                        .clickable(onClick = onResume)
                                        .padding(vertical = 11.dp),
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Text(
                                        "\u25B6 \u041F\u0440\u043E\u0434\u043E\u043B\u0436\u0438\u0442\u044C",
                                        color = Color.Black,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 13.sp,
                                    )
                                }
                            }
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(HubColors.Red.copy(alpha = 0.1f))
                                    .border(1.dp, HubColors.Red.copy(alpha = 0.27f), RoundedCornerShape(12.dp))
                                    .clickable(onClick = onStop)
                                    .padding(vertical = 11.dp),
                                contentAlignment = Alignment.Center,
                            ) {
                                Text(
                                    "\u23F9 \u0421\u0442\u043E\u043F",
                                    color = HubColors.Red,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 13.sp,
                                )
                            }
                        }
                    }
                }

                // Playback controls (DONE)
                if (isDone) {
                    item(key = "playback") {
                        PlaybackControls(
                            isPlaying = playbackState.isPlaying,
                            position = playbackState.position,
                            positionLabel = playbackState.positionLabel,
                            durationLabel = coreState.fileDurationLabel,
                            onToggle = onTogglePlayback,
                            onSeek = onSeekPlayback,
                        )
                    }
                }

                // Bird count label
                if (isActive && timelineState.speciesSummaries.isNotEmpty()) {
                    item(key = "count_active") {
                        Row {
                            Text(
                                "\u041D\u0430\u0439\u0434\u0435\u043D\u043E: ",
                                color = HubColors.TextMuted,
                                fontSize = 11.sp,
                            )
                            Text(
                                "${timelineState.speciesSummaries.size}",
                                color = HubColors.Green,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                            )
                        }
                    }
                }
                if (isDone && timelineState.speciesSummaries.isNotEmpty()) {
                    item(key = "count_done") {
                        Row {
                            Text(
                                "\u041E\u043F\u0440\u0435\u0434\u0435\u043B\u0435\u043D\u043E: ",
                                color = HubColors.TextMuted,
                                fontSize = 11.sp,
                            )
                            Text(
                                "${timelineState.speciesSummaries.size}",
                                color = HubColors.Green,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                " \u00b7 \u0442\u0430\u043F \u2014 \u043E\u0442\u043C\u0435\u0442\u043A\u0430 \u043D\u0430 \u0441\u043F\u0435\u043A\u0442\u0440\u0435",
                                color = HubColors.TextMuted,
                                fontSize = 11.sp,
                            )
                        }
                    }
                }

                // Searching placeholder
                if (isActive && timelineState.speciesSummaries.isEmpty()) {
                    item(key = "searching") {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 12.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                        ) {
                            Text("\uD83D\uDD0D", fontSize = 28.sp)
                            Spacer(Modifier.height(6.dp))
                            Text(
                                "\u0418\u0449\u0443 \u043F\u0442\u0438\u0446 \u0432 \u0430\u0443\u0434\u0438\u043E...",
                                color = HubColors.TextMuted,
                                fontSize = 13.sp,
                            )
                        }
                    }
                }

                // Bird result items
                if (timelineState.speciesSummaries.isNotEmpty()) {
                    val birds = timelineState.timelineBirds
                    val uniqueSpecies = timelineState.speciesSummaries.map { summary ->
                        val representative = birds.firstOrNull { it.scientificName == summary.scientificName }
                        representative ?: FileTimelineBirdUi(
                            id = summary.scientificName,
                            commonName = summary.commonName,
                            scientificName = summary.scientificName,
                            startTimeSec = 0f,
                            endTimeSec = 0f,
                            timeRange = "",
                            v24Confidence = summary.maxV24Confidence,
                            v30Confidence = summary.maxV30Confidence,
                        )
                    }
                    items(uniqueSpecies, key = { it.scientificName }) { bird ->
                        FileBirdResultItem(
                            bird = bird,
                            isHighlighted = bird.scientificName == spectrogramState.highlightedSpecies,
                            isDone = isDone,
                            onClick = {
                                if (isDone) {
                                    onHighlightSpecies(bird.scientificName)
                                } else {
                                    onSpeciesClick(bird.scientificName, bird.commonName)
                                }
                            },
                        )
                    }
                }

                // Error
                if (phase == FileAnalysisPhase.ERROR) {
                    item(key = "error") {
                        Text(
                            text = "\u041E\u0448\u0438\u0431\u043A\u0430: ${coreState.errorMessage}",
                            color = HubColors.Red,
                            fontSize = 13.sp,
                            modifier = Modifier.padding(vertical = 16.dp),
                        )
                    }
                }

                // Bottom spacer for footer
                if (isDone) {
                    item(key = "footer_spacer") {
                        Spacer(Modifier.height(72.dp))
                    }
                }
            }
        }

        // ── FAB History (IDLE only) ──
        if (phase == FileAnalysisPhase.IDLE) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 18.dp, bottom = 16.dp)
                    .size(52.dp)
                    .clip(CircleShape)
                    .background(HubColors.BgCard)
                    .border(1.5.dp, HubColors.Border, CircleShape)
                    .clickable(onClick = onHistory),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("\uD83D\uDDC2", fontSize = 16.sp)
                    Text(
                        "\u0438\u0441\u0442\u043E\u0440\u0438\u044F",
                        color = HubColors.TextMuted,
                        fontSize = 7.sp,
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = 0.3.sp,
                    )
                }
            }
        }

        // ── Done Footer (Save / Cancel) ──
        if (isDone) {
            Row(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
                    .background(HubColors.Bg)
                    .border(
                        width = 1.dp,
                        color = HubColors.Border,
                        shape = RoundedCornerShape(topStart = 0.dp, topEnd = 0.dp),
                    )
                    .padding(horizontal = 18.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Box(
                    modifier = Modifier
                        .weight(2f)
                        .clip(RoundedCornerShape(14.dp))
                        .background(
                            Brush.linearGradient(
                                listOf(HubColors.Accent, HubColors.Accent.copy(alpha = 0.8f)),
                            ),
                        )
                        .clickable(onClick = onSave)
                        .padding(vertical = 13.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        "\uD83D\uDCBE \u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C",
                        color = Color.Black,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                    )
                }
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(14.dp))
                        .background(HubColors.BgEl)
                        .border(1.dp, HubColors.Border, RoundedCornerShape(14.dp))
                        .clickable(onClick = onDiscard)
                        .padding(vertical = 13.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        "\u041E\u0442\u043C\u0435\u043D\u0430",
                        color = HubColors.TextSecondary,
                        fontWeight = FontWeight.SemiBold,
                        fontSize = 13.sp,
                    )
                }
            }
        }
    }
}

// ── Previews ─────────────────────────────────────────────────────────────────

@Preview(showBackground = true, showSystemUi = true, name = "Idle")
@Composable
private fun PreviewIdle() {
    FileAnalysisScreen(
        coreState = FileAnalysisCoreState(
            phase = FileAnalysisPhase.IDLE,
            geoLabel = "Belarus \u00b7 Minsk",
            geoConfigured = true,
        ),
    )
}

@Preview(showBackground = true, showSystemUi = true, name = "Ready")
@Composable
private fun PreviewReady() {
    FileAnalysisScreen(
        coreState = FileAnalysisCoreState(
            phase = FileAnalysisPhase.READY,
            fileName = "morning_forest.mp3",
            fileSizeLabel = "4.2 MB",
            fileDurationLabel = "4:12",
            geoLabel = "Belarus \u00b7 Minsk",
            geoConfigured = true,
        ),
    )
}

@Preview(showBackground = true, showSystemUi = true, name = "Analyzing")
@Composable
private fun PreviewAnalyzing() {
    FileAnalysisScreen(
        coreState = FileAnalysisCoreState(
            phase = FileAnalysisPhase.ANALYZING,
            fileName = "morning_forest.mp3",
            fileSizeLabel = "4.2 MB",
            fileDurationLabel = "4:12",
            geoLabel = "Belarus \u00b7 Minsk",
            geoConfigured = true,
        ),
        progressState = AnalysisProgressState(
            progress = 0.45f,
            elapsedSec = 23,
        ),
        timelineState = TimelineUiState(
            speciesSummaries = listOf(
                FileSpeciesSummary("Oriolus oriolus", "\u0418\u0432\u043E\u043B\u0433\u0430", 94, 78, 2, emptyList()),
            ),
            timelineBirds = listOf(
                FileTimelineBirdUi("1", "\u0418\u0432\u043E\u043B\u0433\u0430", "Oriolus oriolus", 5f, 8f, "0:05 \u2013 0:08", 94, 78),
            ),
        ),
    )
}

@Preview(showBackground = true, showSystemUi = true, name = "Done")
@Composable
private fun PreviewDone() {
    FileAnalysisScreen(
        coreState = FileAnalysisCoreState(
            phase = FileAnalysisPhase.DONE,
            fileName = "morning_forest.mp3",
            fileSizeLabel = "4.2 MB",
            fileDurationLabel = "4:12",
            fileDurationSec = 252f,
            geoLabel = "Belarus \u00b7 Minsk",
            geoConfigured = true,
        ),
        progressState = AnalysisProgressState(progress = 1f),
        spectrogramState = SpectrogramUiState(
            birdMarkers = listOf(
                BirdMarker("Oriolus oriolus", 0.12f, 0.94f),
                BirdMarker("Fringilla coelebs", 0.31f, 0.78f),
                BirdMarker("Parus major", 0.53f, 0.61f),
            ),
        ),
        timelineState = TimelineUiState(
            speciesSummaries = listOf(
                FileSpeciesSummary("Oriolus oriolus", "\u0418\u0432\u043E\u043B\u0433\u0430", 94, 78, 2, emptyList()),
                FileSpeciesSummary("Fringilla coelebs", "\u0417\u044F\u0431\u043B\u0438\u043A", 78, null, 1, emptyList()),
                FileSpeciesSummary("Parus major", "\u0411\u043E\u043B\u044C\u0448\u0430\u044F \u0441\u0438\u043D\u0438\u0446\u0430", 61, 55, 3, emptyList()),
            ),
            timelineBirds = listOf(
                FileTimelineBirdUi("1", "\u0418\u0432\u043E\u043B\u0433\u0430", "Oriolus oriolus", 5f, 8f, "0:05 \u2013 0:08", 94, 78),
                FileTimelineBirdUi("2", "\u0417\u044F\u0431\u043B\u0438\u043A", "Fringilla coelebs", 78f, 81f, "1:18 \u2013 1:21", 78, null),
                FileTimelineBirdUi("3", "\u0411\u043E\u043B\u044C\u0448\u0430\u044F \u0441\u0438\u043D\u0438\u0446\u0430", "Parus major", 130f, 135f, "2:10 \u2013 2:15", 61, 55),
            ),
        ),
    )
}
