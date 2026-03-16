package com.birdsong.analyzer.presentation.theme

import androidx.compose.ui.graphics.Color

// ── Hub / App-wide dark palette (matches prototype) ──────────────────────────
object HubColors {
    val Bg         = Color(0xFF050C18)
    val BgCard     = Color(0xFF0D1926)
    val BgEl       = Color(0xFF0F1E32)
    val BgEl2      = Color(0xFF162842)
    val Accent     = Color(0xFFE8A020)
    val Green      = Color(0xFF3DBA7E)
    val Blue       = Color(0xFF4BA3C7)
    val Purple     = Color(0xFF9B7FE8)
    val Red        = Color(0xFFE05050)
    val RedHot     = Color(0xFFE8504A)
    val RedDark    = Color(0xFFC0392B)
    val Yellow     = Color(0xFFE8C020)
    val TextPrimary   = Color(0xFFEEF2F0)
    val TextSecondary = Color(0xFF7A9E94)
    val TextMuted     = Color(0xFF2E4A5C)
    val Border     = Color(0xFF13283E)
    val NavBg      = Color(0xFF060E1C)
    val NavBorder  = Color(0xFF0C1A2E)
}

// Nature-inspired palette
val Green40 = Color(0xFF2E7D32)
val Green80 = Color(0xFFA5D6A7)
val GreenGrey40 = Color(0xFF4E6B4E)
val GreenGrey80 = Color(0xFFB8CCB8)
val Brown40 = Color(0xFF5D4037)
val Brown80 = Color(0xFFBCAAA4)

// Confidence indicator colors
val ConfidenceHigh = Color(0xFF2E7D32)   // ≥80%
val ConfidenceMedium = Color(0xFFF9A825) // 50-79%
val ConfidenceLow = Color(0xFFC62828)    // <50%
