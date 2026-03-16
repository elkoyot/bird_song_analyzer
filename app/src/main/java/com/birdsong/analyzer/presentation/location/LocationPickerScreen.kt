package com.birdsong.analyzer.presentation.location

import androidx.activity.compose.BackHandler
import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.presentation.theme.HubColors

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Converts ISO 3166-1 alpha-2 country code to flag emoji via regional indicators. */
private fun countryCodeToFlag(code: String): String = buildString {
    for (letter in code.uppercase()) {
        val codePoint = letter.code + 0x1F1A5
        appendCodePoint(codePoint)
    }
}

@DrawableRes
private fun continentIconRes(code: String): Int = when (code) {
    "EUR" -> R.drawable.ic_continent_europe
    "ASI" -> R.drawable.ic_continent_asia
    "AFR" -> R.drawable.ic_continent_africa
    "NAM" -> R.drawable.ic_continent_north_america
    "SAM" -> R.drawable.ic_continent_south_america
    "OCE" -> R.drawable.ic_continent_oceania
    else -> R.drawable.ic_continent_europe
}

private fun continentColor(code: String): Color = when (code) {
    "EUR" -> HubColors.Blue
    "ASI" -> HubColors.Purple
    "AFR" -> HubColors.Accent
    "NAM" -> HubColors.Green
    "SAM" -> HubColors.Green
    "OCE" -> HubColors.Blue
    else -> HubColors.Accent
}

// ── Root composable ──────────────────────────────────────────────────────────

@Composable
fun LocationPickerScreen(
    uiState: LocationPickerUiState,
    onSelectContinent: (String, String) -> Unit = { _, _ -> },
    onSelectCountry: (GeoEntity) -> Unit = {},
    onSelectRegion: (String?) -> Unit = {},
    onBack: () -> Unit = {},
    onGoBack: () -> Boolean = { false },
    onBreadcrumb: (LocationStep) -> Unit = {},
) {
    BackHandler {
        if (!onGoBack()) onBack()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(HubColors.Bg),
    ) {
        // Header
        LocationHeader(
            onBack = { if (!onGoBack()) onBack() },
        )

        // Breadcrumb
        if (uiState.breadcrumbs.size > 1) {
            Breadcrumb(
                crumbs = uiState.breadcrumbs,
                onCrumb = onBreadcrumb,
            )
        }

        // Level label
        val levelLabel = when (uiState.step) {
            LocationStep.CONTINENTS -> "ВЫБЕРИТЕ КОНТИНЕНТ"
            LocationStep.COUNTRIES -> "СТРАНЫ — ${uiState.selectedContinentName.uppercase()}"
            LocationStep.REGIONS -> "РЕГИОНЫ — ${uiState.selectedCountryName.uppercase()}"
        }
        Text(
            text = levelLabel,
            color = HubColors.TextMuted,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.4.sp,
            modifier = Modifier.padding(start = 22.dp, top = 8.dp, bottom = 4.dp),
        )

        // Content
        when (uiState.step) {
            LocationStep.CONTINENTS -> ContinentGrid(
                continents = uiState.continents,
                counts = uiState.continentCounts,
                onSelect = onSelectContinent,
            )
            LocationStep.COUNTRIES -> CountryList(
                countries = uiState.countries,
                regionCounts = uiState.countriesRegionCounts,
                currentCountryCode = uiState.currentCountryCode,
                onSelect = onSelectCountry,
            )
            LocationStep.REGIONS -> RegionList(
                regions = uiState.regions,
                countryName = uiState.selectedCountryName,
                countryCode = uiState.selectedCountryCode,
                currentRegionCode = uiState.currentRegionCode,
                currentCountryCode = uiState.currentCountryCode,
                onSelect = onSelectRegion,
            )
        }
    }
}

// ── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun LocationHeader(onBack: () -> Unit) {
    Column(modifier = Modifier.padding(start = 18.dp, end = 18.dp, top = 12.dp, bottom = 4.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // Back pill button
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(HubColors.BgEl)
                    .border(1.dp, HubColors.Border, RoundedCornerShape(12.dp))
                    .clickable(onClick = onBack)
                    .padding(horizontal = 14.dp, vertical = 8.dp),
            ) {
                Text(
                    text = "\u2190 Назад",
                    color = HubColors.TextSecondary,
                    fontSize = 13.sp,
                )
            }

            Column {
                Text(
                    text = "Выбор региона",
                    color = HubColors.TextPrimary,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.ExtraBold,
                )
                Text(
                    text = "Влияет на точность определения",
                    color = HubColors.TextMuted,
                    fontSize = 10.sp,
                )
            }
        }
    }
}

// ── Breadcrumb ───────────────────────────────────────────────────────────────

@Composable
private fun Breadcrumb(
    crumbs: List<BreadcrumbItem>,
    onCrumb: (LocationStep) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 18.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        crumbs.forEachIndexed { index, crumb ->
            if (index > 0) {
                Text(
                    text = "\u203A",
                    color = HubColors.TextMuted,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(horizontal = 4.dp),
                )
            }
            val isLast = index == crumbs.lastIndex
            val bgColor = if (isLast) HubColors.Accent.copy(alpha = 0.09f) else Color.Transparent
            val borderColor = if (isLast) HubColors.Accent.copy(alpha = 0.2f) else Color.Transparent
            val textColor = if (isLast) HubColors.Accent else HubColors.TextSecondary
            val weight = if (isLast) FontWeight.Bold else FontWeight.Normal

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(bgColor)
                    .then(
                        if (isLast) Modifier.border(1.dp, borderColor, RoundedCornerShape(20.dp))
                        else Modifier,
                    )
                    .clickable { onCrumb(crumb.step) }
                    .padding(horizontal = 10.dp, vertical = 4.dp),
            ) {
                Text(
                    text = crumb.label,
                    color = textColor,
                    fontSize = 12.sp,
                    fontWeight = weight,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

// ── Continent Grid ───────────────────────────────────────────────────────────

@Composable
private fun ContinentGrid(
    continents: List<GeoEntity>,
    counts: Map<String, Int>,
    onSelect: (String, String) -> Unit,
) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        contentPadding = PaddingValues(horizontal = 18.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        modifier = Modifier.fillMaxSize(),
    ) {
        items(continents, key = { it.code }) { continent ->
            val color = continentColor(continent.code)
            ContinentGridItem(
                continent = continent,
                color = color,
                count = counts[continent.code],
                onClick = { onSelect(continent.code, continent.displayName()) },
            )
        }
    }
}

@Composable
private fun ContinentGridItem(
    continent: GeoEntity,
    color: Color,
    count: Int?,
    onClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(18.dp))
            .background(HubColors.BgCard)
            .border(1.5.dp, HubColors.Border, RoundedCornerShape(18.dp))
            .clickable(onClick = onClick)
            .height(130.dp)
            .fillMaxWidth(),
    ) {
        // Top accent line
        Box(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth(0.7f)
                .height(1.5.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(Color.Transparent, color.copy(alpha = 0.4f), Color.Transparent),
                    ),
                ),
        )

        // Chevron
        Text(
            text = "\u203A",
            color = HubColors.TextMuted,
            fontSize = 10.sp,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 8.dp, end = 10.dp),
        )

        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(
                painter = painterResource(continentIconRes(continent.code)),
                contentDescription = null,
                modifier = Modifier.size(52.dp),
                tint = color.copy(alpha = 0.6f),
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = continent.displayName(),
                color = HubColors.TextPrimary,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
            )
            if (count != null && count > 0) {
                Spacer(modifier = Modifier.height(4.dp))
                Box(
                    modifier = Modifier
                        .background(color.copy(alpha = 0.09f), RoundedCornerShape(20.dp))
                        .padding(horizontal = 8.dp, vertical = 3.dp),
                ) {
                    Text(
                        text = "$count",
                        color = color,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

// ── Country List ─────────────────────────────────────────────────────────────

@Composable
private fun CountryList(
    countries: List<GeoEntity>,
    regionCounts: Map<String, Int>,
    currentCountryCode: String?,
    onSelect: (GeoEntity) -> Unit,
) {
    // Group by first letter
    val grouped = countries.groupBy { it.displayName().first().uppercaseChar() }
        .toSortedMap()

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 18.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        grouped.forEach { (letter, list) ->
            item(key = "header_$letter") {
                Text(
                    text = letter.toString(),
                    color = HubColors.TextMuted,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.5.sp,
                    modifier = Modifier.padding(start = 4.dp, top = 6.dp, bottom = 4.dp),
                )
            }
            items(list, key = { it.code }) { country ->
                val isSelected = country.code == currentCountryCode
                val regCount = regionCounts[country.code] ?: 0
                val hasChildren = regCount > 0
                CountryListItem(
                    flag = countryCodeToFlag(country.code),
                    name = country.displayName(),
                    sublabel = if (hasChildren) "$regCount регионов" else null,
                    isSelected = isSelected,
                    hasChildren = hasChildren,
                    onClick = { onSelect(country) },
                )
            }
        }
    }
}

@Composable
private fun CountryListItem(
    flag: String,
    name: String,
    sublabel: String?,
    isSelected: Boolean,
    hasChildren: Boolean,
    onClick: () -> Unit,
) {
    val color = HubColors.Accent
    val bgColor = if (isSelected) color.copy(alpha = 0.07f) else HubColors.BgCard
    val borderColor = if (isSelected) color.copy(alpha = 0.33f) else HubColors.Border

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(bgColor)
            .border(1.5.dp, borderColor, RoundedCornerShape(14.dp))
            .clickable(onClick = onClick),
    ) {
        // Selected top accent line
        if (isSelected) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth(0.8f)
                    .height(1.dp)
                    .background(
                        Brush.horizontalGradient(
                            listOf(Color.Transparent, color.copy(alpha = 0.4f), Color.Transparent),
                        ),
                    ),
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 13.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = flag,
                fontSize = 20.sp,
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = name,
                    color = if (isSelected) color else HubColors.TextPrimary,
                    fontSize = 14.sp,
                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                )
                if (sublabel != null) {
                    Text(
                        text = sublabel,
                        color = HubColors.TextMuted,
                        fontSize = 11.sp,
                    )
                }
            }
            if (isSelected) {
                Text(text = "\u2713", color = color, fontSize = 16.sp)
            } else if (hasChildren) {
                Text(text = "\u203A", color = HubColors.TextMuted, fontSize = 18.sp)
            }
        }
    }
}

// ── Region List ──────────────────────────────────────────────────────────────

@Composable
private fun RegionList(
    regions: List<GeoEntity>,
    countryName: String,
    countryCode: String?,
    currentRegionCode: String?,
    currentCountryCode: String?,
    onSelect: (String?) -> Unit,
) {
    val isWholeCountrySelected = countryCode == currentCountryCode && currentRegionCode == null

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 18.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // "Whole country" option
        item(key = "__whole__") {
            RegionListItem(
                label = "Вся страна ($countryName)",
                isSelected = isWholeCountrySelected,
                color = HubColors.Accent,
                onClick = { onSelect(null) },
            )
        }

        item(key = "__regions_header__") {
            Text(
                text = "РЕГИОНЫ",
                color = HubColors.TextMuted,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.4.sp,
                modifier = Modifier.padding(start = 4.dp, top = 8.dp, bottom = 2.dp),
            )
        }

        items(regions, key = { it.code }) { region ->
            val isSelected = region.code == currentRegionCode
            RegionListItem(
                label = region.displayName(),
                isSelected = isSelected,
                color = HubColors.Green,
                onClick = { onSelect(region.code) },
            )
        }
    }
}

@Composable
private fun RegionListItem(
    label: String,
    isSelected: Boolean,
    color: Color,
    onClick: () -> Unit,
) {
    val bgColor = if (isSelected) color.copy(alpha = 0.07f) else HubColors.BgCard
    val borderColor = if (isSelected) color.copy(alpha = 0.33f) else HubColors.Border

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(bgColor)
            .border(1.5.dp, borderColor, RoundedCornerShape(14.dp))
            .clickable(onClick = onClick),
    ) {
        if (isSelected) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .fillMaxWidth(0.8f)
                    .height(1.dp)
                    .background(
                        Brush.horizontalGradient(
                            listOf(Color.Transparent, color.copy(alpha = 0.4f), Color.Transparent),
                        ),
                    ),
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 13.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = label,
                color = if (isSelected) color else HubColors.TextPrimary,
                fontSize = 14.sp,
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                modifier = Modifier.weight(1f),
            )
            if (isSelected) {
                Text(text = "\u2713", color = color, fontSize = 16.sp)
            }
        }
    }
}

// ── Previews ─────────────────────────────────────────────────────────────────

private val previewContinents = listOf(
    GeoEntity("EUR", "continent", null, "Европа", "Europe", null, null, null, null, sortOrder = 0),
    GeoEntity("ASI", "continent", null, "Азия", "Asia", null, null, null, null, sortOrder = 1),
    GeoEntity("AFR", "continent", null, "Африка", "Africa", null, null, null, null, sortOrder = 2),
    GeoEntity("NAM", "continent", null, "Северная Америка", "North America", null, null, null, null, sortOrder = 3),
    GeoEntity("SAM", "continent", null, "Южная Америка", "South America", null, null, null, null, sortOrder = 4),
    GeoEntity("OCE", "continent", null, "Океания", "Oceania", null, null, null, null, sortOrder = 5),
)

private val previewCountries = listOf(
    GeoEntity("BY", "country", "EUR", "BY", "BY", 51.2f, 56.2f, 23.2f, 32.8f),
    GeoEntity("DE", "country", "EUR", "DE", "DE", 47.3f, 55.1f, 5.9f, 15.0f),
    GeoEntity("PL", "country", "EUR", "PL", "PL", 49.0f, 54.8f, 14.1f, 24.1f),
    GeoEntity("RU", "country", "EUR", "RU", "RU", 41.0f, 77.0f, 27.0f, 169.0f, bufferDeg = 3.0f),
    GeoEntity("UA", "country", "EUR", "UA", "UA", 44.4f, 52.4f, 22.1f, 40.2f),
    GeoEntity("FI", "country", "EUR", "FI", "FI", 59.8f, 70.1f, 20.6f, 31.6f),
)

private val previewRegions = listOf(
    GeoEntity("RU-NW", "region", "RU", "Северо-Западный", "Northwestern Russia", 56.5f, 70.0f, 26.0f, 60.0f),
    GeoEntity("RU-C", "region", "RU", "Центральный", "Central Russia", 50.0f, 60.0f, 30.0f, 50.0f),
    GeoEntity("RU-S", "region", "RU", "Южный", "Southern Russia", 41.0f, 52.0f, 37.0f, 50.0f),
)

@Preview(showBackground = true, widthDp = 390, heightDp = 844, name = "SCR-10 Continents")
@Composable
private fun PreviewContinents() {
    LocationPickerScreen(
        uiState = LocationPickerUiState(
            step = LocationStep.CONTINENTS,
            continents = previewContinents,
            continentCounts = mapOf("EUR" to 45, "ASI" to 48, "AFR" to 57, "NAM" to 34, "SAM" to 14, "OCE" to 28),
            currentCountryCode = "BY",
        ),
    )
}

@Preview(showBackground = true, widthDp = 390, heightDp = 844, name = "SCR-10 Countries")
@Composable
private fun PreviewCountries() {
    LocationPickerScreen(
        uiState = LocationPickerUiState(
            step = LocationStep.COUNTRIES,
            countries = previewCountries,
            countriesRegionCounts = mapOf("BY" to 0, "RU" to 8, "UA" to 0, "DE" to 0, "PL" to 0, "FI" to 0),
            selectedContinentName = "Европа",
            currentCountryCode = "BY",
        ),
    )
}

@Preview(showBackground = true, widthDp = 390, heightDp = 844, name = "SCR-10 Regions")
@Composable
private fun PreviewRegions() {
    LocationPickerScreen(
        uiState = LocationPickerUiState(
            step = LocationStep.REGIONS,
            regions = previewRegions,
            selectedCountryName = "Россия",
            selectedCountryCode = "RU",
            currentCountryCode = "RU",
            currentRegionCode = "RU-C",
        ),
    )
}

@Preview(showBackground = true, widthDp = 390, heightDp = 844, name = "SCR-10 Regions — Whole Country")
@Composable
private fun PreviewRegionsWholeCountry() {
    LocationPickerScreen(
        uiState = LocationPickerUiState(
            step = LocationStep.REGIONS,
            regions = previewRegions,
            selectedCountryName = "Россия",
            selectedCountryCode = "RU",
            currentCountryCode = "RU",
            currentRegionCode = null,
        ),
    )
}
