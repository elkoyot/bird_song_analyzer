package com.birdsong.analyzer.presentation.location

import androidx.activity.compose.BackHandler
import androidx.annotation.DrawableRes
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.birdsong.analyzer.R
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import com.birdsong.analyzer.presentation.theme.ConfidenceHigh

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocationPickerScreen(
    uiState: LocationPickerUiState,
    onSelectContinent: (String, String) -> Unit = { _, _ -> },
    onSelectCountry: (GeoEntity) -> Unit = {},
    onSelectRegion: (String?) -> Unit = {},
    onBack: () -> Unit = {},
    onGoBack: () -> Boolean = { false },
) {
    BackHandler {
        if (!onGoBack()) onBack()
    }

    when (uiState.step) {
        LocationStep.CONTINENTS -> ContinentGrid(
            continents = uiState.continents,
            onSelect = onSelectContinent,
            onBack = onBack,
        )
        LocationStep.COUNTRIES -> CountryList(
            countries = uiState.countries,
            title = uiState.selectedContinentName,
            currentCountryCode = uiState.currentCountryCode,
            onSelect = onSelectCountry,
            onBack = { if (!onGoBack()) onBack() },
        )
        LocationStep.REGIONS -> RegionList(
            regions = uiState.regions,
            countryName = uiState.selectedCountryName,
            countryCode = uiState.selectedCountryCode,
            currentRegionCode = uiState.currentRegionCode,
            currentCountryCode = uiState.currentCountryCode,
            onSelect = onSelectRegion,
            onBack = { if (!onGoBack()) onBack() },
        )
    }
}

@DrawableRes
private fun continentIconRes(code: String): Int = when (code) {
    "EU" -> R.drawable.ic_continent_europe
    "AS" -> R.drawable.ic_continent_asia
    "AF" -> R.drawable.ic_continent_africa
    "NA" -> R.drawable.ic_continent_north_america
    "SA" -> R.drawable.ic_continent_south_america
    "OC" -> R.drawable.ic_continent_oceania
    else -> R.drawable.ic_continent_europe
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ContinentGrid(
    continents: List<GeoEntity>,
    onSelect: (String, String) -> Unit,
    onBack: () -> Unit,
) {
    val colors = listOf(
        MaterialTheme.colorScheme.primaryContainer,
        MaterialTheme.colorScheme.secondaryContainer,
        MaterialTheme.colorScheme.tertiaryContainer,
    )
    val onColors = listOf(
        MaterialTheme.colorScheme.onPrimaryContainer,
        MaterialTheme.colorScheme.onSecondaryContainer,
        MaterialTheme.colorScheme.onTertiaryContainer,
    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.location_picker_title)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null)
                    }
                },
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                text = stringResource(R.string.location_select_continent),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(bottom = 12.dp),
            )

            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxSize(),
            ) {
                items(continents, key = { it.code }) { continent ->
                    val idx = continents.indexOf(continent)
                    ElevatedCard(
                        onClick = { onSelect(continent.code, continent.displayName()) },
                        modifier = Modifier.aspectRatio(1f),
                        colors = CardDefaults.elevatedCardColors(
                            containerColor = colors[idx % colors.size],
                        ),
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center,
                        ) {
                            Icon(
                                painter = painterResource(continentIconRes(continent.code)),
                                contentDescription = null,
                                modifier = Modifier.size(48.dp),
                                tint = onColors[idx % onColors.size],
                            )
                            Spacer(modifier = Modifier.size(8.dp))
                            Text(
                                text = continent.displayName(),
                                style = MaterialTheme.typography.titleMedium,
                                color = onColors[idx % onColors.size],
                            )
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CountryList(
    countries: List<GeoEntity>,
    title: String,
    currentCountryCode: String?,
    onSelect: (GeoEntity) -> Unit,
    onBack: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(title) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null)
                    }
                },
            )
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            items(countries, key = { it.code }) { country ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelect(country) }
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = country.displayName(),
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f),
                    )
                    if (country.code == currentCountryCode) {
                        Icon(
                            Icons.Default.Check,
                            contentDescription = null,
                            tint = ConfidenceHigh,
                        )
                    }
                }
                HorizontalDivider()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RegionList(
    regions: List<GeoEntity>,
    countryName: String,
    countryCode: String?,
    currentRegionCode: String?,
    currentCountryCode: String?,
    onSelect: (String?) -> Unit,
    onBack: () -> Unit,
) {
    val isWholeCountrySelected = countryCode == currentCountryCode && currentRegionCode == null

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(countryName) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null)
                    }
                },
            )
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
        ) {
            // "Whole country" option
            item(key = "__whole__") {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelect(null) }
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = stringResource(R.string.location_all_country, countryName),
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f),
                    )
                    if (isWholeCountrySelected) {
                        Icon(
                            Icons.Default.Check,
                            contentDescription = null,
                            tint = ConfidenceHigh,
                        )
                    }
                }
                HorizontalDivider()
            }

            items(regions, key = { it.code }) { region ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelect(region.code) }
                        .padding(horizontal = 16.dp, vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = region.displayName(),
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.weight(1f),
                    )
                    if (region.code == currentRegionCode) {
                        Icon(
                            Icons.Default.Check,
                            contentDescription = null,
                            tint = ConfidenceHigh,
                        )
                    }
                }
                HorizontalDivider()
            }
        }
    }
}

// --- Previews ---

private val previewContinents = listOf(
    GeoEntity("EU", "continent", null, "Европа", "Europe", null, null, null, null, sortOrder = 0),
    GeoEntity("AS", "continent", null, "Азия", "Asia", null, null, null, null, sortOrder = 1),
    GeoEntity("AF", "continent", null, "Африка", "Africa", null, null, null, null, sortOrder = 2),
    GeoEntity("NA", "continent", null, "Северная Америка", "North America", null, null, null, null, sortOrder = 3),
    GeoEntity("SA", "continent", null, "Южная Америка", "South America", null, null, null, null, sortOrder = 4),
    GeoEntity("OC", "continent", null, "Океания", "Oceania", null, null, null, null, sortOrder = 5),
)

private val previewCountries = listOf(
    GeoEntity("BY", "country", "EU", "Беларусь", "Belarus", 51.2f, 56.2f, 23.2f, 32.8f),
    GeoEntity("RU", "country", "EU", "Россия", "Russia", 41.0f, 77.0f, 27.0f, 169.0f, bufferDeg = 3.0f),
    GeoEntity("UA", "country", "EU", "Украина", "Ukraine", 44.4f, 52.4f, 22.1f, 40.2f),
)

private val previewRegions = listOf(
    GeoEntity("RU-NW", "region", "RU", "Северо-Западный", "Northwestern Russia", 56.5f, 70.0f, 26.0f, 60.0f),
    GeoEntity("RU-C", "region", "RU", "Центральный", "Central Russia", 50.0f, 60.0f, 30.0f, 50.0f),
    GeoEntity("RU-S", "region", "RU", "Южный", "Southern Russia", 41.0f, 52.0f, 37.0f, 50.0f),
)

@Preview(showBackground = true, showSystemUi = true, name = "Location — Continents")
@Composable
private fun PreviewContinents() {
    BirdSongTheme {
        LocationPickerScreen(
            uiState = LocationPickerUiState(
                step = LocationStep.CONTINENTS,
                continents = previewContinents,
                currentCountryCode = "BY",
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Location — Countries")
@Composable
private fun PreviewCountries() {
    BirdSongTheme {
        LocationPickerScreen(
            uiState = LocationPickerUiState(
                step = LocationStep.COUNTRIES,
                countries = previewCountries,
                selectedContinentName = "Европа",
                currentCountryCode = "BY",
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Location — Regions")
@Composable
private fun PreviewRegions() {
    BirdSongTheme {
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
}

@Preview(showBackground = true, showSystemUi = true, name = "Location — Continents Dark")
@Composable
private fun PreviewContinentsDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
        LocationPickerScreen(
            uiState = LocationPickerUiState(
                step = LocationStep.CONTINENTS,
                continents = previewContinents,
            ),
        )
    }
}

@Preview(showBackground = true, showSystemUi = true, name = "Location — Regions Dark")
@Composable
private fun PreviewRegionsDark() {
    BirdSongTheme(darkTheme = true, dynamicColor = false) {
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
}
