package com.birdsong.analyzer.presentation.location

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.repository.GeoRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.text.Collator
import javax.inject.Inject

enum class LocationStep { CONTINENTS, COUNTRIES, REGIONS }

data class BreadcrumbItem(val label: String, val step: LocationStep)

data class LocationPickerUiState(
    val step: LocationStep = LocationStep.CONTINENTS,
    val continents: List<GeoEntity> = emptyList(),
    val continentCounts: Map<String, Int> = emptyMap(),
    val countries: List<GeoEntity> = emptyList(),
    val countriesRegionCounts: Map<String, Int> = emptyMap(),
    val regions: List<GeoEntity> = emptyList(),
    val selectedContinentName: String = "",
    val selectedCountryName: String = "",
    val selectedCountryCode: String? = null,
    val currentCountryCode: String? = null,
    val currentRegionCode: String? = null,
    val done: Boolean = false,
) {
    val breadcrumbs: List<BreadcrumbItem>
        get() = buildList {
            add(BreadcrumbItem("Все регионы", LocationStep.CONTINENTS))
            if (step == LocationStep.COUNTRIES || step == LocationStep.REGIONS) {
                add(BreadcrumbItem(selectedContinentName, LocationStep.COUNTRIES))
            }
            if (step == LocationStep.REGIONS) {
                add(BreadcrumbItem(selectedCountryName, LocationStep.REGIONS))
            }
        }
}

@HiltViewModel
class LocationPickerViewModel @Inject constructor(
    private val geoRepository: GeoRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(LocationPickerUiState())
    val uiState: StateFlow<LocationPickerUiState> = _uiState.asStateFlow()

    private val collator: Collator = Collator.getInstance()

    init {
        loadContinents()
    }

    private fun loadContinents() {
        viewModelScope.launch {
            val continents = geoRepository.getContinents()
            val currentCountry = geoRepository.countryCode.first()
            val currentRegion = geoRepository.regionCode.first()
            val counts = continents.associate { it.code to geoRepository.getChildrenCount(it.code) }
            _uiState.update {
                it.copy(
                    continents = continents,
                    continentCounts = counts,
                    currentCountryCode = currentCountry,
                    currentRegionCode = currentRegion,
                )
            }
        }
    }

    fun selectContinent(code: String, name: String) {
        viewModelScope.launch {
            val countries = geoRepository.getChildren(code)
                .sortedWith(compareBy(collator) { it.displayName() })
            val regionCounts = countries.associate { it.code to geoRepository.getChildrenCount(it.code) }
            _uiState.update {
                it.copy(
                    step = LocationStep.COUNTRIES,
                    countries = countries,
                    countriesRegionCounts = regionCounts,
                    selectedContinentName = name,
                )
            }
        }
    }

    fun selectCountry(entity: GeoEntity) {
        viewModelScope.launch {
            val regions = geoRepository.getChildren(entity.code)
                .sortedWith(compareBy(collator) { it.displayName() })
            if (regions.isNotEmpty()) {
                _uiState.update {
                    it.copy(
                        step = LocationStep.REGIONS,
                        regions = regions,
                        selectedCountryName = entity.displayName(),
                        selectedCountryCode = entity.code,
                    )
                }
            } else {
                geoRepository.selectCountry(entity.code)
                geoRepository.selectRegion(null)
                _uiState.update { it.copy(done = true) }
            }
        }
    }

    fun selectRegion(code: String?) {
        viewModelScope.launch {
            val countryCode = _uiState.value.selectedCountryCode ?: return@launch
            geoRepository.selectCountry(countryCode)
            geoRepository.selectRegion(code)
            _uiState.update { it.copy(done = true) }
        }
    }

    fun navigateToBreadcrumb(step: LocationStep) {
        when (step) {
            LocationStep.CONTINENTS -> _uiState.update {
                it.copy(step = LocationStep.CONTINENTS, countries = emptyList(), regions = emptyList())
            }
            LocationStep.COUNTRIES -> _uiState.update {
                it.copy(step = LocationStep.COUNTRIES, regions = emptyList())
            }
            LocationStep.REGIONS -> { /* already there */ }
        }
    }

    /** @return true if handled internally, false if should pop back stack */
    fun goBack(): Boolean {
        val current = _uiState.value.step
        return when (current) {
            LocationStep.REGIONS -> {
                _uiState.update { it.copy(step = LocationStep.COUNTRIES, regions = emptyList()) }
                true
            }
            LocationStep.COUNTRIES -> {
                _uiState.update { it.copy(step = LocationStep.CONTINENTS, countries = emptyList()) }
                true
            }
            LocationStep.CONTINENTS -> false
        }
    }
}
