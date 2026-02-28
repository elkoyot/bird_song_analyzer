package com.birdsong.analyzer.presentation.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.PreferencesRepository
import com.birdsong.analyzer.ml.ClassifierFactory
import com.birdsong.analyzer.ml.CountryConfig
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val prefsRepo: PreferencesRepository,
    private val classifierFactory: ClassifierFactory,
    val countries: List<CountryConfig>,
) : ViewModel() {

    val selectedCountry: StateFlow<CountryConfig?> = prefsRepo.countryCode
        .map { code -> countries.find { it.code == code } }
        .stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5_000),
            countries.find { it.code == PreferencesRepository.DEFAULT_COUNTRY },
        )

    val selectedRegion: StateFlow<CountryConfig?> = prefsRepo.countryCode
        .combine(prefsRepo.regionCode) { code, regionCode ->
            val country = countries.find { it.code == code } ?: return@combine null
            regionCode?.let { rCode -> country.regions.find { it.code == rCode } }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val activeModel: StateFlow<String> = prefsRepo.activeModel
        .stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5_000),
            ClassifierFactory.MODEL_BIRDNET,
        )

    val isV30Available: Boolean = classifierFactory.isBirdNetV30Available()

    fun selectCountry(code: String) {
        viewModelScope.launch { prefsRepo.setCountry(code) }
    }

    fun selectRegion(code: String?) {
        viewModelScope.launch { prefsRepo.setRegion(code) }
    }

    fun selectModel(modelId: String) {
        viewModelScope.launch { prefsRepo.setActiveModel(modelId) }
    }
}
