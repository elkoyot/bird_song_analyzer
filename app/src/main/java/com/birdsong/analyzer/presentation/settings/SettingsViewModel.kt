package com.birdsong.analyzer.presentation.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.PreferencesRepository
import com.birdsong.analyzer.data.repository.GeoRepository
import com.birdsong.analyzer.ml.ClassifierFactory
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val prefsRepo: PreferencesRepository,
    private val classifierFactory: ClassifierFactory,
    private val geoRepository: GeoRepository,
) : ViewModel() {

    val locationLabel: StateFlow<String> = geoRepository.currentSelectionDisplay
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), "\u2014")

    val activeModel: StateFlow<String> = prefsRepo.activeModel
        .stateIn(
            viewModelScope,
            SharingStarted.WhileSubscribed(5_000),
            ClassifierFactory.MODEL_BIRDNET,
        )

    val isV30Available: Boolean = classifierFactory.isBirdNetV30Available()

    fun selectModel(modelId: String) {
        viewModelScope.launch { prefsRepo.setActiveModel(modelId) }
    }
}
