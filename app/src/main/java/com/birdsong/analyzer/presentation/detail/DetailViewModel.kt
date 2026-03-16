package com.birdsong.analyzer.presentation.detail

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.navigation.toRoute
import com.birdsong.analyzer.data.local.SpeciesCard
import com.birdsong.analyzer.data.repository.SpeciesRepository
import com.birdsong.analyzer.presentation.navigation.DetailRoute
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class DetailViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val speciesRepository: SpeciesRepository,
) : ViewModel() {

    private val route = savedStateHandle.toRoute<DetailRoute>()

    private val _uiState = MutableStateFlow(
        DetailUiState(
            commonName = route.commonName,
            scientificName = route.scientificName,
            confidence = maxOf(
                if (route.v24Confidence >= 0) route.v24Confidence else 0,
                if (route.v30Confidence >= 0) route.v30Confidence else 0,
            ),
        ),
    )
    val uiState: StateFlow<DetailUiState> = _uiState

    init {
        loadSpeciesCard()
    }

    private fun loadSpeciesCard() {
        viewModelScope.launch {
            val card = speciesRepository.getSpeciesCard(route.scientificName) ?: return@launch
            _uiState.value = _uiState.value.copy(
                commonName = card.nameLocal ?: card.nameEn ?: route.commonName,
                scientificName = card.scientificName,
                orderName = card.orderLocal ?: card.orderLatin,
                familyName = card.familyLocal ?: card.familyLatin,
                genus = card.genus,
                taxonClass = card.taxonClass,
                iucnStatus = card.iucnStatus,
            )
        }
    }
}
