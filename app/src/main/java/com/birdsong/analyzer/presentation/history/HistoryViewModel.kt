package com.birdsong.analyzer.presentation.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.data.local.FileAnalysisSummary
import com.birdsong.analyzer.data.repository.FileAnalysisRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HistoryViewModel @Inject constructor(
    private val repository: FileAnalysisRepository,
) : ViewModel() {

    val analyses: StateFlow<List<FileAnalysisSummary>> =
        repository.getAllSummaries()
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun deleteAnalysis(id: String) {
        viewModelScope.launch {
            repository.deleteAnalysis(id)
        }
    }
}
