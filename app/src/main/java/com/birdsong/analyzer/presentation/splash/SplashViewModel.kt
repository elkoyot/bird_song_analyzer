package com.birdsong.analyzer.presentation.splash

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.birdsong.analyzer.ml.BirdClassifier
import dagger.Lazy
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class SplashViewModel @Inject constructor(
    private val classifierLazy: Lazy<BirdClassifier>,
) : ViewModel() {

    data class UiState(
        val phase: Int = 0,       // 0=logo only, 1=+name, 2=+progress bar
        val progress: Float = 0f, // 0..1
        val done: Boolean = false,
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            // Start loading immediately in parallel with phase animations
            var loadComplete = false
            val loadJob = launch(Dispatchers.IO) {
                try {
                    classifierLazy.get()
                } catch (e: Exception) {
                    Log.e(TAG, "Classifier preload failed, will retry on first use", e)
                }
                loadComplete = true
            }

            // Phase animations run concurrently with loading
            delay(400)
            _uiState.update { it.copy(phase = 1) }
            delay(500)
            _uiState.update { it.copy(phase = 2) }

            // Animate progress: fills in ~2 s, caps at 88% until loading is done
            val animStart = System.currentTimeMillis()
            while (true) {
                delay(50)
                val elapsed = System.currentTimeMillis() - animStart
                val raw = elapsed / 2000f
                val capped = if (loadComplete) raw else raw.coerceAtMost(0.88f)
                val progress = capped.coerceAtMost(1f)
                _uiState.update { it.copy(progress = progress) }
                if (progress >= 1f) break
            }

            loadJob.join()
            delay(300)
            _uiState.update { it.copy(done = true) }
        }
    }

    companion object {
        private const val TAG = "SplashViewModel"
    }
}
