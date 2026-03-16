package com.birdsong.analyzer.service

import android.content.Context
import android.media.MediaPlayer
import android.net.Uri
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

enum class PlaybackState { IDLE, PLAYING, PAUSED }

@Singleton
class AudioPlaybackManager @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    private var mediaPlayer: MediaPlayer? = null
    private var positionJob: Job? = null
    private val scope = CoroutineScope(Dispatchers.Main)

    private val _state = MutableStateFlow(PlaybackState.IDLE)
    val state: StateFlow<PlaybackState> = _state.asStateFlow()

    private val _positionMs = MutableStateFlow(0L)
    val positionMs: StateFlow<Long> = _positionMs.asStateFlow()

    private val _durationMs = MutableStateFlow(0L)
    val durationMs: StateFlow<Long> = _durationMs.asStateFlow()

    fun play(uri: Uri) {
        release()
        try {
            mediaPlayer = MediaPlayer().apply {
                setDataSource(context, uri)
                prepare()
                _durationMs.value = duration.toLong()
                setOnCompletionListener {
                    _state.value = PlaybackState.IDLE
                    _positionMs.value = 0L
                    stopPositionUpdates()
                }
                start()
            }
            _state.value = PlaybackState.PLAYING
            startPositionUpdates()
        } catch (e: Exception) {
            Log.e(TAG, "play failed", e)
            release()
        }
    }

    fun resume() {
        mediaPlayer?.let {
            if (!it.isPlaying) {
                it.start()
                _state.value = PlaybackState.PLAYING
                startPositionUpdates()
            }
        }
    }

    fun pause() {
        mediaPlayer?.let {
            if (it.isPlaying) {
                it.pause()
                _state.value = PlaybackState.PAUSED
                stopPositionUpdates()
            }
        }
    }

    fun seekTo(positionMs: Long) {
        mediaPlayer?.let {
            it.seekTo(positionMs.toInt())
            _positionMs.value = positionMs
        }
    }

    fun seekToFraction(fraction: Float) {
        val dur = _durationMs.value
        if (dur > 0) seekTo((fraction * dur).toLong())
    }

    fun stop() {
        mediaPlayer?.let {
            if (it.isPlaying) it.stop()
        }
        _state.value = PlaybackState.IDLE
        _positionMs.value = 0L
        stopPositionUpdates()
    }

    fun release() {
        stopPositionUpdates()
        mediaPlayer?.let {
            try {
                if (it.isPlaying) it.stop()
                it.release()
            } catch (_: Exception) {}
        }
        mediaPlayer = null
        _state.value = PlaybackState.IDLE
        _positionMs.value = 0L
        _durationMs.value = 0L
    }

    private fun startPositionUpdates() {
        stopPositionUpdates()
        positionJob = scope.launch {
            while (true) {
                mediaPlayer?.let { _positionMs.value = it.currentPosition.toLong() }
                delay(100L)
            }
        }
    }

    private fun stopPositionUpdates() {
        positionJob?.cancel()
        positionJob = null
    }

    companion object {
        private const val TAG = "AudioPlaybackManager"
    }
}
