package com.birdsong.analyzer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.birdsong.analyzer.presentation.navigation.BirdSongNavHost
import com.birdsong.analyzer.presentation.theme.BirdSongTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            BirdSongTheme {
                BirdSongNavHost()
            }
        }
    }
}
