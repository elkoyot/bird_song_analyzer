package com.birdsong.analyzer.presentation.navigation

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.detail.DetailScreen
import com.birdsong.analyzer.presentation.detection.LiveDetectionScreen
import com.birdsong.analyzer.presentation.detection.LiveDetectionViewModel
import com.birdsong.analyzer.presentation.history.HistoryScreen
import com.birdsong.analyzer.presentation.settings.SettingsScreen

private data class BottomNavItem<T : Any>(
    val route: T,
    val icon: ImageVector,
    val labelResId: Int,
)

private val bottomNavItems = listOf(
    BottomNavItem(LiveDetectionRoute, Icons.Default.GraphicEq, R.string.nav_detection),
    BottomNavItem(HistoryRoute, Icons.Default.History, R.string.nav_history),
    BottomNavItem(SettingsRoute, Icons.Default.Settings, R.string.nav_settings),
)

@Composable
fun BirdSongNavHost() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    val showBottomBar = bottomNavItems.any { item ->
        currentDestination?.hasRoute(item.route::class) == true
    }

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    bottomNavItems.forEach { item ->
                        val selected = currentDestination?.hasRoute(item.route::class) == true
                        NavigationBarItem(
                            selected = selected,
                            onClick = {
                                navController.navigate(item.route) {
                                    popUpTo(navController.graph.findStartDestination().id) {
                                        saveState = true
                                    }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = { Icon(item.icon, contentDescription = null) },
                            label = { Text(stringResource(item.labelResId)) },
                        )
                    }
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = LiveDetectionRoute,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable<LiveDetectionRoute> {
                val viewModel: LiveDetectionViewModel = hiltViewModel()
                val uiState by viewModel.uiState.collectAsStateWithLifecycle()
                val context = LocalContext.current

                val permissionLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { granted ->
                    if (granted) viewModel.onStart()
                }

                LiveDetectionScreen(
                    uiState = uiState,
                    onBirdClick = { observationId ->
                        navController.navigate(DetailRoute(observationId))
                    },
                    onStart = {
                        if (ContextCompat.checkSelfPermission(
                                context, Manifest.permission.RECORD_AUDIO,
                            ) == PackageManager.PERMISSION_GRANTED
                        ) {
                            viewModel.onStart()
                        } else {
                            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    },
                    onPause = viewModel::onPause,
                    onResume = viewModel::onResume,
                    onStop = viewModel::onStop,
                    onReset = viewModel::onReset,
                    onTestSample = viewModel::onTestSample,
                )
            }

            composable<HistoryRoute> {
                HistoryScreen(
                    onObservationClick = { observationId ->
                        navController.navigate(DetailRoute(observationId))
                    },
                )
            }

            composable<SettingsRoute> {
                SettingsScreen()
            }

            composable<DetailRoute> {
                DetailScreen(
                    onBack = { navController.popBackStack() },
                )
            }
        }
    }
}
