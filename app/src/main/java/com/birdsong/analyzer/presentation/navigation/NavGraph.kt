package com.birdsong.analyzer.presentation.navigation

import android.Manifest
import android.content.pm.PackageManager
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
import androidx.navigation.toRoute
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.detail.DetailScreen
import com.birdsong.analyzer.presentation.detail.DetailUiState
import com.birdsong.analyzer.presentation.detection.DualDetectionScreen
import com.birdsong.analyzer.presentation.detection.DualDetectionViewModel
import com.birdsong.analyzer.presentation.detection.FileAnalysisScreen
import com.birdsong.analyzer.presentation.detection.FileAnalysisViewModel
import com.birdsong.analyzer.presentation.detection.HomeScreen
import com.birdsong.analyzer.presentation.location.LocationPickerScreen
import com.birdsong.analyzer.presentation.location.LocationPickerViewModel
import com.birdsong.analyzer.presentation.settings.SettingsScreen
import com.birdsong.analyzer.presentation.settings.SettingsViewModel

private data class BottomNavItem<T : Any>(
    val route: T,
    val icon: ImageVector,
    val labelResId: Int,
)

private val bottomNavItems = listOf(
    BottomNavItem(HomeRoute, Icons.Default.Home, R.string.nav_home),
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
            startDestination = HomeRoute,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable<HomeRoute> {
                HomeScreen(
                    onNavigateToLiveDetection = { navController.navigate(LiveDetectionRoute) },
                    onNavigateToFileAnalysis = { navController.navigate(FileAnalysisRoute()) },
                )
            }

            composable<LiveDetectionRoute> {
                val viewModel: DualDetectionViewModel = hiltViewModel()
                val uiState by viewModel.uiState.collectAsStateWithLifecycle()
                val context = LocalContext.current

                val permissionLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { granted ->
                    if (granted) viewModel.onStart()
                }

                DualDetectionScreen(
                    uiState = uiState,
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
                    onBack = { navController.popBackStack() },
                )
            }

            composable<SettingsRoute> {
                val context = LocalContext.current
                val viewModel: SettingsViewModel = hiltViewModel()
                val locationLabel by viewModel.locationLabel.collectAsStateWithLifecycle()
                val activeModel by viewModel.activeModel.collectAsStateWithLifecycle()

                fun checkAudio() = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.RECORD_AUDIO,
                ) == PackageManager.PERMISSION_GRANTED

                fun checkLocation() = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.ACCESS_COARSE_LOCATION,
                ) == PackageManager.PERMISSION_GRANTED

                var audioGranted by remember { mutableStateOf(checkAudio()) }
                var locationGranted by remember { mutableStateOf(checkLocation()) }

                val audioLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { audioGranted = checkAudio() }

                val locationLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { locationGranted = checkLocation() }

                SettingsScreen(
                    audioPermissionGranted = audioGranted,
                    locationPermissionGranted = locationGranted,
                    locationLabel = locationLabel,
                    activeModel = activeModel,
                    isV30Available = viewModel.isV30Available,
                    onRequestAudioPermission = {
                        audioLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    },
                    onRequestLocationPermission = {
                        locationLauncher.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
                    },
                    onLocationClick = { navController.navigate(LocationPickerRoute) },
                    onModelSelected = viewModel::selectModel,
                )
            }

            composable<LocationPickerRoute> {
                val viewModel: LocationPickerViewModel = hiltViewModel()
                val uiState by viewModel.uiState.collectAsStateWithLifecycle()

                LaunchedEffect(uiState.done) {
                    if (uiState.done) navController.popBackStack()
                }

                LocationPickerScreen(
                    uiState = uiState,
                    onSelectContinent = viewModel::selectContinent,
                    onSelectCountry = viewModel::selectCountry,
                    onSelectRegion = viewModel::selectRegion,
                    onBack = { navController.popBackStack() },
                    onGoBack = viewModel::goBack,
                )
            }

            composable<FileAnalysisRoute> { backStackEntry ->
                val route = backStackEntry.toRoute<FileAnalysisRoute>()
                val viewModel: FileAnalysisViewModel = hiltViewModel()
                val uiState by viewModel.uiState.collectAsStateWithLifecycle()
                val recentAnalyses by viewModel.recentAnalyses.collectAsStateWithLifecycle()
                val context = LocalContext.current

                // Load from history if analysisId is provided
                LaunchedEffect(route.analysisId) {
                    route.analysisId?.let { viewModel.loadFromHistory(it) }
                }

                val filePickerLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.GetContent(),
                ) { uri ->
                    if (uri != null) {
                        val name = context.contentResolver
                            .query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                            ?.use { cursor ->
                                if (cursor.moveToFirst()) cursor.getString(0) else null
                            } ?: uri.lastPathSegment ?: "audio"
                        viewModel.selectFile(uri, name)
                    }
                }

                FileAnalysisScreen(
                    uiState = uiState,
                    recentAnalyses = recentAnalyses,
                    onSelectFile = { filePickerLauncher.launch("audio/*") },
                    onStartAnalysis = viewModel::startAnalysis,
                    onPause = viewModel::pauseAnalysis,
                    onResume = viewModel::resumeAnalysis,
                    onCancel = viewModel::cancelAnalysis,
                    onSelectSpecies = viewModel::selectSpecies,
                    onSpeciesClick = { sciName, commonName ->
                        navController.navigate(
                            DetailRoute(
                                commonName = commonName,
                                scientificName = sciName,
                            ),
                        )
                    },
                    onLoadFromHistory = { id ->
                        viewModel.loadFromHistory(id)
                    },
                    onDeleteHistory = viewModel::deleteFromHistory,
                    onPickLocation = { navController.navigate(LocationPickerRoute) },
                    onBack = { navController.popBackStack() },
                )
            }

            composable<DetailRoute> { backStackEntry ->
                val route = backStackEntry.toRoute<DetailRoute>()
                DetailScreen(
                    uiState = DetailUiState(
                        commonName = route.commonName,
                        scientificName = route.scientificName,
                        confidence = maxOf(
                            if (route.v24Confidence >= 0) route.v24Confidence else 0,
                            if (route.v30Confidence >= 0) route.v30Confidence else 0,
                        ),
                    ),
                    onBack = { navController.popBackStack() },
                )
            }
        }
    }
}
