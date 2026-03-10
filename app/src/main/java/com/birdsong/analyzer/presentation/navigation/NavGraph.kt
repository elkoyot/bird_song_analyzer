package com.birdsong.analyzer.presentation.navigation

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.birdsong.analyzer.presentation.theme.HubColors
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navigation
import androidx.navigation.toRoute
import com.birdsong.analyzer.R
import com.birdsong.analyzer.presentation.detail.DetailScreen
import com.birdsong.analyzer.presentation.detail.DetailUiState
import com.birdsong.analyzer.presentation.detection.DualDetectionScreen
import com.birdsong.analyzer.presentation.detection.DualDetectionViewModel
import com.birdsong.analyzer.presentation.detection.FileAnalysisScreen
import com.birdsong.analyzer.presentation.detection.FileAnalysisViewModel
import com.birdsong.analyzer.presentation.detection.HomeScreen
import com.birdsong.analyzer.presentation.history.HistoryViewModel
import com.birdsong.analyzer.presentation.location.LocationPickerScreen
import com.birdsong.analyzer.presentation.location.LocationPickerViewModel
import com.birdsong.analyzer.presentation.settings.SettingsScreen
import com.birdsong.analyzer.presentation.settings.SettingsViewModel
import com.birdsong.analyzer.presentation.splash.PermissionScreen
import com.birdsong.analyzer.presentation.splash.SplashScreen
import com.birdsong.analyzer.presentation.splash.SplashViewModel

private data class BottomNavItem<T : Any>(
    val route: T,
    val icon: ImageVector,
    val labelResId: Int,
)

private val bottomNavItems = listOf(
    BottomNavItem(HomeRoute, Icons.Default.Mic, R.string.nav_listen),
    BottomNavItem(InfoRoute, Icons.Default.Info, R.string.nav_info),
    BottomNavItem(ProfileRoute, Icons.Default.Person, R.string.nav_profile),
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
                                    popUpTo<HomeRoute> {
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
            startDestination = SplashRoute,
            modifier = Modifier.padding(innerPadding),
        ) {
            // ── Onboarding (top-level, outside MainGraph) ────────────────

            composable<SplashRoute> {
                val viewModel: SplashViewModel = hiltViewModel()
                val state by viewModel.uiState.collectAsStateWithLifecycle()
                val context = LocalContext.current

                LaunchedEffect(state.done) {
                    if (state.done) {
                        val granted = ContextCompat.checkSelfPermission(
                            context, Manifest.permission.RECORD_AUDIO,
                        ) == PackageManager.PERMISSION_GRANTED

                        val target = if (granted) HomeRoute else PermissionRoute
                        navController.navigate(target) {
                            popUpTo(SplashRoute) { inclusive = true }
                        }
                    }
                }

                SplashScreen(state = state)
            }

            composable<PermissionRoute> {
                val context = LocalContext.current
                var denied by remember { mutableStateOf(false) }
                val lifecycleOwner = LocalLifecycleOwner.current

                // Auto-navigate when permission is granted via Settings and user returns
                LaunchedEffect(lifecycleOwner) {
                    lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                        val granted = ContextCompat.checkSelfPermission(
                            context, Manifest.permission.RECORD_AUDIO,
                        ) == PackageManager.PERMISSION_GRANTED
                        if (granted) {
                            navController.navigate(HomeRoute) {
                                popUpTo(PermissionRoute) { inclusive = true }
                            }
                        }
                    }
                }

                val launcher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) { granted ->
                    if (granted) {
                        navController.navigate(HomeRoute) {
                            popUpTo(PermissionRoute) { inclusive = true }
                        }
                    } else {
                        denied = true
                    }
                }

                PermissionScreen(
                    denied = denied,
                    onRequestPermission = {
                        if (denied) {
                            // Permanently denied — open system settings
                            val intent = Intent(
                                android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                Uri.fromParts("package", context.packageName, null),
                            )
                            context.startActivity(intent)
                        } else {
                            launcher.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    },
                    onSkip = { denied = true },
                )
            }

            // ── Main app (nested graph — bottom nav popUpTo targets this) ─

            navigation<MainGraph>(startDestination = HomeRoute) {

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
                    val historyViewModel: HistoryViewModel = hiltViewModel()
                    val analyses by historyViewModel.analyses.collectAsStateWithLifecycle()
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
                        analyses = analyses,
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
                        onLoadWaveform = viewModel::loadWaveform,
                        onPickLocation = { navController.navigate(LocationPickerRoute) },
                        onAnalysisClick = { analysisId ->
                            viewModel.loadFromHistory(analysisId)
                        },
                        onDeleteAnalysis = historyViewModel::deleteAnalysis,
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

                composable<InfoRoute> {
                    StubTabScreen(
                        icon = "ℹ️",
                        title = "Инфо / Справочник",
                        id = "SCR-04",
                        color = HubColors.Blue,
                    )
                }

                composable<ProfileRoute> {
                    StubTabScreen(
                        icon = "👤",
                        title = "Профиль",
                        id = "SCR-08",
                        color = HubColors.Accent,
                    )
                }
            }
        }
    }
}

@Composable
private fun StubTabScreen(icon: String, title: String, id: String, color: Color) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(HubColors.Bg),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(24.dp))
                    .background(color.copy(alpha = 0.1f))
                    .border(1.dp, color.copy(alpha = 0.2f), RoundedCornerShape(24.dp)),
                contentAlignment = Alignment.Center,
            ) {
                Text(text = icon, fontSize = 36.sp)
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = title,
                    color = HubColors.TextPrimary,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Экран в разработке",
                    color = HubColors.TextMuted,
                    fontSize = 13.sp,
                )
            }
            Box(
                modifier = Modifier
                    .background(HubColors.BgEl, RoundedCornerShape(14.dp))
                    .border(1.dp, color.copy(alpha = 0.27f), RoundedCornerShape(14.dp))
                    .padding(horizontal = 18.dp, vertical = 8.dp),
            ) {
                Text(
                    text = id,
                    color = color.copy(alpha = 0.53f),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = 1.2.sp,
                )
            }
        }
    }
}
