package com.birdsong.analyzer.presentation.navigation

import kotlinx.serialization.Serializable

@Serializable
object SplashRoute

@Serializable
object PermissionRoute

@Serializable
object MainGraph

@Serializable
object HomeRoute

@Serializable
object LiveDetectionRoute

@Serializable
object SettingsRoute

@Serializable
data class DetailRoute(
    val commonName: String = "",
    val scientificName: String = "",
    val v24Confidence: Int = -1,
    val v30Confidence: Int = -1,
    val detectionCount: Int = 0,
)

@Serializable
object DualDetectionRoute

@Serializable
data class FileAnalysisRoute(val analysisId: String? = null)

@Serializable
object LocationPickerRoute

@Serializable
object HistoryRoute

@Serializable
object InfoRoute

@Serializable
object ProfileRoute
