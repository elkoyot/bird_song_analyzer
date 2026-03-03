package com.birdsong.analyzer.presentation.navigation

import kotlinx.serialization.Serializable

@Serializable
object HomeRoute

@Serializable
object LiveDetectionRoute

@Serializable
object HistoryRoute

@Serializable
object SettingsRoute

@Serializable
data class DetailRoute(val observationId: String)

@Serializable
object DualDetectionRoute

@Serializable
object FileAnalysisRoute
