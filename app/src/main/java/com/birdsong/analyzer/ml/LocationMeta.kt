package com.birdsong.analyzer.ml

data class LocationMeta(
    val latitude: Double,
    val longitude: Double,
    val weekOfYear: Int,
    // If set, meta-model runs for each week in range and max score is used.
    // Useful when recording date is unknown or to avoid filtering early migrants.
    val weekRange: IntRange? = null,
)
