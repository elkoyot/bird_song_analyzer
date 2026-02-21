package com.birdsong.analyzer.ml

data class BirdDetection(
    val scientificName: String,
    val commonName: String,
    val confidence: Float,
    val labelIndex: Int,
)
