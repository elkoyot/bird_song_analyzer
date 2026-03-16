package com.birdsong.analyzer.ml

data class BirdDetection(
    val scientificName: String,
    val commonName: String,
    val confidence: Float,
    val labelIndex: Int,
    /** Taxonomic class: "Aves", "Mammalia", "Insecta", "Amphibia", "Squamata", or empty. */
    val taxonClass: String = "",
)
