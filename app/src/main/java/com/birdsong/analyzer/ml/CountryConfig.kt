package com.birdsong.analyzer.ml

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class BoundingBox(
    val minLat: Float,
    val maxLat: Float,
    val minLon: Float,
    val maxLon: Float,
)

@Serializable
data class CountryConfig(
    val code: String,
    val nameRu: String,
    val nameEn: String,
    val bbox: BoundingBox,
    val bufferDeg: Float = 2.5f,
    val regions: List<CountryConfig> = emptyList(),
)

fun CountryConfig.displayName(): String = nameRu  // TODO: locale-aware when i18n lands

object CountryConfigLoader {

    private val json = Json { ignoreUnknownKeys = true }

    fun load(context: Context): List<CountryConfig> {
        val text = context.assets
            .open("${ASSET_PATH}/countries.json")
            .bufferedReader()
            .use { it.readText() }
        return json.decodeFromString<List<CountryConfig>>(text)
    }

    fun findByCode(
        countries: List<CountryConfig>,
        countryCode: String,
        regionCode: String? = null,
    ): CountryConfig? {
        val country = countries.find { it.code == countryCode } ?: return null
        if (regionCode == null || country.regions.isEmpty()) return country
        return country.regions.find { it.code == regionCode } ?: country
    }

    private const val ASSET_PATH = "birdnet/v24"
}
