package com.birdsong.analyzer.data.local

import android.content.Context
import android.util.Log
import com.birdsong.analyzer.data.model.GeoEntity
import com.birdsong.analyzer.data.model.GeoModelEntity
import com.birdsong.analyzer.data.model.MlModelEntity
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

object GeoSeedLoader {

    private const val TAG = "GeoSeedLoader"
    private const val SEED_ASSET = "geo/seed.json"

    private val json = Json { ignoreUnknownKeys = true }

    suspend fun seed(dao: GeoDao, context: Context) {
        if (dao.count() > 0) {
            Log.d(TAG, "Database already seeded, skipping")
            return
        }

        val text = context.assets.open(SEED_ASSET).bufferedReader().use { it.readText() }
        val seed = json.decodeFromString<SeedData>(text)

        val geoEntities = mutableListOf<GeoEntity>()
        val geoModels = mutableListOf<GeoModelEntity>()

        // Flatten continents → countries → regions
        seed.continents.forEachIndexed { idx, continent ->
            geoEntities.add(
                GeoEntity(
                    code = continent.code,
                    type = "continent",
                    parentCode = null,
                    nameRu = continent.nameRu,
                    nameEn = continent.nameEn,
                    minLat = null,
                    maxLat = null,
                    minLon = null,
                    maxLon = null,
                    sortOrder = idx,
                ),
            )

            continent.countries.forEach { country ->
                geoEntities.add(
                    GeoEntity(
                        code = country.code,
                        type = "country",
                        parentCode = continent.code,
                        nameRu = country.nameRu,
                        nameEn = country.nameEn,
                        minLat = country.bbox.minLat,
                        maxLat = country.bbox.maxLat,
                        minLon = country.bbox.minLon,
                        maxLon = country.bbox.maxLon,
                        bufferDeg = country.bufferDeg,
                    ),
                )

                country.regions.forEach { region ->
                    geoEntities.add(
                        GeoEntity(
                            code = region.code,
                            type = "region",
                            parentCode = country.code,
                            nameRu = region.nameRu,
                            nameEn = region.nameEn,
                            minLat = region.bbox.minLat,
                            maxLat = region.bbox.maxLat,
                            minLon = region.bbox.minLon,
                            maxLon = region.bbox.maxLon,
                            bufferDeg = country.bufferDeg,
                        ),
                    )
                }
            }
        }

        // Models
        val models = seed.models.map { m ->
            MlModelEntity(
                id = m.id,
                name = m.name,
                runtime = m.runtime,
                audioPath = m.audioPath,
                metaModelPath = m.metaModelPath,
                labelsPath = m.labelsPath,
                sampleRate = m.sampleRate,
                chunkSeconds = m.chunkSeconds,
                speciesCount = m.speciesCount,
                isBundled = m.isBundled,
            )
        }

        // Geo-model mappings
        seed.geoModels.forEach { mapping ->
            geoModels.add(
                GeoModelEntity(
                    geoCode = mapping.geoCode,
                    modelId = mapping.modelId,
                    isDefault = mapping.isDefault,
                ),
            )
        }

        // Insert in order: models first (FK), then geo entities (self-ref FK: continents before countries), then junction
        dao.insertModels(models)
        // Insert continents first, then countries, then regions (for FK constraint)
        dao.insertGeoEntities(geoEntities.filter { it.type == "continent" })
        dao.insertGeoEntities(geoEntities.filter { it.type == "country" })
        dao.insertGeoEntities(geoEntities.filter { it.type == "region" })
        dao.insertGeoModels(geoModels)

        Log.i(TAG, "Seeded ${geoEntities.size} geo entities, ${models.size} models, ${geoModels.size} mappings")
    }

    // --- Seed JSON schema ---

    @Serializable
    private data class SeedData(
        val continents: List<SeedContinent>,
        val models: List<SeedModel>,
        val geoModels: List<SeedGeoModel>,
    )

    @Serializable
    private data class SeedContinent(
        val code: String,
        val nameRu: String,
        val nameEn: String,
        val countries: List<SeedCountry>,
    )

    @Serializable
    private data class SeedCountry(
        val code: String,
        val nameRu: String,
        val nameEn: String,
        val bbox: SeedBbox,
        val bufferDeg: Float = 2.5f,
        val regions: List<SeedRegion> = emptyList(),
    )

    @Serializable
    private data class SeedRegion(
        val code: String,
        val nameRu: String,
        val nameEn: String,
        val bbox: SeedBbox,
    )

    @Serializable
    private data class SeedBbox(
        val minLat: Float,
        val maxLat: Float,
        val minLon: Float,
        val maxLon: Float,
    )

    @Serializable
    private data class SeedModel(
        val id: String,
        val name: String,
        val runtime: String,
        val audioPath: String,
        val metaModelPath: String? = null,
        val labelsPath: String,
        val sampleRate: Int,
        val chunkSeconds: Int,
        val speciesCount: Int,
        val isBundled: Boolean,
    )

    @Serializable
    private data class SeedGeoModel(
        val geoCode: String,
        val modelId: String,
        val isDefault: Boolean = false,
    )
}
