package com.birdsong.analyzer.data.repository

import com.birdsong.analyzer.data.PreferencesRepository
import com.birdsong.analyzer.data.local.GeoDao
import com.birdsong.analyzer.data.model.GeoEntity
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class GeoRepository @Inject constructor(
    private val geoDao: GeoDao,
    private val prefsRepo: PreferencesRepository,
) {
    suspend fun getContinents(): List<GeoEntity> = geoDao.getContinents()

    suspend fun getChildren(parentCode: String): List<GeoEntity> = geoDao.getChildren(parentCode)

    suspend fun getByCode(code: String): GeoEntity? = geoDao.getByCode(code)

    suspend fun resolveCurrentGeo(): GeoEntity? {
        val regionCode = prefsRepo.regionCode.first()
        val countryCode = prefsRepo.countryCode.first()
        val code = regionCode ?: countryCode
        return geoDao.getByCode(code)
    }

    val currentSelectionDisplay: Flow<String> = prefsRepo.countryCode
        .combine(prefsRepo.regionCode) { country, region ->
            val countryEntity = geoDao.getByCode(country)
            val regionEntity = region?.let { geoDao.getByCode(it) }
            when {
                regionEntity != null && countryEntity != null ->
                    "${countryEntity.displayName()} \u00b7 ${regionEntity.displayName()}"
                countryEntity != null -> countryEntity.displayName()
                else -> "\u2014"
            }
        }

    val countryCode: Flow<String> get() = prefsRepo.countryCode

    val regionCode: Flow<String?> get() = prefsRepo.regionCode

    suspend fun selectCountry(code: String) = prefsRepo.setCountry(code)

    suspend fun selectRegion(code: String?) = prefsRepo.setRegion(code)
}
